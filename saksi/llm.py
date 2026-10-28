"""Model adapter for Meta models (Meta Model API and Muse Spark Open on Baseten).

Both providers speak the OpenAI Chat Completions protocol, so one client
serves both; only base URL, key and a few parameter names differ.

Every call is recorded: the full request and response go to the blob store,
and the caller receives a `CallRecord` (hashes, model, parameters, token
usage, latency) to attach to the event that the call produced.

Modes (env SAKSI_LLM_MODE):
  live   – call the provider (default)
  record – call the provider and keep the response for replay
  replay – never call out; answer from recorded responses (demo safety)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx

from .blobs import BlobStore
from .canonical import sha256_obj


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    default_model: str
    max_tokens_param: str          # parameter name for the output-token cap
    supports_reasoning_effort: bool
    open_weights: bool


PROVIDERS: dict[str, Provider] = {
    "meta": Provider(
        name="meta",
        base_url=os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1"),
        key_env="META_API_KEY",
        default_model="muse-spark-1.3",
        max_tokens_param="max_completion_tokens",
        supports_reasoning_effort=True,
        open_weights=False,
    ),
    "baseten": Provider(
        name="baseten",
        base_url=os.environ.get("BASETEN_API_BASE_URL", "https://inference.baseten.co/v1"),
        key_env="BASETEN_API_KEY",
        default_model="meta-models/muse-spark-open-1026",
        max_tokens_param="max_tokens",
        supports_reasoning_effort=False,
        open_weights=True,
    ),
}

# Model variants whose providers may train on submitted content. Saksi handles
# unpublished manuscripts and study materials, so these are refused outright.
REFUSED_MODEL_SUFFIXES = ("-contributor",)


class LLMError(RuntimeError):
    pass


@dataclass
class CallRecord:
    provider: str
    model: str
    open_weights: bool
    purpose: str
    params: dict[str, Any]
    request_sha256: str
    response_sha256: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    mode: str
    cost_usd: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResult:
    text: str
    record: CallRecord
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    reasoning: str | None = field(repr=False, default=None)


_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_reasoning(message: dict[str, Any]) -> tuple[str, str | None]:
    """Separate a reasoning model's thinking from its answer.

    Providers expose reasoning either as a separate field or inline in
    <think>…</think> tags. Only the answer is used downstream; the reasoning
    is kept in the stored response for audit.
    """
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    inline = _THINK.findall(content)
    if inline:
        content = _THINK.sub("", content)
        reasoning = "\n".join(filter(None, [reasoning, *inline]))
    elif "</think>" in content:            # opening tag sometimes omitted
        head, _, content = content.partition("</think>")
        reasoning = "\n".join(filter(None, [reasoning, head.replace("<think>", "")]))
    elif content.lstrip().startswith("<think>"):   # cut off mid-thought: no answer yet
        reasoning = "\n".join(filter(None, [reasoning, content.lstrip()[len("<think>"):]]))
        content = ""
    return content.strip(), (reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None)


class Transport(Protocol):
    def post(self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> dict[str, Any]: ...


class HttpxTransport:
    def post(self, url, headers, json, timeout):
        r = httpx.post(url, headers=headers, json=json, timeout=timeout)
        if r.status_code >= 400:
            # Never echo headers: they carry the key.
            raise LLMError(f"{url} returned HTTP {r.status_code}: {r.text[:400]}")
        return r.json()


def _price(provider: str, kind: str) -> float | None:
    """USD per million tokens, only if configured. No built-in guesses."""
    v = os.environ.get(f"SAKSI_PRICE_{provider.upper()}_{kind.upper()}_PER_M")
    return float(v) if v else None


class ModelClient:
    def __init__(self, blobs: BlobStore, transport: Transport | None = None, mode: str | None = None):
        self.blobs = blobs
        self.transport = transport or HttpxTransport()
        self.mode = mode or os.environ.get("SAKSI_LLM_MODE", "live")
        if self.mode not in {"live", "record", "replay"}:
            raise ValueError(f"unknown SAKSI_LLM_MODE {self.mode!r}")

    def chat(
        self,
        provider: str,
        messages: list[dict[str, str]],
        *,
        purpose: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        timeout: float = 180.0,
    ) -> LLMResult:
        prov = PROVIDERS[provider]
        model = model or prov.default_model
        if model.endswith(REFUSED_MODEL_SUFFIXES):
            raise LLMError(f"model {model!r} may be trained on submitted content; Saksi does not use it")

        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                prov.max_tokens_param: max_tokens}
        if reasoning_effort and prov.supports_reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if seed is not None:
            body["seed"] = seed

        request = {"provider": provider, "url": f"{prov.base_url}/chat/completions", "body": body}
        request_sha = self.blobs.put_json(request)
        replay_key = f"replay:{request_sha}"

        latency_ms: int | None = None
        if self.mode == "replay":
            idx = self._replay_index_get(replay_key)
            if idx is None:
                raise LLMError(f"replay mode: no recorded response for request {request_sha[:12]} ({purpose})")
            raw = self.blobs.get_json(idx)
        else:
            key = os.environ.get(prov.key_env)
            if not key:
                raise LLMError(f"{prov.key_env} is not set")
            t0 = time.monotonic()
            raw = self.transport.post(request["url"], {"Authorization": f"Bearer {key}",
                                                       "Content-Type": "application/json"}, body, timeout)
            latency_ms = int((time.monotonic() - t0) * 1000)

        response_sha = self.blobs.put_json(raw)
        if self.mode == "record":
            self._replay_index_put(replay_key, response_sha)

        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape from {provider}: {e}") from e
        text, reasoning = split_reasoning(message)
        if not text.strip():
            finish = raw["choices"][0].get("finish_reason")
            raise LLMError(f"{provider} returned no answer text (finish_reason={finish}); "
                           "reasoning models may need a larger max_tokens")

        usage = raw.get("usage") or {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        pin, pout = _price(provider, "input"), _price(provider, "output")
        cost = (pt * pin + ct * pout) / 1e6 if None not in (pt, ct, pin, pout) else None

        record = CallRecord(
            provider=provider, model=model, open_weights=prov.open_weights, purpose=purpose,
            params={k: v for k, v in body.items() if k not in ("messages", "model")},
            request_sha256=request_sha, response_sha256=response_sha,
            prompt_tokens=pt, completion_tokens=ct, latency_ms=latency_ms, mode=self.mode, cost_usd=cost,
        )
        return LLMResult(text=text, record=record, raw=raw, reasoning=reasoning)

    # A tiny replay index stored alongside the blobs.
    def _replay_index_path(self, key: str):
        return self.blobs.root / "replay" / sha256_obj(key)

    def _replay_index_get(self, key: str) -> str | None:
        p = self._replay_index_path(key)
        return p.read_text().strip() if p.exists() else None

    def _replay_index_put(self, key: str, digest: str) -> None:
        p = self._replay_index_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(digest)
