import pytest

from saksi.blobs import BlobStore
from saksi.llm import LLMError, ModelClient


class FakeTransport:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return {"choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3}}


MSG = [{"role": "user", "content": "hi"}]


def test_meta_request_shape_and_record(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "sk-SECRET-123")
    t = FakeTransport()
    c = ModelClient(BlobStore(tmp_path), transport=t, mode="live")
    r = c.chat("meta", MSG, purpose="test", reasoning_effort="medium", json_mode=True, seed=7)
    body = t.calls[0]["json"]
    assert t.calls[0]["url"] == "https://api.meta.ai/v1/chat/completions"
    assert body["model"] == "muse-spark-1.3" and body["max_completion_tokens"] == 4096
    assert body["reasoning_effort"] == "medium" and body["response_format"] == {"type": "json_object"}
    assert r.text == "hello" and r.record.prompt_tokens == 11 and r.record.cost_usd is None
    assert c.blobs.get_json(r.record.request_sha256)["body"] == body
    assert t.calls[0]["headers"]["Authorization"] == "Bearer sk-SECRET-123"
    assert "sk-SECRET-123" not in str(r.record.to_payload())
    assert "sk-SECRET-123" not in str(c.blobs.get_json(r.record.request_sha256))


def test_baseten_uses_open_weights_and_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "k")
    t = FakeTransport()
    r = ModelClient(BlobStore(tmp_path), transport=t, mode="live").chat("baseten", MSG, purpose="t",
                                                                         reasoning_effort="high")
    body = t.calls[0]["json"]
    assert body["model"] == "meta-models/muse-spark-open-1026" and "max_tokens" in body
    assert "reasoning_effort" not in body and r.record.open_weights


def test_refuses_training_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "k")
    c = ModelClient(BlobStore(tmp_path), transport=FakeTransport(), mode="live")
    with pytest.raises(LLMError, match="trained"):
        c.chat("meta", MSG, purpose="t", model="muse-spark-1.3-contributor")


def test_record_then_replay_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "k")
    blobs = BlobStore(tmp_path)
    ModelClient(blobs, transport=FakeTransport(), mode="record").chat("meta", MSG, purpose="t")
    monkeypatch.delenv("META_API_KEY")

    class NoNetwork:
        def post(self, *a, **k):
            raise AssertionError("replay must not call out")

    r = ModelClient(blobs, transport=NoNetwork(), mode="replay").chat("meta", MSG, purpose="t")
    assert r.text == "hello" and r.record.mode == "replay"
    with pytest.raises(LLMError, match="no recorded response"):
        ModelClient(blobs, transport=NoNetwork(), mode="replay").chat("meta", [{"role": "user", "content": "new"}],
                                                                      purpose="t")


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(LLMError, match="not set"):
        ModelClient(BlobStore(tmp_path), transport=FakeTransport(), mode="live").chat("meta", MSG, purpose="t")


def test_cost_only_when_prices_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "k")
    monkeypatch.setenv("SAKSI_PRICE_META_INPUT_PER_M", "1.0")
    monkeypatch.setenv("SAKSI_PRICE_META_OUTPUT_PER_M", "2.0")
    r = ModelClient(BlobStore(tmp_path), transport=FakeTransport(), mode="live").chat("meta", MSG, purpose="t")
    assert r.record.cost_usd == pytest.approx((11 * 1 + 3 * 2) / 1e6)


def test_split_reasoning_variants():
    from saksi.llm import split_reasoning
    assert split_reasoning({"content": "<think>hmm</think>\nsaksi ok"}) == ("saksi ok", "hmm")
    assert split_reasoning({"content": "hmm</think>saksi ok"}) == ("saksi ok", "hmm")
    assert split_reasoning({"content": "saksi ok", "reasoning_content": "hmm"}) == ("saksi ok", "hmm")
    assert split_reasoning({"content": "saksi ok"}) == ("saksi ok", None)


def test_reasoning_only_reply_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "k")

    class ThinkOnly:
        def post(self, *a, **k):
            return {"choices": [{"message": {"content": "<think>still thinking"}, "finish_reason": "length"}],
                    "usage": {}}

    with pytest.raises(LLMError, match="no answer text.*length"):
        ModelClient(BlobStore(tmp_path), transport=ThinkOnly(), mode="live").chat("baseten", MSG, purpose="t")
