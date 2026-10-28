# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Deploy Saksi on Modal.

  modal secret create saksi-keys META_API_KEY=... BASETEN_API_KEY=... SAKSI_CREATE_TOKEN=...
  modal run modal_app.py::smoke      # check both Meta model endpoints
  modal deploy modal_app.py          # serve the API

Study materials and the event logs live on a private Modal Volume. API keys
live in a Modal Secret and never reach the browser.
"""

from __future__ import annotations

import modal

app = modal.App("saksi")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.162.0", "httpx==0.28.1", "pydantic==2.13.5", "python-multipart",
        "numpy", "pandas", "scipy", "statsmodels", "scikit-learn",
    )
    .add_local_python_source("saksi")
)

volume = modal.Volume.from_name("saksi-data", create_if_missing=True)
keys = modal.Secret.from_name("saksi-keys")
DATA = "/data"


@app.function(image=image, volumes={DATA: volume}, secrets=[keys],
              max_containers=1,          # one writer per event log
              timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from saksi.api import create_app
    return create_app(f"{DATA}/projects", on_write=volume.commit)


@app.function(image=image, secrets=[keys], timeout=300)
def smoke() -> list[dict]:
    """One tiny call to each Meta model endpoint. Prints model, tokens and latency only."""
    import tempfile

    from saksi.blobs import BlobStore
    from saksi.llm import ModelClient

    client = ModelClient(BlobStore(tempfile.mkdtemp()), mode="live")
    msg = [{"role": "user", "content": "Reply with exactly: saksi ok"}]
    results = []
    for provider in ("meta", "baseten"):
        try:
            # Generous cap: Muse Spark Open always reasons before it answers.
            r = client.chat(provider, msg, purpose="smoke", max_tokens=2048)
            results.append({"provider": provider, "model": r.record.model, "reply": r.text.strip()[:40],
                            "tokens": (r.record.prompt_tokens, r.record.completion_tokens),
                            "latency_ms": r.record.latency_ms})
        except Exception as e:  # report, don't hide
            results.append({"provider": provider, "error": str(e)[:300]})
    for row in results:
        print(row)
    return results
