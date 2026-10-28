# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Content-addressed blob store.

Large or sensitive content (model prompts and responses, generated code,
analysis outputs, draft text) is stored once under its SHA-256 and the event
log records only the hash. The log stays small and readable while still
pinning exactly what was used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_bytes, sha256_text


class BlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def put_bytes(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        p = self._path(digest)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
        return digest

    def put_text(self, text: str) -> str:
        digest = self.put_bytes(text.encode("utf-8"))
        assert digest == sha256_text(text)
        return digest

    def put_json(self, obj: Any) -> str:
        return self.put_text(canonical_json(obj))

    def has(self, digest: str) -> bool:
        return self._path(digest).exists()

    def get_bytes(self, digest: str) -> bytes:
        data = self._path(digest).read_bytes()
        if sha256_bytes(data) != digest:
            raise ValueError(f"blob {digest[:12]} does not match its hash")
        return data

    def get_text(self, digest: str) -> str:
        return self.get_bytes(digest).decode("utf-8")

    def get_json(self, digest: str) -> Any:
        return json.loads(self.get_text(digest))
