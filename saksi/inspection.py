# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""What Saksi could read from an uploaded file, checked at upload so authors see it at once.
Datasets: rows and columns only, never values. Documents: how much text, and its opening lines."""

from __future__ import annotations

import re
from typing import Any

from .documents import UnsupportedDocument, extract_text
from .profiling import read_table

DOCUMENT_KINDS = {"protocol", "codebook", "ethics_approval", "registration", "other"}
LITTLE_TEXT = 300


def inspect_material(kind: str, filename: str, data: bytes) -> dict[str, Any]:
    try:
        if kind == "dataset":
            df = read_table(data, filename)
            return {"readable": True, "rows": int(len(df)), "columns": int(len(df.columns))}
        if kind in DOCUMENT_KINDS:
            text = extract_text(data, filename)
            body = re.sub(r"\s+", " ", text).strip()
            out = {"readable": len(body) >= LITTLE_TEXT, "characters": len(body), "preview": body[:700]}
            if filename.lower().endswith(".pdf"):
                out["pages"] = text.count("\n\n") + 1 if text else 0
            if len(body) < LITTLE_TEXT:
                out["warning"] = ("Almost no text could be read from this file. If it is a scanned PDF, upload a "
                                  "Word, text or text-based PDF version, or Saksi cannot check quotations against it.")
            return out
        return {"readable": None}
    except UnsupportedDocument as e:
        return {"readable": False, "error": str(e)}
    except Exception as e:  # report, don't hide
        return {"readable": False, "error": f"{type(e).__name__}: {str(e)[:240]}"}
