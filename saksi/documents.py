# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Plain-text extraction from protocol documents (txt, md, pdf, docx)."""

from __future__ import annotations

import io
import re


class UnsupportedDocument(ValueError):
    pass


def extract_text(data: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    raise UnsupportedDocument(f"cannot read {filename}; use .txt, .md, .pdf or .docx")


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace(" ", " ")
    text = re.sub(r"-\s*\n\s*", "", text)          # rejoin words hyphenated across lines (PDFs)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_found(quote: str, source: str) -> bool:
    q = normalise(quote).strip(" .\"'")
    return len(q) >= 12 and q in normalise(source)
