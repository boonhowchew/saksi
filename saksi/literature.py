# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Literature: real records from PubMed, cited by code, formatted by code.

The Writing Agent may cite only records Saksi fetched from PubMed (NCBI E-utilities) for this paper, as [R:<PMID>].
What a record is said to have found must be backed by a quotation from its abstract, which code checks. Reference
lists are numbered and formatted by code in the style the authors choose (Vancouver, AMA, APA 7, Harvard), so no
model ever types a reference.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Callable

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
REF_TOKEN = re.compile(r"\[R:([0-9]{4,9})\]")
REF_GROUP = re.compile(r"(?:\s*\[R:[0-9]{4,9}\])+")

STYLES = {
    "vancouver": "Vancouver (ICMJE; BMJ, Lancet and most medical journals)",
    "ama": "AMA 11th edition (JAMA network)",
    "apa": "APA 7th edition (author-date)",
    "harvard": "Harvard (author-date)",
}
NUMBERED = {"vancouver", "ama"}


def _get(url: str, params: dict[str, Any], timeout: float = 30.0) -> str:
    r = httpx.get(url, params={**params, "tool": "saksi"}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"PubMed answered HTTP {r.status_code}")
    return r.text


class PubMed:
    """Search and fetch PubMed records. `get` is injectable so tests and the offline server need no network."""

    def __init__(self, get: Callable[[str, dict[str, Any]], str] | None = None):
        self.get = get or _get

    def search(self, term: str, retmax: int = 8) -> list[str]:
        xml = self.get(EUTILS + "esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax, "sort": "relevance"})
        return [e.text for e in ET.fromstring(xml).iter("Id") if e.text][:retmax]

    def fetch(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []
        xml = self.get(EUTILS + "efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        return [r for r in (parse_article(a) for a in ET.fromstring(xml).iter("PubmedArticle")) if r]

    def mesh_term(self, term: str) -> str | None:
        """The MeSH descriptor a keyword matches, or None. Checked against NCBI's MeSH database."""
        xml = self.get(EUTILS + "esearch.fcgi", {"db": "mesh", "term": f'"{term}"[MeSH Terms]', "retmax": 1})
        n = ET.fromstring(xml).findtext("Count") or "0"
        return term if int(n) > 0 else None

    def by_doi(self, doi: str) -> list[dict[str, Any]]:
        return self.fetch(self.search(f"{doi}[doi]", 1))


def _text(e: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(e.itertext())).strip() if e is not None else ""


def parse_article(a: ET.Element) -> dict[str, Any] | None:
    pmid = _text(a.find(".//MedlineCitation/PMID"))
    art = a.find(".//Article")
    if not pmid or art is None:
        return None
    authors = []
    for au in art.findall(".//AuthorList/Author"):
        if au.find("CollectiveName") is not None:
            authors.append({"collective": _text(au.find("CollectiveName"))})
        elif au.find("LastName") is not None:
            authors.append({"last": _text(au.find("LastName")), "initials": _text(au.find("Initials"))})
    j = art.find("Journal")
    year = _text(j.find(".//PubDate/Year")) if j is not None else ""
    if not year and j is not None:
        year = (re.findall(r"\d{4}", _text(j.find(".//PubDate/MedlineDate"))) or [""])[0]
    doi = next((_text(x) for x in a.findall(".//ArticleIdList/ArticleId") if x.get("IdType") == "doi"), "") or \
        next((_text(x) for x in art.findall("ELocationID") if x.get("EIdType") == "doi"), "")
    abstract = " ".join(_text(x) for x in art.findall(".//Abstract/AbstractText"))
    return {"pmid": pmid, "title": _text(art.find("ArticleTitle")).rstrip("."), "authors": authors,
            "journal": _text(j.find("ISOAbbreviation")) if j is not None else "",
            "journal_full": _text(j.find("Title")) if j is not None else "", "year": year,
            "volume": _text(j.find(".//JournalIssue/Volume")) if j is not None else "",
            "issue": _text(j.find(".//JournalIssue/Issue")) if j is not None else "",
            "pages": _text(art.find(".//Pagination/MedlinePgn")), "doi": doi, "abstract": abstract[:3000]}


# -- Formatting ----------------------------------------------------------------
def _names(rec: dict[str, Any], style: str) -> str:
    aus = rec.get("authors") or []
    def one(a: dict[str, Any]) -> str:
        if a.get("collective"):
            return a["collective"]
        ini = a.get("initials", "")
        if style in ("apa", "harvard"):
            return f"{a['last']}, {'. '.join(ini)}." if ini else a["last"]
        return f"{a['last']} {ini}".strip()
    if style == "vancouver":
        names = [one(a) for a in aus[:6]] + (["et al"] if len(aus) > 6 else [])
        return ", ".join(names)
    if style == "ama":
        names = [one(a) for a in aus[:6]] if len(aus) <= 6 else [one(a) for a in aus[:3]] + ["et al"]
        return ", ".join(names)
    if style == "apa":
        names = [one(a) for a in aus[:20]]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1] if names else ""
    names = [one(a) for a in aus]
    if len(names) > 3:
        return names[0] + " et al."
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


def format_reference(rec: dict[str, Any], style: str) -> str:
    """One reference, as the style prints it (plain text with *italics* marked)."""
    au, t, j, y = _names(rec, style), rec.get("title", ""), rec.get("journal") or rec.get("journal_full", ""), rec.get("year", "")
    vol, iss, pg, doi = rec.get("volume", ""), rec.get("issue", ""), rec.get("pages", ""), rec.get("doi", "")
    vi = f"{vol}({iss})" if iss else vol
    if style == "vancouver":
        out = f"{au}. {t}. {j}. {y}" + (f";{vi}" if vi else "") + (f":{pg}" if pg else "") + "."
        return out + (f" doi:{doi}" if doi else "")
    if style == "ama":
        out = f"{au}. {t}. *{j}*. {y}" + (f";{vi}" if vi else "") + (f":{pg}" if pg else "") + "."
        return out + (f" doi:{doi}" if doi else "")
    full = rec.get("journal_full") or j
    if style == "apa":
        out = f"{au} ({y}). {t}. *{full}*" + (f", *{vol}*" if vol else "") + (f"({iss})" if iss else "") + (f", {pg}" if pg else "") + "."
        return out + (f" https://doi.org/{doi}" if doi else "")
    out = f"{au} ({y}) '{t}', *{full}*" + (f", {vol}" if vol else "") + (f"({iss})" if iss else "") + (f", pp. {pg}" if pg else "") + "."
    return out + (f" doi:{doi}" if doi else "")


def _short(rec: dict[str, Any], style: str) -> str:
    aus = rec.get("authors") or []
    first = (aus[0].get("last") or aus[0].get("collective")) if aus else "Anonymous"
    if len(aus) == 2 and not aus[0].get("collective"):
        first = f"{first} {'&' if style == 'apa' else 'and'} {aus[1].get('last') or aus[1].get('collective')}"
    elif len(aus) > 2:
        first += " et al."
    return f"{first}, {rec.get('year', 'n.d.')}" if style == "apa" else f"{first} {rec.get('year', 'n.d.')}"


def cite(text: str, library: dict[str, dict[str, Any]], style: str) -> tuple[str, list[str]]:
    """Replace [R:PMID] tokens with the style's citations. Numbered styles number in order of first citation and
    merge ranges ([1-3]); author-date styles give (Author, Year; ...). Returns the text and the cited PMIDs in
    reference-list order."""
    order: list[str] = []
    for m in REF_TOKEN.finditer(text or ""):
        if m.group(1) in library and m.group(1) not in order:
            order.append(m.group(1))
    numbers = {p: i + 1 for i, p in enumerate(order)}

    def group(m: re.Match) -> str:
        ids = [x for x in REF_TOKEN.findall(m.group(0)) if x in library]
        if not ids:
            return m.group(0)
        if style in NUMBERED:
            ns = sorted(set(numbers[x] for x in ids))
            parts, i = [], 0
            while i < len(ns):
                j = i
                while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                    j += 1
                parts.append(f"{ns[i]}-{ns[j]}" if j - i >= 2 else ",".join(map(str, ns[i:j + 1])))
                i = j + 1
            label = ",".join(parts)
            return f"^{label}^" if style == "ama" else f" [{label}]"
        return " (" + "; ".join(_short(library[x], style) for x in dict.fromkeys(ids)) + ")"
    out = REF_GROUP.sub(group, text or "")
    if style not in NUMBERED:
        order = sorted(order, key=lambda p: (_names(library[p], style).lower(), library[p].get("year", "")))
    return out, order


def reference_list(order: list[str], library: dict[str, dict[str, Any]], style: str) -> list[str]:
    if style in NUMBERED:
        return [f"{i}. {format_reference(library[p], style)}" for i, p in enumerate(order, 1)]
    return [format_reference(library[p], style) for p in order]


def quote_in_abstract(quote: str, rec: dict[str, Any]) -> bool:
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()
    q = norm(quote)
    return len(q) >= 12 and q in norm(rec.get("abstract", ""))
