# Saksi

**Research that is witnessed, not reconstructed.**

*Saksi* means *witness* in Malay and Indonesian.

Most tools that check research arrive after the fact. They read a finished
manuscript and try to work out what was done. Saksi takes part while the work
happens: it runs the analysis the protocol specifies, helps authors and
reviewers through drafting and review, and records every step as it occurs.
Each action by a human or an AI agent becomes an entry in a tamper-evident
log. The provenance is captured as the work is done, not inferred later from
a PDF.

> Entry for the **Meta Global AI Developer Hackathon** (Indie Developer track,
> theme *Build for Your World*).

---

## Built during the hackathon

| | |
|---|---|
| Hackathon window | Kickoff 26 Oct 2026, 12:00 PM ET – submission 4 Nov 2026, 9:00 PM ET |
| Repository created | 28 Oct 2026; first commit the same day (see `git log`) |
| New in this repository | **All code.** Event log, workflow engine, model adapter, API, Modal deployment and tests were written during the event. No code from earlier projects was reused, and no template or starter repository was used |
| Pre-existing work (before kickoff) | **No code.** Only the workflow design, which follows the author's earlier conceptual work on AI-verified scholarly publishing (an unpublished white paper). The demo video opens with this same statement |
| AI assistance in development | Written with Claude (Anthropic) as a coding assistant; the commit trailers say so. **At runtime the product uses only Meta models** |
| Test materials | The author's own studies. **None of their data is in this repository** |

### What Saksi is built on

Saksi uses the components below as they are, unmodified. Everything listed
under "New in this repository" was built on top of them.

| Component | Type | What Saksi uses it for |
|---|---|---|
| **Muse Spark 1.3** (`muse-spark-1.3`) via **Meta Model API** | Meta model, hosted API | Data Curator, Analysis Agent and Writing Agent |
| **Muse Spark Open** (`meta-models/muse-spark-open-1026`) via **Baseten** | Meta **open-weights** model, partner inference | Reviewer Assistant, deliberately a separate deployment from the agents that draft |
| **Modal** | Serverless platform (partner) | Hosts the API, the private data volume and the analysis sandboxes |
| FastAPI (MIT), Starlette (BSD-3), Pydantic (MIT), python-multipart (Apache-2.0) | Open-source libraries | HTTP API, request validation and file upload |
| httpx | Open-source library (BSD) | Calls to the model endpoints |
| NumPy, pandas, SciPy, statsmodels, scikit-learn | Open-source libraries (BSD) | Statistical analysis inside the sandbox |
| pytest | Open-source library (MIT) | Tests |
| GNU AGPL-3.0 text | Licence, from the SPDX licence list | `LICENSE` |

---

## How it works

```
 INGEST ──► ANALYSIS ──► DRAFTING ──► AUTHOR_REVISION ──► REVIEW ◄──► RESPONSE
   │            │                          │                │
 dataset +   confirmed plan runs       every author       reviewer accepts,
 protocol;   first; anything else      signs off          every item settled
 every       is labelled EXPLORATORY   (structured)              │
 author      by the engine                                       ▼
 confirms    every author accepts                  ATTESTATION ──► FINAL
 the plan                                          every author attests
```

| Stage | People | Agents (Meta models) |
|---|---|---|
| Ingest | Authors upload the dataset and the protocol; **every** author confirms the analysis plan | **Data Curator** drafts the data dictionary and flags problems |
| Analysis | Authors iterate by prompting; **every** author accepts the results | **Analysis Agent** writes analysis code; it runs in a sandbox |
| Drafting | — | **Writing Agent** drafts from computed outputs only; if the number check fails it gets the findings and one rewrite, then the draft goes to the authors with the findings attached |
| Author revision | Each author edits; every change is attributed; structured sign-off | Writing Agent helps on request, always naming who asked |
| Review | Human reviewer writes every comment and gives every verdict | **Reviewer Assistant** (works for the reviewer only): a different, open-weights model; fresh context for every briefing; never opens the data or sees the authors' side. Raises questions anchored to quotes, each quote checked against the draft. In later rounds it says whether each response deals with the item |
| Response | Authors revise, clarify or rebut each item | As above |
| Attestation | Every author attests the final version | — |

### Rules the engine enforces

- **The log is the record.** Each event holds the hash of the one before it. `GET /projects/{id}/verify` detects any edit, deletion, reordering or truncation.
- **State is rebuilt, never stored.** Replaying the log reconstructs the project exactly.
- **Stage changes are events too**, so the log reads as the project's history.
- **Confirmatory or exploratory is decided by the engine** from the plan every author confirmed. The caller cannot set the label, and it cannot be removed afterwards.
- **Approvals bind to exact content.** A changed plan, a new analysis or an edited draft voids earlier confirmations, acceptances and sign-offs.
- **Every number is traced.** Each draft and each edit, by agent or author, is checked by code: every number must match a result from an analysis the sentence cites, exploratory results must be called exploratory, and every analysis that ran must be reported (no selective reporting). Sign-offs record the check for the exact version signed.
- **No bare "Approve" button.** Sign-off requires restating the primary claim in your own words and answering verification questions, and records the time spent.
- **Agents never act anonymously for a person.** An agent edit must name the author who requested it, and the Contribution Ledger credits both.
- **Review always ends.** Each item is *satisfied* or *agree to disagree*; the second kind goes into the published commentary. Rounds are capped.
- **The final version needs everyone**: the reviewer accepts and every author attests.
- **Refused commands write nothing.**

### Provenance values are measured, never typed

Hashes of inputs, code and outputs, package versions, container image ID and
seeds are all captured at runtime. Anything that cannot be measured is
recorded as empty.

### Models

| Role | Model | Served by |
|---|---|---|
| Curator, Analyst, Writer | `muse-spark-1.3` | Meta Model API |
| Reviewer Assistant | `meta-models/muse-spark-open-1026` (open weights) | Baseten |

Every model call is logged: exact request and response (content-addressed),
model, parameters, token counts and latency. The `-contributor` model
variants, which are discounted in exchange for training on prompts, are
**refused**, because Saksi handles unpublished work.

### Data handling

- Study materials go to a private per-project folder on a private Modal
  Volume. They are never in the repository.
- **Patient-level data is never sent to a language model.** Analyses run as
  code in a sandbox; the models see the protocol, column metadata and computed
  results.
- API keys live in a Modal Secret, never in the browser or the repository.

---

## Status

| Built and tested | Next |
|---|---|
| Hash-chained event log and verification | Web interface |
| Workflow engine: every stage, both loops, all rules above | Live run of the review stage on Muse Spark Open |
| Contribution Ledger | Evaluation on the author's own studies |
| Model adapter for both Meta endpoints, with record/replay | RO-Crate export of the final object |
| HTTP API with per-person tokens; file upload | |
| Aggregate dataset profile (no rows, small cells pooled, identifiers withheld) | |
| Data Curator: data dictionary and flags | |
| Analysis plan from the protocol, with every quote and column checked against its source | |
| Analysis code run in Modal sandboxes (network blocked), one repair attempt, every attempt logged | |
| Writing Agent: draft from computed results only; fixes its own draft once from the checker's findings, then hands over to the authors | |
| Reviewer Assistant: briefing with points of inquiry; quotes checked; item assessments in later rounds; records exactly what it was given | |
| Review loop over the API: two rounds, from draft to the final version | |
| Number tracer: every number in every draft and edit checked against the results; every analysis must be reported | |

**Roadmap beyond the hackathon** (designed, not built): ORCID sign-in,
cryptographic attestation, editor role, open reviewer nomination, DOIs, and
on-premise deployment where the code goes to the data.

## Run it

```bash
pip install -e ".[dev]"
pytest -q

# Deploy
modal secret create saksi-keys META_API_KEY=... BASETEN_API_KEY=... SAKSI_CREATE_TOKEN=...
modal run modal_app.py::smoke
modal deploy modal_app.py
```

Set `SAKSI_LLM_MODE=record` to keep model responses, then `replay` to rerun
without network calls.

## Licence

Copyright © 2026 Boon-How Chew. Licensed under the **GNU Affero General Public License v3.0 only**
(`AGPL-3.0-only`); see `LICENSE`. You may use, study, change and share Saksi. If you run a modified
version as a network service, you must offer its users the source of your version under the same
licence. For other licensing terms, contact the author.
