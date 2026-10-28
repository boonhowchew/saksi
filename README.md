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
| Hackathon window | 26 Oct – 4 Nov 2026 (ET) |
| First commit | 28 Oct 2026 (MYT); see `git log` |
| New in this repository | **All code.** Event log, workflow engine, model adapter, API, Modal deployment and tests were written during the event. No code from earlier projects was reused |
| Prior work by the author | The workflow design follows the author's earlier conceptual work on AI-verified scholarly publishing (an unpublished white paper). That was design, not software |
| AI assistance in development | Written with Claude (Anthropic) as a coding assistant; the commit trailers say so. **At runtime the product uses only Meta models** |
| Test materials | The author's own studies. **None of their data is in this repository** |

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
| Drafting | — | **Writing Agent** drafts from computed outputs only |
| Author revision | Each author edits; every change is attributed; structured sign-off | Writing Agent helps on request, always naming who asked |
| Review | Human reviewer writes comments item by item | **Reviewer Assistant**: separate open-weights deployment, fresh context, never sees author-side history |
| Response | Authors revise, clarify or rebut each item | As above |
| Attestation | Every author attests the final version | — |

### Rules the engine enforces

- **The log is the record.** Each event holds the hash of the one before it. `GET /projects/{id}/verify` detects any edit, deletion, reordering or truncation.
- **State is rebuilt, never stored.** Replaying the log reconstructs the project exactly.
- **Stage changes are events too**, so the log reads as the project's history.
- **Confirmatory or exploratory is decided by the engine** from the plan every author confirmed. The caller cannot set the label, and it cannot be removed afterwards.
- **Approvals bind to exact content.** A changed plan, a new analysis or an edited draft voids earlier confirmations, acceptances and sign-offs.
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
| Hash-chained event log and verification | Sandboxed analysis execution (Modal) |
| Workflow engine: every stage, both loops, all rules above | Agent prompts: curator, analyst, writer, reviewer assistant |
| Contribution Ledger | Web interface |
| Model adapter for both Meta endpoints, with record/replay | Evaluation on the author's own studies |
| HTTP API with per-person tokens; file upload | RO-Crate export of the final object |

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
