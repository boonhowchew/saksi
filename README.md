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
| New in this repository | **All code.** Event log, workflow engine, agents, number tracer, model adapter, API, web page, Modal deployment and tests were written during the event. No code from earlier projects was reused, and no template or starter repository was used |
| Pre-existing work (before kickoff) | **No code.** Only (1) the workflow design, which follows the author's earlier conceptual work on AI-verified scholarly publishing (an unpublished white paper), and (2) the standard sequence of analysis and the model-diagnostic checklist, which follow the author's own teaching materials (a book on clinical epidemiology and scientific-writing workshop modules). Saksi implements that sequence in new code and its own words; no text from those materials is copied here or given to the models. The demo video opens with this same statement |
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
| NumPy, pandas, SciPy, statsmodels, scikit-learn, patsy (BSD); Matplotlib (Matplotlib licence, BSD-style) | Open-source libraries | Statistical analysis and data-dependent figures inside the sandbox |
| openpyxl (MIT), pypdf (BSD-3), python-docx (MIT) | Open-source libraries | Reading uploaded datasets (Excel) and protocols (PDF, Word) |
| pytest (MIT) | Open-source library | Tests |
| Uvicorn (BSD-3), Playwright (Apache-2.0) with Chromium (BSD) | Open-source tools, development only | Offline rehearsal server and the browser walk-through in `scripts/` |
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
| Analysis | Authors iterate by prompting; **every** author accepts the results | Saksi's own code checks data quality and builds Table 1; then the **Analysis Agent** writes code for the planned analyses, descriptive, univariable, then multivariable with diagnostics. All of it runs in a sandbox |
| Drafting | — | **Writing Agent** drafts from computed outputs only; if the number check fails it gets the findings and one rewrite, then the draft goes to the authors with the findings attached |
| Author revision | Each author edits; every change is attributed; structured sign-off | Writing Agent helps on request, always naming who asked |
| Review | Human reviewer writes every comment and gives every verdict | **Reviewer Assistant** (works for the reviewer only): a different, open-weights model; fresh context for every briefing; never opens the data or sees the authors' side. Raises questions anchored to quotes, each quote checked against the draft. In later rounds it says whether each response deals with the item |
| Response | Authors revise, clarify or rebut each item | As above |
| Attestation | Every author attests the final version | — |

### Rules the engine enforces

- **The log is the record.** Each event holds the hash of the one before it. `GET /projects/{id}/verify` detects any edit, deletion, reordering or truncation.
- **State is rebuilt, never stored.** Replaying the log reconstructs the project exactly.
- **Stage changes are events too**, so the log reads as the project's history.
- **Confirmatory or exploratory is decided by the engine, before any result.** Confirmatory means the pre-specified primary analysis the study was designed and powered for: it must be in the plan every author confirmed, it must be the primary analysis (named so by the protocol, or by the authors as below), and the protocol's sample-size justification must cover it. Code checks both quotations against the stored protocol, whoever proposed the plan; a quotation that is missing or not found makes it exploratory. Everything else is exploratory, either pre-specified or post hoc. The caller cannot set the label, and it cannot be changed afterwards.
- **Authors decide the plan; code checks it.** Authors can leave a protocol analysis out of this paper (with a reason, which the draft must report as a protocol analysis not performed), change an analysis's role, or paste the protocol's own sentences. Every quotation is checked against the protocol again; a near-exact match is accepted only if every number in it is in the protocol, and the protocol's wording is shown. Each revision records who changed what and why, and every author confirms the result.
- **Every study is analysed in the same standard sequence.** (1) Data quality: missing values, duplicates and outliers, counted (never silently removed). (2) Table 1: the sample's characteristics stratified by the outcome or the compared groups, with the summary (mean (SD) or median (IQR)) and the test (t or Welch, Mann-Whitney, ANOVA or Kruskal-Wallis, chi-square or Fisher) chosen by rule from the data. Both steps are Saksi's own fixed code, not model-written; the plan only says which variables Table 1 holds and how it is stratified, and the authors confirm that. (3) Other descriptive analyses, (4) univariable, (5) multivariable analyses, in that order whatever the protocol's order.
- **Every multivariable model is checked before it is trusted.** The analysis code must return Saksi's standard diagnostics (`saksi_checks`, inside the sandbox): events per predictor parameter, collinearity (VIF, tolerance, correlations), linearity of the logit (Box-Tidwell), influential observations (Cook's distance, leverage, DfBeta, residuals), overall fit, discrimination (C-statistic), calibration (slope, intercept, Hosmer-Lemeshow and a calibration plot) and independence; for linear models, residual normality and homoscedasticity. Code missing them is sent back once; still missing, the result is flagged on the page. Calibration on the fitting data is labelled apparent. The draft gives at least one paragraph on the diagnostics right after the last multivariable result, and its numbers are traced like any other.
- **Authors may name the primary analysis; the departure is disclosed.** If the protocol names no primary analysis, or the authors choose a different one, an author can name it in the plan with a reason, before any result. It can then be confirmatory if the protocol describes it and its sample-size calculation covers it. The record keeps who decided and why, the reviewer sees it, and the draft is not passed until a sentence citing that analysis says how it departs from the protocol. Only an author can do this, never an agent. Giving the protocol's own primary another role needs a reason too.
- **Tables are built by code, ready for a journal, and placed in the Results.** Table 1 and a results table for every analysis with estimates (crude and adjusted side by side when both are reported) are built from the recorded results, never by a model: three rules, no vertical lines, estimates as "x (a to b)", P values to three decimals or "<0.001", footnotes for tests, missing values, the sample, the model, the diagnostics, the label and the abbreviations. Tables and figures are numbered in reporting order and placed in the draft's Results after the paragraph that first refers to them. The draft check requires every table and figure to be referred to (and no others); the Writing Agent is told to explain each one in words, giving only the key estimates, not to repeat it.
- **The Analyses page is the generated Results section.** When the last author confirms the plan, Saksi goes straight on: it checks the data, builds Table 1, runs the planned analyses in the standard order, draws the figures and writes the text for each result, and lays them out as a journal Results section (with the data checks set aside for Methods). How each result was produced (code, provenance, diagnostics, earlier runs) is folded away below it.
- **Saksi AI is beside every result, and above them all.** Authors write in plain words ("add waist circumference", "shorter, please", "redo it with DeLong's interval"); the assistant answers and proposes one action, checked by code, which runs through the same recorded routes as any other: a variant (exploratory), a redo (needs one more click), a rewritten text, Table 1, or leaving an exploratory analysis out. It refuses requests that would reshape results because of what they show.
- **Saksi suggests before it is asked.** Fixed rules from epidemiology and biostatistics, applied by code to the results, list what a methodologist would raise, each with a one-click remedy: Firth's penalised regression when events per parameter are few, splines when linearity fails, sensitivity analyses without influential observations or with multiple imputation, bootstrap validation for prediction models, crude estimates or diagnostics that are missing, identical tables, texts not yet written or with untraced numbers, and multiplicity when exploratory analyses accumulate.
- **Saksi prepares the raw data with the authors, openly.** From the Data Curator's findings, the Analysis Agent proposes cleaning steps of fixed, readable types (remove identifiers or duplicate encodings, read missing codes as missing, set impossible values to missing, recode, derive a yes/no variable from a threshold or from several variables, remove duplicated records, exclude records outside the study population), each with its reason. The authors tick the steps or tell Saksi what to change; Saksi's own code, never model code, applies them to a copy, counts what each step did, and keeps the original. The prepared copy is analysed from then on; the steps, their effects and who applied them go to the Methods text, the Writing Agent and the Reviewer Assistant. The Data page compares the original and the prepared data column by column (removed, new, changed, with missing values, distinct values, range and mean before and after), in aggregate so a reader can see it without seeing a record; authors can download both files.
- **Materials can be added and removed at any time, and every change is recorded.** Supporting materials (codebook, ethics approval, registration, other documents) can be uploaded at any stage, and the Writing Agent uses them for Methods details; the dataset and protocol are fixed once the plan is confirmed. Any file can be removed, with a reason; removing the dataset or protocol after the plan is confirmed is recorded as such, and the results stay in the record.
- **What the paper reports, written for the authors before any analysis.** On the Plan page the Writing Agent sets out the aim, the research questions, the design (and why), the population, the outcomes and exposures with their definitions and columns, the approach, and the data in use, with a draft Methods section (design and setting, participants, variables, data preparation, sample size, statistical analysis). It is drawn from the protocol, the analyses the authors chose, their answers and the data preparation, and kept up to date as these change; authors edit the Methods in place under their name, and the draft's Methods start from it. When Saksi has nothing more to ask, one button confirms the plan, and once every author has confirmed, the analysis starts by itself.
- **The Data Curator's report stays current.** When the protocol, the data in use or the authors' answers change, the report is written again, with the answers and the preparation in view, so settled issues are not raised again.
- **Saksi's questions are answered and remembered, and Saksi proposes the answers.** For each question it raises (from the protocol, the plan or the data), Saksi proposes an answer or a solution, grounded in the protocol, the dictionary and the aggregate profile (for example "derive it from B.BMI above 27 kg/m2 in data preparation"); the authors accept it with one click or change it. Each question has an answer box; answers are recorded under the author's name and given to the agents from then on, for the plan, the data preparation, the analysis code and the text.
- **Table 1 is laid out as journals do.** A yes/no characteristic is one row (the number with it, and the percentage); coded values carry their labels from the data dictionary; the groups are named ("Gestational diabetes" and "No gestational diabetes", not "yes/no: 1").
- **The Results read as a paper, and authors edit them in place.** Each result has a journal-style subheading written with its text (never the wording of a request), its table and figures, and nothing else in view. Authors click into any text or subheading and change it; the change is saved under their name, number-checked again, and shown as "Wording edited by" with their names. Who did what to the results, with names and times, is folded below them and in the Record.
- **Every result comes with its text, written to report, not discuss.** After each analysis the Writing Agent writes the passage that reports it: for a table or figure, it points to it and says in words what matters, with the key estimates only; for the data checks, a passage for the statistical-analysis part of Methods. Each passage is number-checked against that analysis (one rewrite if it fails), shown under the analysis, and used as the basis of the draft. Authors can ask for it to be rewritten.
- **Plans made before the standard sequence still follow it.** An analysis without a place in the sequence is given one by a fixed rule from its method (multivariable, univariable, descriptive or other), and Table 1 can be built after the plan was confirmed, by Saksi's own code.
- **Analysis code can lay out its own table** when an author asks for a layout (for example groups as columns); every cell is formatted by the code from values it computed, and Saksi typesets it.
- **An exploratory analysis asked for after the plan can be left out of the paper**, with a reason; it stays in the record and the reviewer is told. A planned analysis cannot be left out this way.
- **The draft is edited on the page; numbers are locked.** Authors click into any paragraph and change its words, and reword a table's title, headings, row labels and footnotes. Numbers in the tables cannot be edited there, and code refuses any table change that would alter a number (a group's code can become words, "0" to "No GDM", but its size stays). Each save is a new version under the author's name, number-checked again.
- **Authors can ask for changes beside each result, and nothing is overwritten.** Under each analysis an author can ask the Analysis Agent for a *variant* (another covariate, model or subgroup): a new analysis, labelled exploratory because it was asked for after the results were seen, while the original stays in the paper. Or the author can ask for a *redo* when something is wrong, with a reason: the new run replaces the old one in the paper and keeps the label fixed before any result, while the old run stays in the record, shown under the new one, and the Reviewer Assistant is told. Table 1's variables can be changed the same way, rebuilt by Saksi's own code.
- **Confirmatory needs the planned sample too (author's definition, 31 Oct).** A pre-specified primary analysis is confirmatory only if the completed study reached the sample its protocol planned; otherwise it is reported as exploratory, with the planned and achieved numbers. This is decided by code from the counts analysed, never from the estimates. Labels appear in the text only, never in headings, captions or table footnotes; for a confirmatory analysis the text shows that the study had the size it planned.
- **What the paper reports is settled first.** The plan states the paper's aim and its study design, which sets the reporting guideline (STROBE, CONSORT, STARD, TRIPOD+AI, PRISMA, SRQR/COREQ). Saksi asks the authors what it needs to know for this paper, and when an answer settles something, offers to redraft the plan that way.
- **Every draft is checked against its reporting guideline.** The Writing Agent is given the guideline's Methods, Results and Discussion items (as short reminders in Saksi's own words); after each draft, each item is listed as reported, partly or missing, with a quotation from the draft that code verifies, and "[Authors]" where only the authors can supply it.
- **Saksi completes what the plan requires.** A planned analysis that fails is tried again, learning from the failure; acceptance counts only the analyses this paper includes; once every author accepts, Saksi writes the draft straight away.
- **A shortfall against the planned sample is flagged, not hidden.** After the analysis runs, code compares the participants and events analysed with the protocol's targets. The label stays as fixed before the results, and the draft must state the planned and achieved numbers.
- **Approvals bind to exact content.** A changed plan, a new analysis or an edited draft voids earlier confirmations, acceptances and sign-offs.
- **Every number is traced.** Each draft and each edit, by agent or author, is checked by code: every number must match a result from an analysis the sentence cites, exploratory results must be called exploratory, and every analysis that ran must be reported (no selective reporting). Sign-offs record the check for the exact version signed.
- **Figures are drawn by code, never by a model.** Saksi draws forest plots and the participant flow diagram from the recorded results. Figures that need the data, such as a ROC or Kaplan-Meier curve, are required by a fixed rule from the plan and drawn by the analysis code in the sandbox, saved by Saksi as SVG with the analysis's provenance. Every caption is number-checked, and a flow diagram whose counts do not add up is flagged. Models see captions, never images.
- **Models are given facts, not asked to compare numbers.** Whether each 95% interval includes its null value (1 for ratios, 0.5 for AUC, 0 for differences) is computed by code and given to the Writing Agent, the Reviewer Assistant and the reviewer.
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
| Hash-chained event log and verification | Evaluation against the author's own study, once its paper is published (now under peer review) |
| Workflow engine: every stage, both loops, all rules above | |
| Contribution Ledger | Qualitative path (transcripts or evidence synthesis: coding and themes, with the same record, approvals and quotation checks): designed, held back so that the quantitative path is finished well (author's decision, 30 Oct). The page says so |
| Model adapter for both Meta endpoints, with record/replay | RO-Crate export of the final object |
| HTTP API with per-person tokens; file upload | |
| Aggregate dataset profile (no rows, small cells pooled, identifiers withheld) | |
| Data Curator: data dictionary and flags; wide datasets in batches of columns, a cut-off reply split and retried | |
| Analysis plan from the protocol, with every quote and column checked against its source | |
| Analysis code run in Modal sandboxes (network blocked), one repair attempt, every attempt logged | |
| Writing Agent: draft from computed results only; fixes its own draft once from the checker's findings, then hands over to the authors | |
| Reviewer Assistant: briefing with points of inquiry; quotes checked; item assessments in later rounds; records exactly what it was given | |
| Review loop over the API: two rounds, from draft to the final version | |
| Number tracer: every number in every draft and edit checked against the results; every analysis must be reported | |
| Interval facts: whether each interval includes its null value, computed by code for the models and the reviewer | |
| Confirmatory only for the pre-specified, powered primary analysis, checked against the protocol; shortfall against the planned sample flagged. Authors may name the primary when the protocol does not, or change it, with a reason: a departure the draft must disclose | |
| Standard analysis sequence for every study: data quality and Table 1 by Saksi's own code, then descriptive, univariable, multivariable; standard diagnostics after every multivariable model | |
| Publication-ready tables built by code and placed in the Results beside the text that explains them; the draft edited on the page, with table numbers locked; per-analysis AI box for a variant or a redo, with the earlier run kept | |
| Figures: forest plot and participant flow by Saksi; ROC, Kaplan-Meier or calibration curves by the analysis code when the plan calls for them; captions checked | |
| Sign in with the personal token alone; project and person ids made automatically. Wrong uploads can be removed (stored copy deleted, name, hash and reason kept). One step from files to plan: the Data Curator, then the Analysis Agent, with unfound quotations sent back once to be corrected and re-checked; authors tick and confirm | |
| Web page: every stage for authors and the reviewer, next step for each person, live record with integrity check, provenance and ledger. No third-party scripts or fonts | |
| Live runs on Modal with both Meta models, from upload to the Reviewer Assistant's briefing (see PLAN.md) | |

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
modal deploy modal_app.py      # the web page is at the deployed URL itself
```

To try the web page without keys or network, run the offline rehearsal server.
It is the real app with scripted model replies:

```bash
python scripts/offline_server.py          # open http://127.0.0.1:8765
python scripts/web_walkthrough.py out/    # optional: the whole flow in a headless browser, with screenshots
```

Set `SAKSI_LLM_MODE=record` to keep model responses, then `replay` to rerun
without network calls.

## Licence

Copyright © 2026 Boon-How Chew. Licensed under the **GNU Affero General Public License v3.0 only**
(`AGPL-3.0-only`); see `LICENSE`. You may use, study, change and share Saksi. If you run a modified
version as a network service, you must offer its users the source of your version under the same
licence. For other licensing terms, contact the author.
