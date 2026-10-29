# Plan

Single source of truth for the build. Deadline: **Wed 4 Nov 2026, 21:00 ET =
Thu 5 Nov, 10:00 MYT**. Submit on Wed 4 Nov (MYT) evening.

| Day (MYT) | Build | Done when |
|---|---|---|
| Wed 28 Oct | Event log, workflow engine, model adapter, API, Modal deploy, tests | ✅ tests green; smoke test passes on both endpoints |
| Thu 29 | Sandboxed analysis execution; Data Curator; confirmatory plan runs automatically | ✅ built 28 Oct; live Modal demo passed 28 Oct 22:17 MYT (sandbox, network blocked, data hash verified) |
| Thu 29 (cont.) | Honest attempt labels; demo shows model and notes; **Writing Agent**; number tracer | ✅ Live demo 29 Oct (early afternoon MYT): draft written by Muse Spark 1.3. Live run exposed a tracer bug (0 numbers checked, false pass); fixed and re-verified: 22/22 numbers trace. CI test setup fixed |
| Thu 29 (cont.) | Writing Agent loop: check → findings → one rewrite → authors. Completeness rule: every analysis that ran must be reported | ✅ built and tested 29 Oct (offline, scripted model); not yet run live |
| Thu 29 (cont.) | **Labels corrected** to the ICH E9 sense (author's decision, 29 Oct): confirmatory = pre-specified primary analysis covered by the protocol's sample-size justification, both quotations checked by code; otherwise exploratory (pre-specified or post hoc). A shortfall against the planned sample is flagged and must be stated; the label does not change (author chose this over automatic downgrading). Previously every planned analysis was "confirmatory", which mislabelled the secondary AUC in the live runs | ✅ built and tested 29 Oct; browser walk-through passes |
| Thu 29 (cont.) | **Figures** (author asked 29 Oct): forest plot and participant flow drawn by Saksi from recorded results; ROC, Kaplan-Meier or calibration curve required by rule and drawn by the analysis code in the sandbox (matplotlib, SVG, same code+data+seed gives the same SVG); captions number-checked; flow counts that do not add up flagged | ✅ built and tested 29 Oct (offline); browser walk-through passes. Not yet run on Modal (the sandbox image gains matplotlib) |
| Fri 30 | Reviewer Assistant (Muse Spark Open); review loop over the API. *Author-requested exploratory analyses already built 28 Oct* | ✅ built early, 29 Oct: two review rounds run end to end over the API (offline, scripted model). Live run on Muse Spark Open pending |
| Sat 31 | Web interface: authors, reviewer, ledger, provenance | ✅ built early, 29 Oct: whole flow (create to FINAL, two review rounds) driven in a headless browser against the offline server; no console errors; phone width checked. Not yet seen on the Modal deployment |
| Sun 1 Nov | Qualitative (review-stage) path; evaluation harness | Harness runs from one command |
| Mon 2 | Live session with real co-authors and a reviewer; evaluation run | Results table exists |
| Tue 3 | Hardening, replay mode, README, diagram; **code freeze in the evening** | Demo path works three times in a row |
| Wed 4 | Video (morning), submission text, **submit**. Video **opens with one line** on what was pre-existing and what was built during the event | Submitted |
| Thu 5 (to 10:00) | Buffer only | — |

## Live runs (synthetic data, Modal)

| When (MYT) | What ran | Result |
|---|---|---|
| 28 Oct 22:17 | Curate, plan, sandboxed analyses | Network blocked, data hash verified, both analyses ok |
| 29 Oct, early afternoon | + Writing Agent | Tracer bug found (0 numbers checked, false pass); fixed, 22/22 traced |
| 29 Oct ~14:00 | + sign-off, Reviewer Assistant (Muse Spark Open) | Draft passed first time (22 numbers, all traced, no rewrite needed). Briefing: 8 points, 8/8 quotes verified, 24.6 s, 12,226 tokens. Whole run: 6 model calls, 38,496 tokens, ~93 s of agent time. **One factual error**: P5 said the AUC interval (0.51 to 0.79) "includes 0.5"; it does not. A number flag would not have caught it ("0.5" matches 0.506 at one decimal), so whether each interval includes its null value is now computed by code and given to both models and to the reviewer. Also seen: P2's quote is real but unrelated to its question (quote checks prove the text exists, not that it is the right place) |

| 29 Oct ~15:45 | + corrected labels, sample check, figures (ROC curve drawn in the Modal sandbox) | Labels right first time (primary confirmatory; AUC exploratory, pre-specified, with reasons). Shortfall flagged (388/400, 18/40 events) and stated by the Writing Agent at its first attempt: 26 numbers, all traced, no rewrite. Four figures, ROC drawn in the sandbox. Reviewer Assistant: 7 points, 7/7 quotes verified, and it used the interval facts correctly (AUC lower limit "very close to 0.5", not "includes 0.5"). Run: 6 model calls, 52,048 tokens, ~136 s of agent time. **It found a Saksi flaw** (P7): two identical flow diagrams, because de-duplication compared drawings and the analyses worded exclusions differently; now merged by counts. Also fixed: Saksi states the exploratory label on captions written by analysis code; the writer no longer says "associated with" when the interval includes no association |
| 29 Oct ~16:30 | **First use of the web page on the Modal deployment**, by the author, with his own wide datasets | Signed in, created a project, uploaded two datasets. The Data Curator failed on both: its single reply was cut off at the length limit mid-dictionary ("Expecting ',' delimiter"). Fixed: wide datasets are curated in batches of 40 columns plus an overview; a cut-off batch is split; an invalid reply gets one retry; every call is logged. On wide datasets each analysis now sees only the columns it names. The page shows which dataset is in use |
| 29 Oct ~17:10 | Author's review of the deployed page with his own protocol | Four problems: (1) not clear the files were read, with two datasets listed; (2) many analyses, none confirmatory; (3) the reasons were confusing; (4) authors could not choose which analyses go ahead. Fixed in update 9: each upload reports what Saksi read (rows × columns; characters of text, with a preview; a warning for a scanned PDF); the plan shows a plain three-line checklist per analysis with what to do; authors can edit the plan (include or leave out with a reason, set roles, paste protocol sentences), re-checked by code and confirmed by every author; the Analysis Agent can redraft with instructions; quotation matching tolerates PDF damage but never a changed number. Why nothing was confirmatory: see next row |
| 29 Oct ~17:30 | Diagnosis from the author's record and the published GDM-RIDMAN protocol (BMJ Open 2022;12:e052554) | The protocol lists five numbered objectives and names none primary, so under the agreed rule nothing is confirmatory; the agent was right on roles. It missed attaching the sample-size calculations (312; 263 → 292 cases + 584 controls; at least 876 records, complete-case if 789) to the analyses they cover. Of two quotations not found, one failed only because the PDF text reads "W e will report" (found by update 9's matching); the other was invented by analogy with objective 4 (rightly rejected). The sample-size paragraph is broken mid-sentence by a footer and Table 1. The plan reply used 15,024 of 16,000 tokens. Update 10: the planned-versus-achieved check covers every analysis a sample-size calculation covers (862 against 876 would be flagged); the agent is told to attach each calculation, quote unbroken sentences and never quote by analogy, and not to infer a primary from the order of objectives; the plan reply has a larger limit and a clear message if cut off. Decision pending: whether authors may name a primary when the protocol names none |

## Organiser guidance (Discord, 28–29 Oct)
- Repo public and created on or after 26 Oct ✅ (created 28 Oct).
- Commit as we go, not in one big push ✅ (small commits, pushed at least daily).
- README states what was pre-existing and what was built during the event ✅.
- Demo video opens with one line saying the same. Draft:
  *"Everything you'll see in Saksi was built during this hackathon, starting 28 October. The only thing
  that existed before is the workflow idea, from my earlier unpublished white paper. No code."*
- README labels every open-source model and library used ✅.

## Cut first, if time runs short
1. Qualitative path
2. Second review round in the demo
3. RO-Crate export

## Never in scope this week
DOIs, publisher decision screen, editor role, authorship-order algorithm,
image or voice models, anything from earlier prototypes.
