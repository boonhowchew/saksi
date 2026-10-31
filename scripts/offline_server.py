# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew
"""Offline rehearsal server: the real Saksi app with SCRIPTED model replies.

No API keys, no network, no tokens spent. For trying the web page and for
scripts/web_walkthrough.py. Each scripted reply is used once, so restart the
server before each run-through. Real runs use Modal (see README).

  pip install -e ".[dev]"
  python scripts/offline_server.py      # then open http://127.0.0.1:8765
"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(META_API_KEY="offline", BASETEN_API_KEY="offline")
os.environ.pop("SAKSI_CREATE_TOKEN", None)
import numpy as np
import statsmodels.formula.api as smf
import uvicorn
from saksi import demo_data, standard
from saksi.agents import Agents
from saksi.api import create_app
from saksi.llm import ModelClient
from saksi.sandbox import LocalExecutor
from sklearn.metrics import roc_auc_score
from tests.test_agents import AUC_FIXED, CURATOR_REPLY, PLAN_REPLY, ScriptedModel, fence
from tests.test_sandbox import GOOD_WITH_DIAGNOSTICS

df = demo_data.make().dropna(subset=["gdm", "booking_bmi", "age", "parity", "previous_gdm"])
m = smf.logit("gdm ~ booking_bmi + age + parity + previous_gdm", data=df).fit(disp=0)
ci = np.exp(m.conf_int()); orr = np.exp(m.params)
f = lambda t: f"{orr[t]:.2f} (95% CI {ci.loc[t,0]:.2f} to {ci.loc[t,1]:.2f})"
auc = roc_auc_score(df["gdm"], m.predict(df))
# The standard steps and the diagnostics, computed the way the server will compute them, so the draft's numbers trace.
CSV = demo_data.make().to_csv(index=False).encode()
TABLE1 = {"stratify_by": "gdm", "variables": ["age", "booking_bmi", "parity", "previous_gdm", "ethnicity", "clinic"],
          "why": "Characteristics of the women by the outcome."}
run = lambda code: LocalExecutor().run(code, CSV, seed=7).results
dq, t1 = run(standard.code("data_quality")), run(standard.code("table1", TABLE1))
diag = run(GOOD_WITH_DIAGNOSTICS)["diagnostics"]
bmi_missing = next(c["missing"] for c in dq["columns"] if c["column"] == "booking_bmi")
vif_max = max(diag["assumptions"]["collinearity"]["vif"].values())
cal = diag["fit"]["calibration"]
AUC_PLAN = {"id": "auc", "title": "Discrimination of the primary model", "objective": "discrimination",
            "outcome": {"name": "GDM", "column": "gdm"}, "exposures": [], "covariates": [],
            "method": "AUC with 95% CI", "effect_measure": "AUC", "missing_data": "complete case",
            "protocol_quote": "Discrimination of the primary model will be summarised with the area under the receiver",
            "role": "secondary", "primary_quote": "", "sample_size_quote": "", "sample_size_target": {}, "unresolved": [],
            "kind": "other"}
DRAFT = f"""[Authors: introduction]

## Methods
As pre-specified in the protocol, a multivariable logistic regression was fitted with gestational diabetes (GDM) as the outcome and booking BMI as the exposure, adjusted for maternal age, parity and previous GDM. Records with missing values were excluded (complete-case analysis). The data were first checked for missing values, duplicates and outliers. Characteristics were summarised by GDM status as mean (SD) or median (IQR), as their distribution indicated, and as n (%), and compared with t, Mann-Whitney, chi-square or Fisher's exact tests. The model was checked for events per predictor parameter, collinearity, linearity of the logit, influential observations, fit, discrimination and calibration. The dataset held {dq['n_rows']} records and {dq['n_columns']} columns, with no duplicated records; booking BMI was missing for {bmi_missing} women [A:s_data_quality].

## Results
Table 1 describes the {t1['n_analysed']} women by GDM status; those with GDM were slightly older and more often had previous GDM [A:s_table1.r1].

Table 2 and Figure 1 give the adjusted estimates of the primary model, reported as exploratory because the achieved sample fell short of the plan (below). In this exploratory analysis, booking BMI was associated with GDM with an adjusted odds ratio of {f('booking_bmi')} per unit [A:primary]. In the same exploratory model, previous GDM had an adjusted odds ratio of {f('previous_gdm')} [A:primary].

The exploratory model had {diag['assumptions']['events_per_parameter']} outcome events per predictor parameter, fewer than usually recommended, so its estimates may be unstable [A:primary]. Its C-statistic was {diag['fit']['c_statistic']:.2f} and its apparent calibration slope (Figure 2) {cal['slope']:.2f}, as expected on the data the model was fitted to; performance in new women needs validation [A:primary]. The largest variance inflation factor was {vif_max:.2f}, so collinearity was not a concern [A:primary].

Of 400 women planned (about 40 GDM events), 388 were analysed, with 18 GDM events (Figure 3), so this pre-specified primary analysis is reported as exploratory [A:primary].

In an exploratory variant of this model that the authors asked for after seeing the results, booking BMI had an adjusted odds ratio of {f('booking_bmi')} (the same as in Table 2) [A:x_q1].

In a pre-specified exploratory analysis, the area under the ROC curve of the primary model was {auc:.2f} (Table 3; Figures 4 and 5) [A:auc.r1].

## Discussion
The intervals are wide and include no association, so the data are compatible with both no association and a modest one. [Authors: comparison with previous studies]
"""
LIMIT = "The number of GDM events was small relative to the number of model parameters, so the estimates may be unstable."
REVIEW1 = {"primary_claim": "Booking BMI was not clearly associated with GDM after adjustment.",
           "summary": "One confirmatory model. Few events and wide intervals.",
           "points_of_inquiry": [
               {"category": "statistics", "severity": "major", "quote": "Booking BMI was associated with GDM with an adjusted odds ratio",
                "question": "How many GDM events were there, and how many per model parameter?",
                "why_it_matters": "Few events per parameter make logistic regression estimates unstable."},
               {"category": "interpretation", "severity": "minor", "quote": "compatible with both no association and a modest one",
                "question": "Could you state the range of associations the interval supports in plain terms?",
                "why_it_matters": "Readers over-interpret non-significant results."}]}
REVIEW2 = {"primary_claim": "Unchanged.", "summary": "R1 is dealt with in the Discussion.", "points_of_inquiry": [],
           "item_assessments": [{"item_id": "R1", "assessment": "addressed", "quote": LIMIT[:70], "note": "A limitation now states this."}]}
PLAN_OUT = {"analyses": [dict(PLAN_REPLY["analyses"][0], kind="multivariable"), AUC_PLAN], "table1": TABLE1,
            "report": {"aim": "Is booking BMI associated with gestational diabetes, independent of age, parity and previous GDM?",
                       "design": "cohort", "questions": ["Should this paper also report the discrimination of the model?"]}}
model = ScriptedModel({"curate": [CURATOR_REPLY] * 6,      # the Curator updates its report as the work goes on
                       # the first plan, then the redraft agreed in the plan conversation
                       "plan": [PLAN_OUT, PLAN_OUT],
                       # the planned analyses; then, from the walk-through's AI boxes, a variant and a redo
                       "code": [fence(GOOD_WITH_DIAGNOSTICS), fence(AUC_FIXED), fence(GOOD_WITH_DIAGNOSTICS), fence(AUC_FIXED)], "write": [DRAFT], "review": [REVIEW1, REVIEW2]})
root = tempfile.mkdtemp()
app = create_app(root, agents_factory=lambda p: Agents(p, ModelClient(p.blobs, transport=model, mode="live"), LocalExecutor()))
print(f"Offline Saksi on http://127.0.0.1:8765 (data in {root})")
uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
