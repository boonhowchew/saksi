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
from saksi import demo_data
from saksi.agents import Agents
from saksi.api import create_app
from saksi.llm import ModelClient
from saksi.sandbox import LocalExecutor
from sklearn.metrics import roc_auc_score
from tests.test_agents import AUC_FIXED, CURATOR_REPLY, GOOD, PLAN_REPLY, ScriptedModel, fence

df = demo_data.make().dropna(subset=["gdm", "booking_bmi", "age", "parity", "previous_gdm"])
m = smf.logit("gdm ~ booking_bmi + age + parity + previous_gdm", data=df).fit(disp=0)
ci = np.exp(m.conf_int()); orr = np.exp(m.params)
f = lambda t: f"{orr[t]:.2f} (95% CI {ci.loc[t,0]:.2f} to {ci.loc[t,1]:.2f})"
auc = roc_auc_score(df["gdm"], m.predict(df))
AUC_PLAN = {"id": "auc", "title": "Discrimination of the primary model", "objective": "discrimination",
            "outcome": {"name": "GDM", "column": "gdm"}, "exposures": [], "covariates": [],
            "method": "AUC with 95% CI", "effect_measure": "AUC", "missing_data": "complete case",
            "protocol_quote": "Discrimination of the primary model will be summarised with the area under the receiver",
            "role": "secondary", "primary_quote": "", "sample_size_quote": "", "sample_size_target": {}, "unresolved": []}
DRAFT = f"""[Authors: introduction]

## Methods
As pre-specified in the protocol, a multivariable logistic regression was fitted with gestational diabetes (GDM) as the outcome and booking BMI as the exposure, adjusted for maternal age, parity and previous GDM. Records with missing values were excluded (complete-case analysis).

## Results
Of 400 women planned (about 40 GDM events), 388 were analysed, with 18 GDM events [A:primary]. Booking BMI was associated with GDM with an adjusted odds ratio of {f('booking_bmi')} per unit [A:primary]. Previous GDM had an adjusted odds ratio of {f('previous_gdm')} [A:primary]. In a pre-specified exploratory analysis, the area under the ROC curve of the primary model was {auc:.2f} [A:auc].

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
model = ScriptedModel({"curate": [CURATOR_REPLY], "plan": [{"analyses": [PLAN_REPLY["analyses"][0], AUC_PLAN]}],
                       "code": [fence(GOOD), fence(AUC_FIXED)], "write": [DRAFT], "review": [REVIEW1, REVIEW2]})
root = tempfile.mkdtemp()
app = create_app(root, agents_factory=lambda p: Agents(p, ModelClient(p.blobs, transport=model, mode="live"), LocalExecutor()))
print(f"Offline Saksi on http://127.0.0.1:8765 (data in {root})")
uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
