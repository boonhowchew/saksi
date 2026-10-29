# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

import hashlib

import pytest

from saksi.sandbox import CodeRejected, LocalExecutor, check_code
from saksi import demo_data as synthetic

GOOD = '''
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def analyse(data_path, seed):
    df = pd.read_csv(data_path)
    d = df.dropna(subset=["gdm", "booking_bmi", "age", "parity", "previous_gdm"])
    m = smf.logit("gdm ~ booking_bmi + age + parity + previous_gdm", data=d).fit(disp=0)
    ci = m.conf_int()
    est = []
    for term in ["booking_bmi", "age", "parity", "previous_gdm"]:
        est.append({"name": term, "measure": "adjusted odds ratio", "estimate": float(np.exp(m.params[term])),
                    "ci_lower": float(np.exp(ci.loc[term, 0])), "ci_upper": float(np.exp(ci.loc[term, 1])),
                    "p_value": float(m.pvalues[term])})
    return {"n": int(len(d)), "n_analysed": int(len(d)), "n_events": int(d["gdm"].sum()), "estimates": est}
'''


def data_bytes():
    return synthetic.make().to_csv(index=False).encode()


def test_good_code_passes_check_and_runs():
    check_code(GOOD)
    data = data_bytes()
    r = LocalExecutor().run(GOOD, data, seed=7)
    assert r.status == "ok", r.error
    assert r.results["n"] == 388                      # 12 missing BMI dropped
    names = [e["name"] for e in r.results["estimates"]]
    assert names == ["booking_bmi", "age", "parity", "previous_gdm"]
    assert r.meta["data_sha256"] == hashlib.sha256(data).hexdigest()
    assert r.meta["code_sha256"] == hashlib.sha256(GOOD.encode()).hexdigest()
    assert r.meta["seed"] == 7 and r.meta["packages"]["statsmodels"]


def test_same_inputs_same_outputs():
    data = data_bytes()
    a = LocalExecutor().run(GOOD, data, seed=7).results
    b = LocalExecutor().run(GOOD, data, seed=7).results
    assert a == b


@pytest.mark.parametrize("bad, why", [
    ("import os\ndef analyse(data_path, seed):\n    return {}", "import of 'os'"),
    ("import subprocess\ndef analyse(data_path, seed):\n    return {}", "import of 'subprocess'"),
    ("def analyse(data_path, seed):\n    return open(data_path).read()", "open()"),
    ("def analyse(data_path, seed):\n    return eval('1')", "eval()"),
    ("import pandas as pd\ndef analyse(data_path, seed):\n    pd.read_csv(data_path).to_csv('x')\n    return {}",
     ".to_csv()"),
    ("def analyse(data_path, seed):\n    return ().__class__.__bases__", "__class__"),
    ("from urllib import request\ndef analyse(data_path, seed):\n    return {}", "urllib"),
    ("def run(x):\n    return {}", "must define analyse"),
    ("def analyse(data_path, seed)\n    return {}", "syntax error"),
])
def test_bad_code_rejected(bad, why):
    with pytest.raises(CodeRejected, match=why.replace("(", r"\(").replace(")", r"\)").replace(".", r"\.")):
        check_code(bad)


def test_runtime_error_is_captured_not_raised():
    code = "import pandas as pd\ndef analyse(data_path, seed):\n    return {'x': pd.read_csv(data_path)['nope'].mean()}"
    r = LocalExecutor().run(code, data_bytes(), seed=1)
    assert r.status == "error" and "KeyError" in r.error


def test_non_json_results_are_an_error():
    code = "import pandas as pd\ndef analyse(data_path, seed):\n    return {'df': object()}"
    r = LocalExecutor().run(code, data_bytes(), seed=1)
    assert r.status == "error" and "JSON" in r.error


def test_timeout():
    code = "def analyse(data_path, seed):\n    while True:\n        pass"
    r = LocalExecutor().run(code, data_bytes(), seed=1, timeout=2)
    assert r.status == "timeout"
