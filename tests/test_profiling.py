# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

import json

from saksi.profiling import SMALL_CELL, leaks_rows, profile, read_table
from saksi import demo_data as synthetic


def cols(p):
    return {c["name"]: c for c in p["columns"]}


def test_profile_has_no_rows_or_identifiers():
    df = synthetic.make()
    p = profile(df)
    c = cols(p)
    assert p["rows"] == 400
    assert c["study_id"]["kind"] == "identifier_or_free_text" and "levels" not in c["study_id"]
    assert c["patient_name"]["kind"] == "identifier_or_free_text" and "warning" in c["patient_name"]
    assert c["booking_date"]["kind"] == "date" and len(c["booking_date"]["year_range"]) == 2
    assert c["booking_bmi"]["kind"] == "numeric" and c["booking_bmi"]["missing"] == 12
    assert c["gdm"]["kind"] == "binary" and c["previous_gdm"]["kind"] == "binary"
    assert leaks_rows(p, df) == []
    text = json.dumps(p)
    assert "Person 1" not in text and "S100001" not in text


def test_small_cells_are_pooled():
    import pandas as pd
    df = pd.DataFrame({"site": ["A"] * 10 + ["B"] * 6 + ["rare1"] * 2 + ["rare2"], "y": range(19)})
    lv = cols(profile(df))["site"]["levels"]
    assert set(lv["counts"]) == {"A", "B"}
    assert lv["pooled_small_levels"] == {"levels": 2, "records": 3}
    assert all(v >= SMALL_CELL for v in lv["counts"].values())


def test_read_csv_roundtrip():
    df = synthetic.make(50)
    back = read_table(df.to_csv(index=False).encode(), "x.csv")
    assert list(back.columns) == list(df.columns) and len(back) == 50
