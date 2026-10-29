# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Synthetic antenatal-style dataset and protocol for tests and the demo.

Entirely simulated. No real records, no real study."""

import numpy as np
import pandas as pd


def make(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(30, 5, n).round(0)
    bmi = rng.normal(26, 4.5, n).round(1)
    parity = rng.poisson(1.2, n)
    prev_gdm = rng.binomial(1, 0.08, n)
    lp = -5.0 + 0.09 * (bmi - 26) * 1.0 + 0.06 * age + 1.4 * prev_gdm
    gdm = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    ethnicity = rng.choice(["Malay", "Chinese", "Indian", "Other"], n, p=[0.6, 0.2, 0.15, 0.05])
    clinic = rng.choice([f"K{i}" for i in range(1, 7)], n)
    booking = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    df = pd.DataFrame({
        "study_id": [f"S{100000 + i}" for i in range(n)],
        "patient_name": [f"Person {i}" for i in range(n)],
        "age": age, "booking_bmi": bmi, "parity": parity, "previous_gdm": prev_gdm,
        "ethnicity": ethnicity, "clinic": clinic, "booking_date": booking.strftime("%d/%m/%Y"),
        "gdm": gdm,
    })
    df.loc[rng.choice(n, 12, replace=False), "booking_bmi"] = np.nan
    return df


PROTOCOL = """Title: Risk factors for gestational diabetes mellitus in routine antenatal care (synthetic test protocol)

Primary objective: To estimate the association between booking body mass index (BMI) and gestational diabetes
mellitus (GDM), adjusted for maternal age, parity and previous GDM.

Primary analysis: Multivariable logistic regression with GDM as the outcome and booking BMI as the exposure,
adjusted for maternal age, parity and history of previous GDM. Results will be reported as adjusted odds ratios
with 95% confidence intervals. Complete-case analysis will be used.

Secondary analysis: Discrimination of the primary model will be summarised with the area under the receiver
operating characteristic curve (AUC) with a 95% confidence interval.

Sample size: Assuming a GDM prevalence of 10% and a standard deviation of booking BMI of 5 kg/m2, 400 women
(about 40 GDM events) give 80% power to detect an odds ratio of 1.10 per kg/m2 for the primary analysis at a
two-sided 5% significance level.
"""
