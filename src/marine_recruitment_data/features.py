"""Feature engineering for the ASVAB outcome models.

Two rules govern what may enter the right-hand side:

* Nothing downstream of the outcome.  ``Education Code At Contract`` is null for
  98.8% of applicants scoring under 31, and the POOLEE lead source means the
  applicant already enlisted in the DEP.  Both are consequences of the score,
  not causes, and are asserted out of the design matrix.
* Anything derived from the outcome of *other* rows (school pass rates, ZIP
  target encodings) must be computed leave-one-out inside a CV fold, never on
  the full sample.  ``add_school_aggregates`` takes an explicit fit index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEAKY = ["edu_contract", "contracted", "src_poolee", "mental_group", "alpha_bravo",
         "meets_tier_minimum", "afqt", "passed", "alpha"]

SOURCES = ["AREA CANVASSING", "LIST CANVASS", "PRIORITY PROSPECT CARD", "POOLEE", "REFERRAL"]

_ZIP_TABLE = Path(__file__).parent / "data" / "zip_income_irs2022.csv"


def load_zip_income():
    """IRS SOI 2022 ZIP aggregates: mean AGI per return ($thousands), returns filed,
    and pop_est, the number of individuals covered by those returns.  pop_est is a
    population proxy, not a census count -- it misses non-filers."""
    z = pd.read_csv(_ZIP_TABLE, dtype={"zipcode": str})
    return z[["zipcode", "mean_agi_k", "n_returns", "pop_est"]]


def build(df, min_credential=20, min_rss=20, min_zip3=20):
    """Applicant table -> modelling frame.  Pure function of each row."""
    d = df.copy()

    for s in SOURCES:
        d["src_" + s.split()[0].lower()] = d["sources"].apply(lambda L: int(s in L))

    # Asian (7) and Pacific Islander (7) applicants all passed, which produces
    # perfect separation and an unbounded likelihood.  Collapsed to keep the
    # observations while leaving the model estimable.
    d["race_g"] = np.where(d["race"] == "WHITE", "WHITE",
                  np.where(d["race"] == "AMERICAN INDIAN OR ALASKA NATIVE", "AIAN",
                           "OTHER_NONWHITE"))

    counts = d["credential"].value_counts()
    d["cred_g"] = d["credential"].where(d["credential"].isin(counts[counts >= min_credential].index),
                                        "OTHER")
    rc = d["rss"].value_counts()
    d["rss_g"] = d["rss"].where(d["rss"].isin(rc[rc >= min_rss].index), "OTHER_SMALL")

    # geography
    d["zip5"] = d["zipcode"].astype(str).str[:5]
    d["zip3"] = d["zip5"].str[:3]
    # 41 of 54 ZIP3 levels hold fewer than 20 applicants and 32 are perfectly
    # separated on the outcome; kept as a single OTHER level so the FE spec stays
    # estimable, the same treatment given to rare RSS and credential levels.
    zc = d["zip3"].value_counts()
    d["zip3_g"] = d["zip3"].where(d["zip3"].isin(zc[zc >= min_zip3].index), "OTHER_ZIP3")
    d = d.merge(load_zip_income(), left_on="zip5", right_on="zipcode",
                how="left", suffixes=("", "_irs"))
    d["log_agi"] = np.log(d["mean_agi_k"])
    d["log_agi"] = d["log_agi"].fillna(d["log_agi"].median())
    d["agi_missing"] = d["mean_agi_k"].isna().astype(int)
    d["log_zip_returns"] = np.log(d["n_returns"].fillna(d["n_returns"].median()))

    # time: linear trend plus quarter effects, since senior share is seasonal
    # and rises sharply across the window (16% -> 30%)
    d["t"] = (d["test_date"] - d["test_date"].min()).dt.days / 365.25
    d["quarter"] = d["test_date"].dt.quarter.astype(str)
    d["fy"] = d["test_date"].dt.year + (d["test_date"].dt.month >= 10).astype(int)

    d["school_volume"] = d.groupby("school")["school"].transform("size")
    d["log_school_volume"] = np.log(d["school_volume"])
    return d


def add_school_aggregates(d, fit_idx, outcome="passed", prior_weight=10.0):
    """Leave-one-out school pass rate, shrunk toward the global mean.

    Rates are computed only from ``fit_idx`` rows so no test-fold outcome leaks
    into its own predictor.  The applicant's own row is always excluded.
    """
    d = d.copy()
    fit = d.loc[fit_idx]
    prior = fit[outcome].mean()
    stats = fit.groupby("school")[outcome].agg(["sum", "count"])
    s = d["school"].map(stats["sum"]).fillna(0.0)
    n = d["school"].map(stats["count"]).fillna(0.0)
    own = d.index.isin(fit_idx)                       # exclude self where in fit set
    s = s - np.where(own, d[outcome], 0.0)
    n = n - own.astype(float)
    d[f"school_{outcome}_rate"] = (s + prior_weight * prior) / (n + prior_weight)
    d[f"school_n_prior"] = n
    return d


def assert_no_leakage(design_columns):
    bad = sorted(set(design_columns) & set(LEAKY))
    if bad:
        raise ValueError(f"outcome-derived columns in design matrix: {bad}")
