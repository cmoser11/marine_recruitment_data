"""Load a BPI-report export and collapse it to one row per applicant.

Raw exports carry one row per (applicant x lead source), so an applicant can
repeat up to six times.  Within a repeat group only Source, RSS Name and
Education Code At Contract vary; everything else is constant.  Education Code At
Contract is populated on contract rows and blank on prospect rows, so sparse
fields are coalesced rather than taken from the first row.
"""
from __future__ import annotations

import hashlib

import numpy as np
import openpyxl
import pandas as pd

from .codes import alpha_bravo, meets_tier_minimum, mental_group, parse_education_code
from .codes import ALPHA_CUTOFF, PASS_CUTOFF

RAW_COLUMNS = {
    "RSS Name": "rss",
    "ASAVAB Test Date": "test_date",
    "ASVAB - AFQT Score": "afqt",
    "Race": "race",
    "School Name": "school",
    "Source": "source",
    "Current Zipcode": "zipcode",
    "Education Code At Contract": "edu_contract",
    "Education Code": "edu_code",
}

# Fields that identify one applicant's test event.  No applicant ID exists in the
# export, so identity is reconstructed from these.
IDENTITY_KEY = ["test_date", "afqt", "school", "zipcode", "race", "edu_code"]


def _stable_id(values, prefix):
    raw = "|".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                   for v in values)
    return prefix + hashlib.sha1(raw.encode()).hexdigest()[:10].upper()


def read_raw(path, sheet="BPI Report"):
    """Read the export.  Every cell arrives as text, including scores and dates."""
    ws = openpyxl.load_workbook(path, data_only=True)[sheet]
    rows = list(ws.values)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"unrecognised export; missing columns: {sorted(missing)}")
    df = df.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
    df["afqt"] = pd.to_numeric(df["afqt"], errors="coerce")
    df["has_timestamp"] = df["test_date"].astype(str).str.contains(" ")
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce", format="mixed")
    return df


def to_applicants(df):
    """Collapse rows to one per applicant, assigning a stable applicant_id."""
    grouped = df.groupby(IDENTITY_KEY, dropna=False)
    out = grouped.agg(
        n_rows=("source", "size"),
        n_sources=("source", "nunique"),
        sources=("source", lambda s: sorted(set(s))),
        rss=("rss", lambda s: sorted(set(s))[0]),
        n_rss=("rss", "nunique"),
        contracted=("edu_contract", lambda s: bool(s.notna().any())),
        # coalesce: contract rows carry the code, prospect rows do not
        edu_contract=("edu_contract", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        has_timestamp=("has_timestamp", "all"),
    ).reset_index()

    out["applicant_id"] = [_stable_id(r, "MRD-") for r in out[IDENTITY_KEY].itertuples(index=False)]

    decoded = out.apply(lambda r: parse_education_code(r.edu_code, r.afqt), axis=1)
    out["credential"] = [d[0] for d in decoded]
    out["tier"] = [d[1] for d in decoded]
    out["edu_years"] = [d[2] for d in decoded]

    out["mental_group"] = out["afqt"].map(mental_group)
    out["alpha_bravo"] = out["afqt"].map(alpha_bravo)
    out["meets_tier_minimum"] = out.apply(lambda r: meets_tier_minimum(r.afqt, r.tier), axis=1)
    out["passed"] = (out["afqt"] >= PASS_CUTOFF).astype(int)
    out["alpha"] = (out["afqt"] >= ALPHA_CUTOFF).astype(int)
    out["is_senior"] = out["credential"].eq("HSSenior")

    # Retests cannot be linked without a real applicant ID: the same person
    # tested twice has two dates and two scores, so counts as two applicants.
    # This heuristic grouping exists only for sensitivity checks.
    out["retest_group_id"] = [
        _stable_id(r, "RG-")
        for r in out[["school", "zipcode", "race", "credential"]].itertuples(index=False)
    ]
    n_dates = out.groupby("retest_group_id")["test_date"].transform("nunique")
    out["possible_retest"] = n_dates > 1

    cols = ["applicant_id"] + IDENTITY_KEY + [c for c in out.columns
                                              if c not in IDENTITY_KEY + ["applicant_id"]]
    return out[cols].sort_values("test_date").reset_index(drop=True)


def load(path, sheet="BPI Report"):
    return to_applicants(read_raw(path, sheet))
