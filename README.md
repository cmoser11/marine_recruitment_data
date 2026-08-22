# marine_recruitment_data

ASVAB outcome analysis for Marine Corps recruiting exports (RS Albuquerque, FY25–FY26).

Two questions, two models:

1. **P(pass)** — AFQT ≥ 31
2. **P(alpha)** — AFQT ≥ 50 vs. bravo (31–49), among applicants who passed

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The ambient anaconda install will not work: statsmodels 0.12 breaks against numpy ≥ 1.24 and
pandas ≥ 2.0, and `OrderedModel` needs statsmodels ≥ 0.14. Use the venv.

## Commands

```bash
mrd inspect  EXPORT.xlsx                      # counts, code coverage, date range
mrd ingest   EXPORT.xlsx -o applicants.xlsx   # one row per applicant, decoded
mrd model    EXPORT.xlsx --outcome pass       # coefficients, 95% CI, cluster-robust SE
mrd compare  EXPORT.xlsx --outcome alpha      # nested specs + random forest, CV AUC
```

## How the data is read

Raw exports carry **one row per (applicant × lead source)** — an applicant can repeat up to six
times. `ingest.to_applicants` collapses on (test date, AFQT, school, ZIP, race, education code) and
assigns a stable `applicant_id`. Sparse fields are **coalesced**, not taken from the first row:
`Education Code At Contract` is populated on contract rows and blank on prospect rows, so
first-row-wins silently drops it for 126 people.

Retests cannot be linked without a real applicant ID — the same person tested twice has two dates
and two scores and counts as two applicants. `possible_retest` flags candidates for sensitivity
checks only.

## EPM reference

Codes follow **MCRCO 1100.1** (Enlistment Processing Manual). Education codes are compositional, not
a lookup: first two digits are academic years, the third character is the credential category, which
sets the tier (`codes.py`). Mental groups per para 3-6(7): I 93–99, II 65–92, IIIA 50–64, IIIB
31–49, IV 21–30 — so **alpha is AFQT ≥ 50**. Table 3-6 sets tier-aware AFQT minimums (Tier I 21,
Tier II 31, Tier III 50), which is why `meets_tier_minimum` is reported alongside the flat-31
`passed` flag.

## Reference data

`data/zip_income_irs2022.csv` — IRS SOI 2022 ZIP-level mean AGI per return. Public, no key. The
Census ACS API now requires a key, which is why this is used instead.

## What must never enter a model

`Education Code At Contract`, `contracted`, and the POOLEE lead source are all downstream of the
score (the contract code is null for 98.8% of sub-31 applicants). `features.assert_no_leakage`
enforces this. School and ZIP outcome rates must be computed leave-one-out inside a CV fold —
`add_school_aggregates` takes an explicit fit index for that reason.

Cross-validation is **grouped by school**: with 300 schools and 143 singletons, a random split
leaks school-level information and inflates every score.
