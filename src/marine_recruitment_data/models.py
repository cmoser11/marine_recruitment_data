"""Model specifications, fitting, and cross-validated comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .features import add_school_aggregates

BASE_REF = ("C(race_g, Treatment('WHITE'))")
CRED_REF = ("C(cred_g, Treatment('TradHSDiploma'))")
RSS_REF = ("C(rss_g, Treatment('EL PASO EAST'))")

# Nested specifications, each adding one block of variables.
SPECS = {
    "M0 demographics":      f"{BASE_REF}",
    "M1 +education":        f"{BASE_REF} + {CRED_REF}",
    "M2 +station/source":   f"{BASE_REF} + {CRED_REF} + {RSS_REF} + src_area + src_list + src_priority + n_sources",
    "M3 +seasonality":      f"{BASE_REF} + {CRED_REF} + {RSS_REF} + src_area + src_list + src_priority + n_sources + t + C(quarter)",
    "M4 +ZIP income":       f"{BASE_REF} + {CRED_REF} + {RSS_REF} + src_area + src_list + src_priority + n_sources + t + C(quarter) + log_agi + agi_missing + log_zip_returns",
    "M5 +ZIP3 fixed eff":   f"{BASE_REF} + {CRED_REF} + {RSS_REF} + src_area + src_list + src_priority + n_sources + t + C(quarter) + log_agi + agi_missing + log_zip_returns + C(zip3_g)",
}
SCHOOL_SPEC = ("M6 +school history",
               f"{BASE_REF} + {CRED_REF} + {RSS_REF} + src_area + src_list + src_priority + n_sources + t + C(quarter) + log_agi + agi_missing + log_zip_returns + school_{{outcome}}_rate + school_n_prior")

RF_FEATURES = ["race_g", "cred_g", "rss_g", "zip3_g", "quarter", "src_area", "src_list",
               "src_priority", "src_referral", "n_sources", "t", "log_agi", "agi_missing",
               "log_zip_returns", "log_school_volume", "edu_years", "tier"]


def fit_logit(formula, data, outcome, cluster="school"):
    """Logit with cluster-robust standard errors (applicants cluster in schools)."""
    groups = data[cluster].astype("category").cat.codes.values
    plain = smf.logit(f"{outcome} ~ {formula}", data=data).fit(disp=0, maxiter=200)
    robust = smf.logit(f"{outcome} ~ {formula}", data=data).fit(
        disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    ci = np.asarray(robust.conf_int())
    table = pd.DataFrame(
        {"coef": robust.params, "se": robust.bse, "z": robust.tvalues, "p": robust.pvalues,
         "odds_ratio": np.exp(robust.params), "ci_lo": np.exp(ci[:, 0]), "ci_hi": np.exp(ci[:, 1])},
        index=plain.params.index)
    return plain, table


def rf_model(seed=0):
    return RandomForestClassifier(
        n_estimators=800, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1)


def _rf_matrix(data):
    X = pd.get_dummies(data[RF_FEATURES], columns=["race_g", "cred_g", "rss_g", "zip3_g", "quarter"],
                       drop_first=False)
    return X.astype(float).fillna(X.astype(float).median())


def cv_compare(data, outcome, n_splits=5, seed=0):
    """Grouped CV AUC for every spec plus the random forest.

    Folds are grouped by school so no school appears in both train and test:
    with 300 schools and 143 singletons, a random split would leak school-level
    information and inflate every score.
    """
    y = data[outcome].values
    groups = data["school"].values
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups=groups))
    rows = []

    for name, formula in SPECS.items():
        aucs = []
        for tr, te in folds:
            try:
                m = smf.logit(f"{outcome} ~ {formula}", data=data.iloc[tr]).fit(disp=0, maxiter=200)
                aucs.append(roc_auc_score(y[te], m.predict(data.iloc[te])))
            except Exception:
                pass
        ins = smf.logit(f"{outcome} ~ {formula}", data=data).fit(disp=0, maxiter=200)
        rows.append({"model": name, "kind": "logit", "cv_auc": np.mean(aucs),
                     "cv_sd": np.std(aucs), "in_sample_auc": roc_auc_score(y, ins.predict()),
                     "pseudo_r2": ins.prsquared, "aic": ins.aic, "k": int(ins.df_model)})

    # school history: aggregates rebuilt inside each fold from training rows only
    name, tpl = SCHOOL_SPEC
    formula = tpl.format(outcome=outcome)
    aucs = []
    for tr, te in folds:
        dd = add_school_aggregates(data, data.index[tr], outcome=outcome)
        try:
            m = smf.logit(f"{outcome} ~ {formula}", data=dd.iloc[tr]).fit(disp=0, maxiter=200)
            aucs.append(roc_auc_score(y[te], m.predict(dd.iloc[te])))
        except Exception:
            pass
    dd_full = add_school_aggregates(data, data.index, outcome=outcome)
    ins = smf.logit(f"{outcome} ~ {formula}", data=dd_full).fit(disp=0, maxiter=200)
    rows.append({"model": name, "kind": "logit", "cv_auc": np.mean(aucs), "cv_sd": np.std(aucs),
                 "in_sample_auc": roc_auc_score(y, ins.predict()), "pseudo_r2": ins.prsquared,
                 "aic": ins.aic, "k": int(ins.df_model)})

    X = _rf_matrix(data)
    aucs = []
    for tr, te in folds:
        rf = rf_model(seed).fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], rf.predict_proba(X.iloc[te])[:, 1]))
    rf_full = rf_model(seed).fit(X, y)
    rows.append({"model": "RF random forest", "kind": "ml", "cv_auc": np.mean(aucs),
                 "cv_sd": np.std(aucs),
                 "in_sample_auc": roc_auc_score(y, rf_full.predict_proba(X)[:, 1]),
                 "pseudo_r2": np.nan, "aic": np.nan, "k": X.shape[1]})

    out = pd.DataFrame(rows)
    out["shrinkage"] = out.in_sample_auc - out.cv_auc
    return out, X, rf_full
