import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build
from marine_recruitment_data.models import cv_compare, fit_logit, SPECS
pd.set_option("display.width",200)

d = build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
print(f"applicants={len(d)}  passed={int(d.passed.sum())} ({d.passed.mean():.1%})")
sub = d[d.passed==1].reset_index(drop=True)
print(f"alpha|passed={int(sub.alpha.sum())} ({sub.alpha.mean():.1%})\n")

for label, data, outcome in [("MODEL 1  P(pass, AFQT>=31)", d, "passed"),
                             ("MODEL 2  P(alpha, AFQT>=50 | passed)", sub, "alpha")]:
    print("="*96); print(label); print("="*96)
    tbl, X, rf = cv_compare(data, outcome)
    print(tbl[["model","kind","cv_auc","cv_sd","in_sample_auc","shrinkage","pseudo_r2","aic","k"]]
          .round(4).to_string(index=False))
    best = tbl.loc[tbl.cv_auc.idxmax()]
    print(f"\n  BEST BY CV AUC: {best.model}  (CV AUC {best.cv_auc:.3f})")
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  RF feature importance (top 12):")
    print(imp.head(12).round(4).to_string())
    tbl.to_csv(f"cv_{outcome}.csv", index=False); imp.to_csv(f"rfimp_{outcome}.csv")
    print()
