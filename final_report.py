import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build, assert_no_leakage
from marine_recruitment_data.models import fit_logit, rf_model, _rf_matrix, SPECS
pd.set_option("display.width",220)

d = build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
sub = d[d.passed==1].reset_index(drop=True)
BEST = SPECS["M4 +ZIP income"]

def hl(y,p,g=10):
    t=pd.DataFrame({"y":np.asarray(y),"p":np.asarray(p)})
    t["b"]=pd.qcut(t.p.rank(method="first"),g,labels=False)
    o=t.groupby("b").agg(obs=("y","sum"),exp=("p","sum"),n=("y","size"))
    s=(((o.obs-o.exp)**2)/(o.exp*(1-o.exp/o.n))).sum(); return s,1-chi2.cdf(s,g-2)

for label,data,outcome in [("MODEL 1 — P(pass, AFQT>=31)",d,"passed"),
                           ("MODEL 2 — P(alpha, AFQT>=50 | passed)",sub,"alpha")]:
    print("="*100); print(f"{label}   [spec M4: +ZIP income]"); print("="*100)
    m,tbl = fit_logit(BEST, data, outcome)
    assert_no_leakage([c for c in data.columns if c in tbl.index])
    print(f"N={int(m.nobs)}  events={int(data[outcome].sum())}  pseudoR2={m.prsquared:.4f}  "
          f"AIC={m.aic:.1f}  LR chi2({int(m.df_model)})={m.llr:.1f}  p={m.llr_pvalue:.2e}")
    t=tbl.copy(); t.index=[i.replace("C(race_g, Treatment('WHITE'))[T.","race:").replace(
        "C(cred_g, Treatment('TradHSDiploma'))[T.","cred:").replace(
        "C(rss_g, Treatment('EL PASO EAST'))[T.","rss:").replace(
        "C(quarter)[T.","qtr:").replace("]","") for i in t.index]
    print("\n-- full table, cluster-robust SE by school, 95% CI on odds ratio --")
    print(t[["coef","se","z","p","odds_ratio","ci_lo","ci_hi"]].round(4).to_string())
    sig=t[(t.p<0.05)&(t.index!="Intercept")].sort_values("p")
    print(f"\n-- SIGNIFICANT AT 95% ({len(sig)}) --")
    print(sig[["odds_ratio","ci_lo","ci_hi","p"]].round(4).to_string() if len(sig) else "none")
    pv=t[t.index!="Intercept"].p.sort_values(); k=len(pv)
    surv=pv.index[pv.values<=np.arange(1,k+1)/k*0.05]
    print(f"\n-- survives Benjamini-Hochberg FDR 5% --\n{list(surv) if len(surv) else 'none'}")
    s,p_=hl(data[outcome],m.predict()); print(f"\nHosmer-Lemeshow: chi2={s:.2f} p={p_:.3f} -> {'poor fit' if p_<0.05 else 'no evidence of misfit'}")

    # permutation importance (unbiased, unlike impurity importance)
    X=_rf_matrix(data); y=data[outcome].values; g=data["school"].values
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=0)
    tr,te=next(iter(cv.split(np.zeros(len(y)),y,groups=g)))
    rf=rf_model().fit(X.iloc[tr],y[tr])
    r=permutation_importance(rf,X.iloc[te],y[te],n_repeats=30,random_state=0,scoring="roc_auc",n_jobs=-1)
    imp=pd.DataFrame({"perm_importance":r.importances_mean,"sd":r.importances_std},index=X.columns)
    imp=imp.sort_values("perm_importance",ascending=False)
    print("\n-- RF PERMUTATION importance, top 10 (held-out AUC drop when shuffled) --")
    print(imp.head(10).round(4).to_string())
    tbl.to_csv(f"coefficients_{outcome}.csv"); imp.to_csv(f"perm_importance_{outcome}.csv")
    print()
