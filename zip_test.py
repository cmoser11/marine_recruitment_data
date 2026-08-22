import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import chi2
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build
from marine_recruitment_data.models import SPECS
d=build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
sub=d[d.passed==1].reset_index(drop=True)
def lr(restricted,full,data,outcome,label):
    r=smf.logit(f"{outcome} ~ {restricted}",data=data).fit(disp=0,maxiter=200)
    f=smf.logit(f"{outcome} ~ {full}",data=data).fit(disp=0,maxiter=200)
    stat=2*(f.llf-r.llf); df=int(f.df_model-r.df_model); p=1-chi2.cdf(stat,df)
    print(f"  {label:32s} LR chi2({df})={stat:6.2f}  p={p:.4f}   {'** significant' if p<0.05 else 'not significant'}")
    return p
print("="*88); print("DOES ZIP MATTER?  Likelihood-ratio tests, nested specs"); print("="*88)
for lbl,data,outcome in [("MODEL 1  P(pass)",d,"passed"),("MODEL 2  P(alpha|passed)",sub,"alpha")]:
    print(f"\n{lbl}   N={len(data)}")
    lr(SPECS["M3 +seasonality"],SPECS["M4 +ZIP income"],data,outcome,"ZIP income block (3 df)")
    lr(SPECS["M4 +ZIP income"],SPECS["M5 +ZIP3 fixed eff"],data,outcome,"ZIP3 geography on top of income")
    lr(SPECS["M3 +seasonality"],SPECS["M5 +ZIP3 fixed eff"],data,outcome,"any ZIP info at all")
    # isolate log_agi alone
    base=SPECS["M3 +seasonality"]
    lr(base,base+" + log_agi + agi_missing",data,outcome,"mean AGI alone (2 df)")
print("\n"+"="*88); print("ZIP INCOME GRADIENT (raw, applicant level)"); print("="*88)
d["agi_q"]=pd.qcut(d.mean_agi_k,4,labels=["Q1 lowest","Q2","Q3","Q4 highest"])
t=d.groupby("agi_q",observed=True).agg(n=("passed","size"),mean_agi=("mean_agi_k","mean"),
    pass_rate=("passed","mean"),alpha_rate_all=("alpha","mean"))
t["alpha_rate_given_pass"]=d[d.passed==1].groupby("agi_q",observed=True).alpha.mean()
print(t.round(3).to_string())
