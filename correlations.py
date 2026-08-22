import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import chi2, pointbiserialr
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build
from marine_recruitment_data.models import SPECS
d=build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
sub=d[d.passed==1].reset_index(drop=True)

CONT={"log_agi":"ZIP mean income (log)","t":"time trend (years)","edu_years":"years of education",
      "n_sources":"number of lead sources","log_school_volume":"school applicant volume",
      "log_zip_returns":"ZIP population (log returns)"}
CAT={"race_g":"race","cred_g":"credential","rss_g":"recruiting substation","quarter":"quarter",
     "zip3_g":"ZIP3 geography","tier":"education tier"}
BIN={"is_senior":"is high-school senior","src_area":"source: area canvassing",
     "src_list":"source: list canvass","src_priority":"source: priority prospect card",
     "src_referral":"source: referral"}

def lr_test(var, data, outcome, categorical):
    term=f"C({var})" if categorical else var
    dd=data.dropna(subset=[var,outcome])
    null=smf.logit(f"{outcome} ~ 1",data=dd).fit(disp=0)
    alt =smf.logit(f"{outcome} ~ {term}",data=dd).fit(disp=0,maxiter=200)
    stat=2*(alt.llf-null.llf); df=int(alt.df_model)
    return stat, df, 1-chi2.cdf(stat,df), alt.prsquared, len(dd)

def screen(data,outcome,label):
    rows=[]
    for grp,cats in ((CONT,False),(CAT,True),(BIN,False)):
        for v,name in grp.items():
            try:
                s,df,p,r2,n=lr_test(v,data,outcome,cats)
                rows.append(dict(variable=name,term=v,chi2=s,df=df,p=p,pseudo_r2=r2,n=n))
            except Exception as e: pass
    t=pd.DataFrame(rows).sort_values("p")
    k=len(t); t["bh_crit"]=np.arange(1,k+1)/k*0.05
    t["sig_05"]=t.p<0.05; t["survives_fdr"]=t.p<=t.bh_crit
    print("="*104); print(f"{label}   N={len(data)}"); print("="*104)
    print("univariate association with the outcome, likelihood-ratio test\n")
    print(t[["variable","chi2","df","p","pseudo_r2","sig_05","survives_fdr"]]
          .to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    sig=t[t.sig_05]
    print(f"\n  SIGNIFICANT AT alpha=0.05: {len(sig)} of {k}")
    print(f"  -> ranked: {', '.join(sig.variable.tolist())}")
    return t

t1=screen(d,"passed","DEPENDENT VARIABLE 1 — passed (AFQT >= 31)")
print()
t2=screen(sub,"alpha","DEPENDENT VARIABLE 2 — alpha (AFQT >= 50 | passed)")
t1.to_csv("univariate_passed.csv",index=False); t2.to_csv("univariate_alpha.csv",index=False)
