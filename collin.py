import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, patsy, statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build
from marine_recruitment_data.models import SPECS
d=build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
sub=d[d.passed==1].reset_index(drop=True)

print("="*84); print("1. IS RACE COLLINEAR WITH ZIP INCOME?"); print("="*84)
# how much of ZIP income variation does race explain?
r=smf.ols("log_agi ~ C(race_g)",data=d).fit()
print(f"  R^2 of log_agi ~ race            = {r.rsquared:.4f}   (race explains {r.rsquared:.1%} of ZIP-income variance)")
r3=smf.ols("log_agi ~ C(zip3_g)",data=d).fit()
print(f"  R^2 of log_agi ~ zip3            = {r3.rsquared:.4f}")
print("\n  mean ZIP income by race group:")
print(d.groupby("race_g").agg(n=("mean_agi_k","size"),mean_agi_k=("mean_agi_k","mean"),
                              median=("mean_agi_k","median")).round(1).to_string())
# Cramer's V race x zip3
ct=pd.crosstab(d.race_g,d.zip3_g)
from scipy.stats import chi2_contingency
c2,p,_,_=chi2_contingency(ct); n=ct.values.sum()
v=np.sqrt(c2/(n*(min(ct.shape)-1)))
print(f"\n  race x ZIP3 association: Cramer's V = {v:.3f}  (chi2 p={p:.2e})")
print("   -> V ~0.3 is moderate: race and geography overlap but are far from interchangeable")

print("\n"+"="*84); print("2. VIF ON THE FULL M4 DESIGN (>5 = concern, >10 = serious)"); print("="*84)
X=patsy.dmatrix(SPECS["M4 +ZIP income"],d,return_type="dataframe")
v_=pd.Series([vif(X.values,i) for i in range(X.shape[1])],index=X.columns).drop("Intercept")
top=v_.sort_values(ascending=False).head(6)
print(top.round(2).to_string())
print(f"\n  max VIF = {v_.max():.2f}  -> {'CONCERN' if v_.max()>5 else 'no multicollinearity problem'}")
print(f"  VIF on race:AIAN = {v_[[i for i in v_.index if 'AIAN' in i][0]]:.2f}")
print(f"  VIF on log_agi   = {v_['log_agi']:.2f}")

print("\n"+"="*84); print("3. DOES ZIP INCOME EXPLAIN AWAY RACE?  (coefficient stability)"); print("="*84)
base=SPECS["M3 +seasonality"]
for lbl,data,outcome in [("MODEL 1 pass",d,"passed"),("MODEL 2 alpha|pass",sub,"alpha")]:
    a=smf.logit(f"{outcome} ~ {base}",data=data).fit(disp=0,maxiter=200)
    b=smf.logit(f"{outcome} ~ {SPECS['M4 +ZIP income']}",data=data).fit(disp=0,maxiter=200)
    k=[i for i in a.params.index if "AIAN" in i][0]
    ca,cb=a.params[k],b.params[k]
    print(f"\n  {lbl}")
    print(f"    AIAN coef without ZIP income : {ca:+.4f}  (OR {np.exp(ca):.3f})")
    print(f"    AIAN coef with    ZIP income : {cb:+.4f}  (OR {np.exp(cb):.3f})")
    print(f"    attenuation                  : {(1-cb/ca)*100:+.1f}%  of the race effect absorbed by ZIP income")
    print(f"    SE without / with            : {a.bse[k]:.4f} / {b.bse[k]:.4f}  "
          f"({'inflated' if b.bse[k]>a.bse[k]*1.1 else 'stable'})")
