import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build
d=build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
ll=pd.read_csv("/tmp/zip_latlon.csv",dtype={"zipcode":str})

K=8.0   # empirical-Bayes prior weight: a ZIP needs ~8 applicants to move off the prior
def shrink(g, col, prior):
    s=g[col].sum(); n=len(g)
    return pd.Series({"n":n,"raw":s/n,"shrunk":(s+K*prior)/(n+K)})

rows=[]
prior_pass = d.passed.mean()
sub = d[d.passed==1]
prior_alpha = sub.alpha.mean()

gp = d.groupby("zip5").apply(lambda g: shrink(g,"passed",prior_pass)).reset_index()
ga = sub.groupby("zip5").apply(lambda g: shrink(g,"alpha",prior_alpha)).reset_index()
m = gp.merge(ga,on="zip5",how="left",suffixes=("_pass","_alpha"))
m = m.merge(ll,left_on="zip5",right_on="zipcode",how="left")
inc = d.groupby("zip5").agg(agi=("mean_agi_k","first"),rss=("rss",lambda s:s.mode().iat[0])).reset_index()
m = m.merge(inc,on="zip5",how="left")
m = m.dropna(subset=["lat","lon"])

def odds(p): return p/(1-p)
o_pass, o_alpha = odds(prior_pass), odds(prior_alpha)
m["or_pass"]  = odds(m.shrunk_pass)/o_pass
m["or_alpha"] = odds(m.shrunk_alpha)/o_alpha

recs=[]
for _,r in m.iterrows():
    recs.append(dict(zip=r.zip5, lat=round(r.lat,4), lon=round(r.lon,4),
        n=int(r.n_pass), nPassed=int(0 if np.isnan(r.n_alpha) else r.n_alpha),
        pPass=round(r.shrunk_pass,4), rawPass=round(r.raw_pass,4),
        pAlpha=None if np.isnan(r.shrunk_alpha) else round(r.shrunk_alpha,4),
        rawAlpha=None if np.isnan(r.raw_alpha) else round(r.raw_alpha,4),
        orPass=round(r.or_pass,3),
        orAlpha=None if np.isnan(r.or_alpha) else round(r.or_alpha,3),
        agi=None if np.isnan(r.agi) else round(r.agi,1), rss=r.rss))
payload=dict(zips=recs, priorPass=round(prior_pass,4), priorAlpha=round(prior_alpha,4),
             k=K, totalApplicants=int(len(d)), totalPassed=int(d.passed.sum()),
             totalAlpha=int(sub.alpha.sum()), nZips=len(recs))
json.dump(payload,open("zipmap.json","w"))
print("zips mapped:",len(recs),"of",d.zip5.nunique())
print(f"prior pass {prior_pass:.3f}  prior alpha|pass {prior_alpha:.3f}")
print("\nn distribution per zip:"); print(m.n_pass.describe().round(1).to_string())
print("\ntop 8 zips by volume:")
print(m.nlargest(8,"n_pass")[["zip5","rss","n_pass","raw_pass","shrunk_pass","raw_alpha","shrunk_alpha","agi"]].round(3).to_string(index=False))
print("\nlat/lon extent:",round(m.lat.min(),2),round(m.lat.max(),2),round(m.lon.min(),2),round(m.lon.max(),2))
