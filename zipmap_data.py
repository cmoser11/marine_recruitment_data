import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from marine_recruitment_data.ingest import load
from marine_recruitment_data.features import build
d  = build(load("/Users/caseymoser/Downloads/RS ABQ FY25 AND FY26 ASVAB SCORES.xlsx"))
ll = pd.read_csv("/tmp/zip_latlon.csv", dtype={"zipcode": str})

K = 8.0    # beta-binomial prior weight for rates
A = 2.0    # gamma-Poisson prior events for per-capita yields

sub = d[d.passed == 1]
prior_pass, prior_alpha = d.passed.mean(), sub.alpha.mean()

g = d.groupby("zip5")
tbl = pd.DataFrame({
    "n":        g.size(),
    "nPassed":  g.passed.sum(),
    "nAlpha":   g.alpha.sum(),
    "agi":      g.mean_agi_k.first(),
    "pop":      g.pop_est.first(),
    "rss":      g.rss.agg(lambda s: s.mode().iat[0]),
}).reset_index()

# --- rates, shrunk toward the command baseline -----------------------------
tbl["rawPass"]   = tbl.nPassed / tbl.n
tbl["pPass"]     = (tbl.nPassed + K*prior_pass) / (tbl.n + K)
tbl["rawAlpha"]  = np.where(tbl.nPassed > 0, tbl.nAlpha / tbl.nPassed.replace(0, np.nan), np.nan)
tbl["pAlpha"]    = np.where(tbl.nPassed > 0,
                            (tbl.nAlpha + K*prior_alpha) / (tbl.nPassed + K), np.nan)
odds = lambda p: p / (1 - p)
tbl["orPass"]  = odds(tbl.pPass)  / odds(prior_pass)
tbl["orAlpha"] = odds(tbl.pAlpha) / odds(prior_alpha)

# --- per-capita yields: qualified applicants per 1,000 residents -----------
tbl["popk"] = tbl["pop"] / 1000.0
have = tbl.popk.notna() & (tbl.popk > 0)
basePassPC  = tbl.loc[have, "nPassed"].sum() / tbl.loc[have, "popk"].sum()
baseAlphaPC = tbl.loc[have, "nAlpha"].sum()  / tbl.loc[have, "popk"].sum()

def yield_pc(count, base):
    raw = np.where(have, count / tbl.popk, np.nan)
    shrunk = np.where(have, (count + A) / (tbl.popk + A / base), np.nan)
    return raw, shrunk, np.where(have, shrunk / base, np.nan)

tbl["rawPassPC"], tbl["pcPass"],  tbl["orPassPC"]  = yield_pc(tbl.nPassed, basePassPC)
tbl["rawAlphaPC"], tbl["pcAlpha"], tbl["orAlphaPC"] = yield_pc(tbl.nAlpha, baseAlphaPC)

tbl = tbl.merge(ll, left_on="zip5", right_on="zipcode", how="left").dropna(subset=["lat", "lon"])

def num(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)

recs = [dict(zip=r.zip5, lat=round(r.lat,4), lon=round(r.lon,4),
             n=int(r.n), nPassed=int(r.nPassed), nAlpha=int(r.nAlpha),
             pop=None if np.isnan(r["pop"]) else int(r["pop"]),
             rawPass=num(r.rawPass,4), pPass=num(r.pPass,4), orPass=num(r.orPass),
             rawAlpha=num(r.rawAlpha,4), pAlpha=num(r.pAlpha,4), orAlpha=num(r.orAlpha),
             rawPassPC=num(r.rawPassPC), pcPass=num(r.pcPass), orPassPC=num(r.orPassPC),
             rawAlphaPC=num(r.rawAlphaPC), pcAlpha=num(r.pcAlpha), orAlphaPC=num(r.orAlphaPC),
             agi=num(r.agi,1), rss=r.rss)
        for _, r in tbl.iterrows()]

json.dump(dict(zips=recs, priorPass=round(prior_pass,4), priorAlpha=round(prior_alpha,4),
               basePassPC=round(basePassPC,4), baseAlphaPC=round(baseAlphaPC,4), k=K, a=A,
               totalApplicants=int(len(d)), totalPassed=int(d.passed.sum()),
               totalAlpha=int(sub.alpha.sum())), open("zipmap.json","w"))

print(f"zips {len(recs)} | pop coverage {tbl['pop'].notna().mean():.1%}")
print(f"baseline yield per 1,000 residents:  passers {basePassPC:.3f}   alphas {baseAlphaPC:.3f}")
print("\ntop 8 by per-capita passer yield (pop>=5000):")
q = tbl[tbl["pop"] >= 5000].nlargest(8, "pcPass")
print(q[["zip5","rss","n","nPassed","pop","rawPassPC","pcPass","orPassPC"]].round(3).to_string(index=False))
