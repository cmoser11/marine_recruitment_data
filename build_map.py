import json, pathlib
d = json.load(open("zipmap.json"))
core = [r for r in d["zips"] if 31.0 <= r["lat"] <= 37.5 and -115 <= r["lon"] <= -102]
payload = dict(zips=core, priorPass=d["priorPass"], priorAlpha=d["priorAlpha"], k=d["k"],
               totalApplicants=d["totalApplicants"], totalPassed=d["totalPassed"],
               totalAlpha=d["totalAlpha"], nCore=len(core),
               nOutside=len(d["zips"]) - len(core),
               applicantsOutside=sum(r["n"] for r in d["zips"]) - sum(r["n"] for r in core))
html = pathlib.Path("map_template.html").read_text()
states = pathlib.Path("states_compact.json").read_text()
out = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
out = out.replace("/*__STATES__*/null", states)
pathlib.Path("asvab_zip_map.html").write_text(out)
print("wrote asvab_zip_map.html", len(out), "bytes |", len(core), "zips")
