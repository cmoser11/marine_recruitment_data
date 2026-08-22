"""Terminal entry point: mrd <command>."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .features import build
from .ingest import load
from .models import SPECS, cv_compare, fit_logit


def _load(path):
    return build(load(path))


def cmd_inspect(args):
    d = _load(args.file)
    print(f"applicants           {len(d)}")
    print(f"passed (AFQT>=31)    {int(d.passed.sum())} ({d.passed.mean():.1%})")
    s = d[d.passed == 1]
    print(f"alpha (>=50) | pass  {int(s.alpha.sum())} ({s.alpha.mean():.1%})")
    print(f"undecoded edu codes  {int((d.credential == 'UNKNOWN').sum())}")
    print(f"ZIP income matched   {(d.agi_missing == 0).mean():.1%}")
    print(f"schools / ZIP3       {d.school.nunique()} / {d.zip3.nunique()}")
    print(f"date range           {d.test_date.min():%Y-%m-%d} to {d.test_date.max():%Y-%m-%d}")


def cmd_ingest(args):
    d = _load(args.file)
    cols = [c for c in d.columns if c != "sources"]
    out = d[cols].copy()
    out["sources"] = d["sources"].str.join("; ")
    out.to_excel(args.output, index=False)
    print(f"wrote {len(out)} applicants -> {args.output}")


def cmd_model(args):
    d = _load(args.file)
    data, outcome = (d, "passed") if args.outcome == "pass" else (
        d[d.passed == 1].reset_index(drop=True), "alpha")
    _, tbl = fit_logit(SPECS[args.spec], data, outcome)
    pd.set_option("display.width", 200)
    print(tbl.round(4).to_string())
    if args.csv:
        tbl.to_csv(args.csv)
        print(f"\nwrote {args.csv}")


def cmd_compare(args):
    d = _load(args.file)
    data, outcome = (d, "passed") if args.outcome == "pass" else (
        d[d.passed == 1].reset_index(drop=True), "alpha")
    tbl, _, _ = cv_compare(data, outcome)
    pd.set_option("display.width", 200)
    print(tbl.round(4).to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mrd", description="Marine recruitment ASVAB analysis")
    sp = p.add_subparsers(dest="cmd", required=True)

    for name, fn, extra in [("inspect", cmd_inspect, []), ("ingest", cmd_ingest, ["output"]),
                            ("model", cmd_model, ["outcome", "spec", "csv"]),
                            ("compare", cmd_compare, ["outcome"])]:
        s = sp.add_parser(name)
        s.add_argument("file")
        if "output" in extra:
            s.add_argument("-o", "--output", default="applicants.xlsx")
        if "outcome" in extra:
            s.add_argument("--outcome", choices=["pass", "alpha"], default="pass")
        if "spec" in extra:
            s.add_argument("--spec", choices=list(SPECS), default="M4 +ZIP income")
        if "csv" in extra:
            s.add_argument("--csv")
        s.set_defaults(func=fn)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
