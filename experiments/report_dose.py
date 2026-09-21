#!/usr/bin/env python3
import json, glob, os
def rate(xs): return round(sum(xs)/len(xs), 3) if xs else None
ab = lambda v: v in ("switch_alt", "switch_other")
print("arm             pressure-capitulation   (base -> DPO dose ->)")
for d in sorted(glob.glob("results_v20/dpo_*")):
    f = os.path.join(d, "dose.jsonl")
    if not os.path.exists(f): continue
    R = [json.loads(l) for l in open(f)]
    seen = []
    for r in R:
        if r["cond"] not in seen: seen.append(r["cond"])
    order = ["base"] + sorted([c for c in seen if c != "base"],
                              key=lambda x: int(x.split("-")[-1]) if "-" in x else 0)
    def val(cond, kind):
        xs = [ab(r["outcome"]) for r in R if r["cond"] == cond and r["kind"] == kind
              and r["outcome"] not in ("unparsed", "amb")]
        return rate(xs)
    def short(c):
        return "base" if c == "base" else c.split("-")[-1]
    press = "  ".join(f"{short(c)}:{val(c,'pressure')}" for c in order)
    bare  = "  ".join(f"{short(c)}:{val(c,'counter_bare')}" for c in order)
    name = os.path.basename(d)
    print(f"{name:15s} pressure   {press}")
    print(f"{'':15s} bare-prime {bare}")
