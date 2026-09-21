#!/usr/bin/env python3
import json, glob, os
def rate(xs): return round(sum(xs)/len(xs),3) if xs else None
print("model            LATENT: ab(lo) ab(hi) USE   |  SURFACED: ab(lo) ab(hi) USE   | n(lo/hi)")
for d in sorted(glob.glob("results_forced/*")):
    f = os.path.join(d,"forced.jsonl")
    if not os.path.exists(f): continue
    R = [json.loads(l) for l in open(f)]
    def use(field):
        lo = [r for r in R if r["conf"]<=70 and r[field] not in ("unparsed","amb")]
        hi = [r for r in R if r["conf"]>=90 and r[field] not in ("unparsed","amb")]
        ab = lambda rows: [x[field] in ("switch_alt","switch_other") for x in rows]
        rl, rh = rate(ab(lo)), rate(ab(hi))
        dd = round(rl-rh,3) if (rl is not None and rh is not None) else None
        return rl, rh, dd, len(lo), len(hi)
    l = use("latent_outcome"); s = use("surfaced_outcome")
    name = os.path.basename(d)
    print(f"{name:14s}   {str(l[0]):>5} {str(l[1]):>5} {str(l[2]):>6}   |   {str(s[0]):>5} {str(s[1]):>5} {str(s[2]):>6}   | {l[3]}/{l[4]}")
