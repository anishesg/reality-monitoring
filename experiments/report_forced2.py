#!/usr/bin/env python3
import json, glob, os
def rate(xs): return round(sum(xs)/len(xs),3) if xs else None
ab = lambda v: v in ("switch_alt","switch_other")
print("model          LATENT use(lo-hi)   SURFACED use(lo-hi)   | latent by value        surfaced by value")
for d in sorted(glob.glob("results_forced2/*")):
    f=os.path.join(d,"forced2.jsonl")
    if not os.path.exists(f): continue
    R=[json.loads(l) for l in open(f)]
    def use(field):
        lo=[ab(r[field]) for r in R if r["v"]<=40 and r[field] not in ("unparsed","amb")]
        hi=[ab(r[field]) for r in R if r["v"]>=80 and r[field] not in ("unparsed","amb")]
        rl,rh=rate(lo),rate(hi)
        return (round(rl-rh,3) if rl is not None and rh is not None else None), rl, rh
    lu=use("latent_outcome"); su=use("surfaced_outcome")
    # monotonic curve across values for surfaced
    def curve(field):
        out=[]
        for v in [20,30,40,60,80,95]:
            xs=[ab(r[field]) for r in R if r["v"]==v and r[field] not in ("unparsed","amb")]
            out.append(f"{v}:{rate(xs)}")
        return " ".join(out)
    print(f"{os.path.basename(d):13s}  {str(lu[0]):>6} ({lu[1]},{lu[2]})   {str(su[0]):>6} ({su[1]},{su[2]})")
    print(f"                 latent   {curve('latent_outcome')}")
    print(f"                 surfaced {curve('surfaced_outcome')}")
