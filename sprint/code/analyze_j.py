#!/usr/bin/env python3
"""Exp J readout: note effect P(act|5) - P(act|95), paired within item, per task; task x note interaction; manipulation check."""
import glob, json, os
from collections import defaultdict
import numpy as np
out = {}
for d in sorted(glob.glob("res_j/*")):
    f = f"{d}/trials.jsonl"
    if not os.path.exists(f): continue
    m = os.path.basename(d); T = [json.loads(l) for l in open(f)]
    A = defaultdict(dict)  # (task, v) -> id -> act   (v from oracle/anti only)
    lev = defaultdict(list)
    for t in T:
        if t["act"] is None: continue
        lev[(t["task"], t["cond"])].append(t["act"])
        if t["cond"] in ("oracle", "anti"):
            A[(t["task"], t["v"])][t["id"]] = t["act"]
    rng = np.random.default_rng(0)
    def eff(task):
        ks = sorted(set(A[(task, 5)]) & set(A[(task, 95)]))
        return np.array([A[(task, 5)][k] - A[(task, 95)][k] for k in ks]), ks
    ea, ka = eff("abstain"); er, kr = eff("revise")
    common = sorted(set(ka) & set(kr))
    da = {k: A[("abstain", 5)][k] - A[("abstain", 95)][k] for k in common}
    dr = {k: A[("revise", 5)][k] - A[("revise", 95)][k] for k in common}
    inter = np.array([da[k] - dr[k] for k in common])
    def bci(x): 
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]; return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
    ca, cr, ci_ = bci(ea), bci(er), bci(inter)
    chk = lev[("check", "oracle")] + lev[("check", "anti")]
    print(f"===== {m}")
    for task in ("abstain", "revise"):
        print(f"  {task:8s} " + "  ".join(f"{c}:{np.mean(lev[(task, c)]):.3f}" for c in ("none", "oracle", "anti", "shuffled")))
    print(f"  note effect P(act|5%)-P(act|95%): abstain {ca[0]:+.3f} [{ca[1]:+.3f},{ca[2]:+.3f}] (n={len(ea)})   revise {cr[0]:+.3f} [{cr[1]:+.3f},{cr[2]:+.3f}] (n={len(er)})")
    print(f"  interaction (abstain - revise): {ci_[0]:+.3f} [{ci_[1]:+.3f},{ci_[2]:+.3f}]   manipulation check recall (|recalled-v|<=2): {np.mean(chk):.3f} (n={len(chk)})")
    out[m] = {"abstain_eff": ca, "revise_eff": cr, "interaction": ci_, "check": float(np.mean(chk)),
              "levels": {f"{t}|{c}": float(np.mean(v)) for (t, c), v in lev.items()}}
json.dump(out, open("res_j/summary_j.json", "w"), default=float, indent=1)
