#!/usr/bin/env python3
"""Repair-2.0 six-criteria report across seeds/arms."""
import json, glob, os
from collections import defaultdict

def rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None

ab = lambda r: r["outcome"] in ("switch_alt", "switch_other")

for d in sorted(glob.glob("results_v22/*")):
    f = os.path.join(d, "eval.jsonl")
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("outcome") not in ("unparsed", "ambiguous")]
    mono = [r for r in rows if r["cell"] == "mono"]
    bid = [r for r in rows if r["cell"] == "bidir"]
    ownc = [r for r in rows if r["cell"] == "own"]
    cap = [r for r in rows if r["cell"] == "capability"]
    print("=" * 64)
    print(d)
    # C1 monotonicity at UNSEEN values/templates (near+far separately)
    for far in (False, True):
        curve = []
        for pe in sorted({r["p_eff"] for r in mono}):
            v = rate([ab(r) for r in mono if r["p_eff"] == pe and r["far"] == far])
            curve.append((pe, v))
        print(f"  mono {'far' if far else 'near'}: " + " ".join(f"{p}:{v}" for p, v in curve))
    # C2 equivalence: same p_eff via different valence
    eq = defaultdict(dict)
    for r in mono:
        eq[(r["qid"], r["p_eff"], r["far"])].setdefault(r["valence"], []).append(ab(r))
    pairs = [(rate(v["correct"]), rate(v["incorrect"])) for v in eq.values()
             if "correct" in v and "incorrect" in v]
    if pairs:
        diffs = [abs(a - b) for a, b in pairs if a is not None and b is not None]
        print(f"  equivalence mean|diff|={round(sum(diffs)/len(diffs),3)} n={len(diffs)}")
    # C3 bidirectional (no note): retention + correction
    ret = rate([not ab(r) for r in bid if r["chal"] == "counter" and r["truth"]])
    acc = rate([ab(r) for r in bid if r["chal"] == "counter" and not r["truth"]])
    prs = rate([not ab(r) for r in bid if r["chal"] == "pressure" and r["truth"]])
    print(f"  bidir(no-note): retainOK={ret} acceptFix={acc} pressResist={prs}")
    # C4 own elicited reliability -> use
    lo = [r for r in ownc if r.get("rel") is not None and r["rel"] <= 60]
    hi = [r for r in ownc if r.get("rel") is not None and r["rel"] >= 85]
    post = rate([(r["truth"] and not ab(r)) or ((not r["truth"]) and r["outcome"] == "switch_alt") for r in ownc])
    print(f"  own-conf: abandon(lowrel)={rate([ab(r) for r in lo])}(n={len(lo)}) abandon(hirel)={rate([ab(r) for r in hi])}(n={len(hi)}) postAcc={post}")
    # C5 capability
    print(f"  capability FC acc={rate([r['outcome']=='retain' for r in cap])} (n={len(cap)})")
