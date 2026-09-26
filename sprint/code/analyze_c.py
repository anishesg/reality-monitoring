#!/usr/bin/env python3
"""Exp C/E readout: post-challenge accuracy by model x note condition; paired item bootstrap.
usage: analyze_c.py trials1.jsonl [trials2.jsonl ...]"""
import json, sys
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
acc = defaultdict(dict)       # (model, cond) -> id -> correct
sw = defaultdict(lambda: defaultdict(list))
init = defaultdict(dict)
for r in rows:
    if r["final_correct"] is None:
        continue
    acc[(r["model"], r["cond"])][r["id"]] = int(r["final_correct"])
    sw[(r["model"], r["cond"])]["right" if r["right0"] else "wrong"].append(int(r["outcome"] != "retain"))
    init[r["model"]][r["id"]] = int(r["right0"])
models = sorted({m for m, _ in acc})
conds = ["none", "raw", "calib", "shuffled", "oracle"]


def group(prefix):
    return [m for m in models if m.startswith(prefix)]


def pooled(ms, cond):
    """item -> mean correct across seeds in ms"""
    d = defaultdict(list)
    for m in ms:
        for k, v in acc[(m, cond)].items():
            d[k].append(v)
    return {k: np.mean(v) for k, v in d.items()}


def paired(a, b, n=4000, seed=0):
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] - b[k] for k in ks])
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(ks)


print("model        init   " + "  ".join(f"{c:>8}" for c in conds) + "   | switch|right  switch|wrong (calib)")
for m in models:
    i0 = np.mean(list(init[m].values()))
    line = "  ".join(f"{np.mean(list(acc[(m, c)].values())):8.3f}" if acc[(m, c)] else "     nan" for c in conds)
    s = sw[(m, "calib")]
    print(f"{m:12s} {i0:.3f}  {line}   | {np.mean(s['right']) if s['right'] else float('nan'):.3f}  {np.mean(s['wrong']) if s['wrong'] else float('nan'):.3f}")

prefixes = sorted({m.rsplit("_s", 1)[0] for m in models})
print("\nPooled over seeds (paired item bootstrap, 95% CI):")
out = {}
base_none = pooled(["base"], "none") if "base" in models else None
for p in prefixes:
    ms = [m for m in models if m.rsplit("_s", 1)[0] == p]
    P = {c: pooled(ms, c) for c in conds}
    res = {c: float(np.mean(list(P[c].values()))) for c in conds if P[c]}
    for a, b in [("calib", "shuffled"), ("raw", "shuffled"), ("calib", "none"), ("oracle", "calib")]:
        if P[a] and P[b]:
            d = paired(P[a], P[b]); res[f"{a}-{b}"] = d
            print(f"  {p:10s} {a:>7}-{b:<8} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] n={d[3]}")
    if base_none and p != "base" and P["calib"]:
        d = paired(P["calib"], base_none); res["calib-base_none"] = d
        print(f"  {p:10s} calib - base(no note) {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
    out[p] = res
# cross-arm comparisons (Exp E): own vs shuffled arms under the same condition
for a, b in [("own", "shuffled"), ("conf", "ctrl"), ("own", "base"), ("shuffled", "base"), ("conf", "base")]:
    if a in out and b in out:
        for c in ("calib", "raw", "none", "shuffled", "oracle"):
            A = pooled([m for m in models if m.rsplit("_s", 1)[0] == a], c)
            B = pooled([m for m in models if m.rsplit("_s", 1)[0] == b], c if b != "base" else c)
            if A and B:
                d = paired(A, B)
                print(f"  ARM {a} vs {b} [{c}]: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] n={d[3]}")
                out[f"{a}_vs_{b}_{c}"] = d
json.dump(out, open("summary_c_last.json", "w"), default=float, indent=1)
