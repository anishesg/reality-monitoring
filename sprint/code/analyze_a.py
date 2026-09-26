#!/usr/bin/env python3
"""Exp A readout: dose response of switching on reliability value, by authorship x challenge x claim truth.
Slope = OLS slope of switch on v/100 (per 100 points) with item-cluster bootstrap CI; plus low(20,40)-high(80,95)."""
import glob, json, os
import numpy as np

out = {}
for d in sorted(glob.glob("res_a/*")):
    f = os.path.join(d, "trials.jsonl")
    if not os.path.exists(f):
        continue
    tag = os.path.basename(d)
    T = [json.loads(l) for l in open(f) if '"unparsed"' not in l]
    print("=" * 16, tag, len(T))
    for ch in ("weak", "pressure"):
        for truth in (True, False):
            base = [t["outcome"] != "retain" for t in T if t["chal"] == ch and t["truth"] == truth and t["auth"] == "none"]
            parts = [f"[{ch:8s} claim={'T' if truth else 'F'}] none {np.mean(base):.2f} |"]
            for auth in ("latent", "surfaced"):
                xs = [t for t in T if t["chal"] == ch and t["truth"] == truth and t["auth"] == auth]
                curve = [np.mean([t["outcome"] != "retain" for t in xs if t["v"] == v]) for v in (20, 40, 60, 80, 95)]
                by = {}
                for t in xs:
                    by.setdefault(t["qid"], []).append((t["v"] / 100, int(t["outcome"] != "retain")))
                keys = list(by); rng = np.random.default_rng(0)

                def slope(ks):
                    pts = [p for k in ks for p in by[k]]
                    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
                    return np.polyfit(x, y, 1)[0] if len(set(x)) > 1 else float("nan")
                s = slope(keys)
                bs = [slope([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(500)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                lohi = np.mean([t["outcome"] != "retain" for t in xs if t["v"] <= 40]) - np.mean([t["outcome"] != "retain" for t in xs if t["v"] >= 80])
                out[f"{tag}|{ch}|{truth}|{auth}"] = {"curve": curve, "slope": s, "lo": lo, "hi": hi, "low_minus_high": lohi, "none": float(np.mean(base))}
                parts.append(f"{auth}: " + " ".join(f"{c:.2f}" for c in curve) + f" slope {s:+.3f}[{lo:+.2f},{hi:+.2f}] lo-hi {lohi:+.3f} |")
            print(" ".join(parts))
json.dump(out, open("res_a/summary_a.json", "w"), default=float, indent=1)
