#!/usr/bin/env python3
"""Exp H readout: post-challenge accuracy vs note AUROC (synthetic curve + real signals), per model."""
import glob, json, os
from collections import defaultdict
import numpy as np

rows = defaultdict(lambda: defaultdict(dict))  # model -> cond -> id -> correct
best = defaultdict(lambda: defaultdict(dict))  # model -> cond -> id -> correct under the Bayes rule (keep iff note >= 50)
meta = {}
for d in sorted(glob.glob("res_h/*")):
    if not os.path.exists(f"{d}/summary.json"):
        continue
    s = json.load(open(f"{d}/summary.json")); meta[s["model"]] = s
    for l in open(f"{d}/trials.jsonl"):
        r = json.loads(l)
        if r["final_correct"] is not None:
            rows[r["model"]][r["cond"]][r["id"]] = int(r["final_correct"])
        if r["v"] is not None:
            best[r["model"]][r["cond"]][r["id"]] = int(r["right0"] if r["v"] >= 50 else not r["right0"])


def ci(d, n=2000):
    x = np.array(list(d.values()), float); rng = np.random.default_rng(0)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


out = {}
for m in rows:
    R = meta[m]["realized_auroc"]
    print(f"===== {m}  n={meta[m]['n']}  initial acc={meta[m]['base_rate']:.3f}")
    none = ci(rows[m]["none"]); orc = ci(rows[m]["oracle"])
    print(f"  none   acc {none[0]:.3f}   oracle acc {orc[0]:.3f}")
    syn = defaultdict(list)
    for c in rows[m]:
        if c.startswith("syn_"):
            A = float(c.split("_")[1]); syn[A].append((R[c], np.mean(list(rows[m][c].values())), np.mean(list(best[m][c].values()))))
    curve = []; bcurve = []
    keep = meta[m]["base_rate"]
    for A in sorted(syn):
        au = np.mean([x[0] for x in syn[A]]); ac = np.mean([x[1] for x in syn[A]]); bb = np.mean([x[2] for x in syn[A]])
        curve.append((au, ac)); bcurve.append((au, bb))
        eff = (ac - keep) / (bb - keep) if bb - keep >= 0.02 else float("nan")
        print(f"  syn target {A:.2f}  realized AUROC {au:.3f}  acc {ac:.3f}  best-possible {bb:.3f}  keep-all {keep:.3f}  efficiency {eff:+.2f}")
    real = {}
    for c in rows[m]:
        if c.startswith("real_"):
            a, lo, hi = ci(rows[m][c]); real[c[5:]] = (R.get(c), a, lo, hi)
            # predicted accuracy from the synthetic curve at the same AUROC
            xs, ys = zip(*curve); pred = float(np.interp(R.get(c), xs, ys))
            print(f"  real {c[5:]:7s} AUROC {R.get(c):.3f}  acc {a:.3f} [{lo:.3f},{hi:.3f}]  curve-predicted {pred:.3f}")
    if len(curve) > 2:
        xs, ys = zip(*curve); slope = np.polyfit(xs, ys, 1)[0]
        print(f"  synthetic slope: {slope:+.3f} accuracy per unit AUROC (0.5->1.0 span {slope*0.5:+.3f})")
    out[m] = {"none": none, "oracle": orc, "curve": curve, "best_curve": bcurve, "real": real, "initial": meta[m]["base_rate"]}
json.dump(out, open("res_h/summary_h.json", "w"), default=float, indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    styles = {"own": ("C0", "own-confidence trained"), "shuffled": ("C1", "shuffled-confidence trained"),
              "rule": ("C2", "rule-trained (threshold)"), "base": ("0.4", "untrained base")}
    done = set()
    for m, o in sorted(out.items()):
        key = m.split("_")[0]; col, lab = styles.get(key, ("k", m))
        xs, ys = zip(*o["curve"])
        ax.plot(xs, ys, "-", color=col, alpha=0.7, lw=1.2, label=None if key in done else lab)
        for sig, (au, a, lo, hi) in o["real"].items():
            if au is None: continue
            ax.errorbar([au], [a], yerr=[[a - lo], [hi - a]], fmt="o", color=col, ms=4, capsize=2)
            if key == "own" and m.endswith("s0"):
                ax.annotate(sig, (au, a), textcoords="offset points", xytext=(4, -9), fontsize=7)
        done.add(key)
    bm = next((o for mm, o in out.items() if mm.startswith("own")), None)
    if bm:
        xs, ys = zip(*bm["best_curve"]); ax.plot(xs, ys, "k--", lw=1, label="best possible (Bayes rule)")
        ax.axhline(bm["initial"], color="k", lw=0.6, ls=":", label="keep every answer")
    ax.set_xlabel("AUROC of the confidence note (vs. own correctness)"); ax.set_ylabel("post-challenge accuracy")
    ax.set_xlim(0.48, 1.01); ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig("res_h/fig_monitor_curve.pdf"); fig.savefig("res_h/fig_monitor_curve.png", dpi=160)
    print("figure -> res_h/fig_monitor_curve.pdf")
except Exception as e:
    print("figure skipped:", e)
