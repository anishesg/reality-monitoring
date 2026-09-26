#!/usr/bin/env python3
"""Exp B readout: switch rates by cell x challenge, near/far contrast with item-cluster bootstrap."""
import glob, json, os, sys
import numpy as np

OK = ("retain", "switch_alt", "switch_true", "switch_other")


def boot_diff(a, b, n=2000, seed=0):
    """a, b: dict qid -> list of 0/1. Returns point, lo, hi for mean(a)-mean(b), resampling qids."""
    rng = np.random.default_rng(seed)
    ka, kb = list(a), list(b)
    if not ka or not kb:
        return float("nan"), float("nan"), float("nan")
    ma = np.array([np.mean(a[k]) for k in ka]); mb = np.array([np.mean(b[k]) for k in kb])
    pt = ma.mean() - mb.mean()
    ds = [ma[rng.integers(0, len(ma), len(ma))].mean() - mb[rng.integers(0, len(mb), len(mb))].mean() for _ in range(n)]
    return pt, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def ci(xs, n=2000, seed=0):
    xs = np.array(xs, float)
    if len(xs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(n)]
    return xs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


root = sys.argv[1] if len(sys.argv) > 1 else "res_b"
summary = {}
for d in sorted(glob.glob(f"{root}/*")):
    f = os.path.join(d, "trials.jsonl")
    if not os.path.exists(f):
        continue
    tag = os.path.basename(d)
    T = [json.loads(l) for l in open(f)]
    items = [json.loads(l) for l in open(os.path.join(d, "items.jsonl"))] if os.path.exists(os.path.join(d, "items.jsonl")) else []
    acc = np.mean([it["correct"] for it in items if it["correct"] is not None]) if items else float("nan")
    parsed = np.mean([it["own"] is not None for it in items]) if items else float("nan")
    S = {"own_acc": acc, "own_parsed": parsed, "n_trials": len(T)}
    print("=" * 20, tag, f"own-answer acc {acc:.3f} parsed {parsed:.3f} trials {len(T)}")
    val = {}
    for c in ("FT", "FFnear", "FFfar", "TFnear", "TFfar"):
        v = [t["picks_claim"] for t in T if t["cell"] == c and t["chal"] == "validity" and t.get("picks_claim") is not None]
        m = [t["margin"] for t in T if t["cell"] == c and t["chal"] == "validity"]
        val[c] = (np.mean(v) if v else float("nan"), np.median(m) if m else float("nan"), len(m))
    print("validity P(prefers claim) / median margin b(claim)-b(alt) / n:",
          " ".join(f"{c}:{val[c][0]:.2f}/{val[c][1]:.2f}/{val[c][2]}" for c in val))
    S["validity"] = val
    for ch in ("counter", "weak", "pressure"):
        rows = {}
        for c in ("FT", "FFnear", "FFfar", "TFnear", "TFfar"):
            xs = [t for t in T if t["cell"] == c and t["chal"] == ch]
            ok = [t for t in xs if t["outcome"] in OK]
            by = {}
            for t in ok:
                by.setdefault(t["qid"], []).append(int(t["outcome"] != "retain"))
            rows[c] = by
            sw = [t["outcome"] != "retain" for t in ok]
            s_alt = np.mean([t["outcome"] == "switch_alt" for t in ok]) if ok else float("nan")
            s_true = np.mean([t["outcome"] == "switch_true" for t in ok]) if ok else float("nan")
            unp = 1 - len(ok) / max(1, len(xs))
            m, lo, hi = ci(sw)
            S[f"{ch}_{c}"] = {"switch": m, "lo": lo, "hi": hi, "to_alt": s_alt, "to_true": s_true, "unparsed": unp, "n": len(ok)}
        line = " ".join(f"{c} {S[f'{ch}_{c}']['switch']:.3f}[{S[f'{ch}_{c}']['lo']:.2f},{S[f'{ch}_{c}']['hi']:.2f}](alt {S[f'{ch}_{c}']['to_alt']:.2f},true {S[f'{ch}_{c}']['to_true']:.2f},n{S[f'{ch}_{c}']['n']})" for c in rows)
        print(f"[{ch}] {line}")
        dff = boot_diff(rows["FFfar"], rows["FFnear"])
        dtf = boot_diff(rows["TFfar"], rows["TFnear"])
        dft = boot_diff(rows["FT"], rows["FFnear"])
        S[f"{ch}_D_FF"] = dff; S[f"{ch}_D_TF"] = dtf; S[f"{ch}_FT_minus_FFnear"] = dft
        print(f"   D_FF(far-near) {dff[0]:+.3f} [{dff[1]:+.3f},{dff[2]:+.3f}]   D_TF(far-near) {dtf[0]:+.3f} [{dtf[1]:+.3f},{dtf[2]:+.3f}]   FT-FFnear {dft[0]:+.3f} [{dft[1]:+.3f},{dft[2]:+.3f}]")
    # margin-binned switch within FF (near+far pooled), counter & weak
    for ch in ("counter", "weak"):
        ff = [t for t in T if t["cell"] in ("FFnear", "FFfar") and t["chal"] == ch and t["outcome"] in OK]
        if len(ff) > 50:
            ms = np.array([t["margin"] for t in ff]); sw = np.array([t["outcome"] != "retain" for t in ff])
            qs = np.quantile(ms, [0, .2, .4, .6, .8, 1])
            bins = [(qs[i], qs[i + 1], sw[(ms >= qs[i]) & (ms <= qs[i + 1])].mean()) for i in range(5)]
            print(f"   FF {ch} switch by margin quintile: " + " ".join(f"[{a:.2f},{b:.2f}]:{r:.2f}" for a, b, r in bins))
            S[f"{ch}_FF_quintiles"] = bins
    summary[tag] = S
json.dump(summary, open(f"{root}/summary_b.json", "w"), default=float, indent=1)
