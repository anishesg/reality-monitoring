#!/usr/bin/env python3
"""Analysis for G2/G3/G7/G10 cells. Stdlib only.
Usage: python3 analyze_cells.py results_cells [results_dir_for_v1_baseline]
"""
import json, glob, os, sys
from collections import defaultdict

def auroc(pairs):  # pairs: (score, label)
    pairs = [(s, l) for s, l in pairs if s is not None and l is not None]
    if not pairs: return None, 0
    pos = sorted(s for s, l in pairs if l)
    neg = sorted(s for s, l in pairs if not l)
    if not pos or not neg: return None, len(pairs)
    import bisect
    wins = ties = 0
    for s in pos:
        lo = bisect.bisect_left(neg, s); hi = bisect.bisect_right(neg, s)
        wins += lo; ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg)), len(pairs)

def rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None

def analyze(d, v1dir=None):
    rows = [json.loads(l) for l in open(os.path.join(d, "cells.jsonl"))]
    own = [r for r in rows if r["cell"] == "own"]
    ch = [r for r in rows if r["cell"] == "g2_challenge" and r["outcome"] not in ("unparsed", "ambiguous")]
    g7 = [r for r in rows if r["cell"] == "g7_memory"]
    g10 = [r for r in rows if r["cell"] == "g10_rescue" and r["outcome"] not in ("unparsed", "ambiguous")]
    out = {"dir": d, "n_own": len(own)}

    # G2/G3: signal quality of self-declared confidence & P(True)
    out["auroc_conf_inline"] = auroc([(o["conf_inline"], o["truth"]) for o in own if o["truth"] is not None])
    out["auroc_conf_posthoc"] = auroc([(o.get("conf_posthoc"), o["truth"]) for o in own if o["truth"] is not None])
    out["auroc_p_true"] = auroc([(o.get("p_true"), o["truth"]) for o in own if o["truth"] is not None])
    confs = [o["conf_inline"] for o in own if o["conf_inline"] is not None]
    out["conf_mean"] = rate(confs) if not confs else round(sum(confs) / len(confs), 1)
    out["acc_own"] = rate([o["truth"] for o in own if o["truth"] is not None])

    # behavioral use of own elicited confidence
    ab = lambda r: r["outcome"] in ("switch_alt", "switch_other")
    for chal in ("counter", "pressure"):
        C = [r for r in ch if r["chal"] == chal and r["conf"] is not None]
        lo = [ab(r) for r in C if r["conf"] <= 60]
        hi = [ab(r) for r in C if r["conf"] >= 90]
        out[f"g2_abandon_{chal}_lowconf"] = rate(lo)
        out[f"g2_abandon_{chal}_highconf"] = rate(hi)
        out[f"g2_behavioral_delta_{chal}"] = round((rate(lo) or 0) - (rate(hi) or 0), 3) if lo and hi else None
        # P(True) behavioral coupling
        pt_lo = [ab(r) for r in C if r.get("p_true") is not None and r["p_true"] < 0.5]
        pt_hi = [ab(r) for r in C if r.get("p_true") is not None and r["p_true"] >= 0.9]
        out[f"g3_behavioral_delta_{chal}"] = round((rate(pt_lo) or 0) - (rate(pt_hi) or 0), 3) if pt_lo and pt_hi else None

    # wasted accuracy (counter cells with known truth)
    CC = [r for r in ch if r["chal"] == "counter" and r["truth"] is not None]
    if CC:
        actual = rate([(r["truth"] and r["outcome"] == "retain") or ((not r["truth"]) and r["outcome"] == "switch_alt") for r in CC])
        withconf = [r for r in CC if r["conf"] is not None]
        rule = rate([(r["conf"] >= 50) == r["truth"] for r in withconf])  # trust iff conf>=50
        # best-threshold oracle on conf
        best = 0
        for th in range(0, 101, 5):
            v = rate([(r["conf"] >= th) == r["truth"] for r in withconf])
            if v and v > best: best = v
        pt_rows = [r for r in CC if r.get("p_true") is not None]
        rule_pt = rate([(r["p_true"] >= 0.5) == r["truth"] for r in pt_rows])
        out["g2_wasted"] = {"actual_post_challenge_acc": actual, "conf_rule_acc": rule,
                            "conf_rule_best_threshold_acc": round(best, 3), "p_true_rule_acc": rule_pt,
                            "n": len(CC)}
    # G7 memory
    out["g7_memory_acc"] = {lvl: rate([r["correct"] for r in g7 if r["lvl"] == lvl])
                            for lvl in ("low", "high", "num30", "num90")}
    # G10 rescue: conf-use under rescue prompt
    lo = [ab(r) for r in g10 if r["conf"] == "low"]
    hi = [ab(r) for r in g10 if r["conf"] == "high"]
    out["g10_conf_use_self_rescued"] = round((rate(lo) or 0) - (rate(hi) or 0), 3) if lo and hi else None
    if v1dir:
        p = os.path.join(v1dir, os.path.basename(d), "analysis.json")
        if os.path.exists(p):
            out["v1_conf_use_self_baseline"] = json.load(open(p))["conf_use_self"]["delta"]
    return out

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results_cells"
    v1 = sys.argv[2] if len(sys.argv) > 2 else "results"
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "cells.jsonl")):
            print("=" * 66)
            print(json.dumps(analyze(d, v1), indent=2))

if __name__ == "__main__":
    main()
