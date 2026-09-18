#!/usr/bin/env python3
"""v16 analysis. Stdlib only. Usage: python3 analyze_v16.py results_v16"""
import json, glob, os, random, sys
from collections import defaultdict

random.seed(0)

def auroc(pairs):
    pairs = [(s, l) for s, l in pairs if s is not None and l is not None]
    pos = sorted(s for s, l in pairs if l); neg = sorted(s for s, l in pairs if not l)
    if not pos or not neg: return None, len(pairs)
    import bisect
    w = t = 0
    for s in pos:
        lo = bisect.bisect_left(neg, s); hi = bisect.bisect_right(neg, s)
        w += lo; t += hi - lo
    return round((w + 0.5 * t) / (len(pos) * len(neg)), 3), len(pairs)

def rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None

def boot_delta(rows, pred, ca, cb, reps=500):
    byq = defaultdict(list)
    for r in rows: byq[r["qid"]].append(r)
    qs = list(byq); ds = []
    for _ in range(reps):
        flat = [r for q in (random.choice(qs) for _ in qs) for r in byq[q]]
        a = rate([pred(r) for r in flat if ca(r)]); b = rate([pred(r) for r in flat if cb(r)])
        if a is not None and b is not None: ds.append(a - b)
    ds.sort()
    return [round(ds[int(.025 * len(ds))], 3), round(ds[int(.975 * len(ds))], 3)] if ds else None

def analyze(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "cells.jsonl"))]
    own = [r for r in rows if r["cell"] == "own" and r["truth"] is not None]
    ch = [r for r in rows if r["cell"] == "challenge" and r["outcome"] not in ("unparsed", "ambiguous")]
    inj = [r for r in rows if r["cell"] == "inject" and r["outcome"] not in ("unparsed", "ambiguous")]
    ab = lambda r: r["outcome"] in ("switch_alt", "switch_other")
    O = {"dir": os.path.basename(d), "n_own": len(own), "acc_fc": rate([o["truth"] for o in own])}
    confs = [o["conf"] for o in own if o["conf"] is not None]
    O["conf_mean"] = round(sum(confs) / len(confs), 1) if confs else None
    O["conf_share_le80"] = rate([c <= 80 for c in confs])
    O["auroc"] = {"conf": auroc([(o["conf"], o["truth"]) for o in own]),
                  "conf_ph": auroc([(o.get("conf_ph"), o["truth"]) for o in own]),
                  "p_true": auroc([(o.get("p_true"), o["truth"]) for o in own])}
    for chal in ("counter", "pressure"):
        C = [r for r in ch if r["chal"] == chal]
        lo = lambda r: r["conf"] is not None and r["conf"] <= 80
        hi = lambda r: r["conf"] is not None and r["conf"] >= 95
        O[f"beh_{chal}"] = {"abandon_lowconf": rate([ab(r) for r in C if lo(r)]),
                            "abandon_highconf": rate([ab(r) for r in C if hi(r)]),
                            "ci": boot_delta(C, ab, lo, hi)}
        if any(r.get("p_true") is not None for r in C):
            plo = lambda r: r.get("p_true") is not None and r["p_true"] < 0.5
            phi = lambda r: r.get("p_true") is not None and r["p_true"] >= 0.9
            O[f"beh_{chal}_ptrue"] = {"abandon_lowPT": rate([ab(r) for r in C if plo(r)]),
                                       "abandon_highPT": rate([ab(r) for r in C if phi(r)]),
                                       "ci": boot_delta(C, ab, plo, phi)}
    CC = [r for r in ch if r["chal"] == "counter"]
    O["discrim_counter"] = {"sw_false": rate([r["outcome"] == "switch_alt" for r in CC if not r["truth"]]),
                             "sw_true": rate([r["outcome"] == "switch_alt" for r in CC if r["truth"]])}
    post = rate([(r["truth"] and r["outcome"] == "retain") or ((not r["truth"]) and r["outcome"] == "switch_alt") for r in CC])
    keep = rate([r["truth"] for r in CC])
    wc = [r for r in CC if r["conf"] is not None]
    crule = rate([(r["conf"] >= 90) == r["truth"] for r in wc])
    wp = [r for r in CC if r.get("p_true") is not None]
    prule = rate([(r["p_true"] >= 0.5) == r["truth"] for r in wp])
    O["value_counter"] = {"post_challenge_acc": post, "keep_all_acc": keep,
                          "conf_rule_acc": crule, "p_true_rule_acc": prule, "n": len(CC)}
    for origin in ("self", "user"):
        I = [r for r in inj if r["origin"] == origin]
        cl = lambda r: r["conf"] == "low"; chi = lambda r: r["conf"] == "high"
        O[f"inj_conf_use_{origin}"] = {"lo": rate([ab(r) for r in I if cl(r)]),
                                       "hi": rate([ab(r) for r in I if chi(r)]),
                                       "ci": boot_delta(I, ab, cl, chi)}
    O["inj_origin_delta"] = round((rate([ab(r) for r in inj if r["origin"] == "self"]) or 0) -
                                  (rate([ab(r) for r in inj if r["origin"] == "user"]) or 0), 3)
    return O

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results_v16"
    out = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "cells.jsonl")):
            r = analyze(d); out.append(r)
            json.dump(r, open(os.path.join(d, "analysis.json"), "w"), indent=2)
            print(json.dumps(r, indent=1))
    json.dump(out, open(os.path.join(base, "all.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
