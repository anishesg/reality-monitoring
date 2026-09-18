#!/usr/bin/env python3
"""v17 decomposition. The paper-deciding table.
Key quantities per model:
 1. Source vs repetition: abandon(counter_src) - abandon(counter_bare)   [same content, +/- source]
 2. Source-only pressure: abandon(src_only) - abandon(pressure)          [source, no content]
 3. Self-conf use per challenge kind: abandon(low)-abandon(high) for each kind
 4. MATCHED-RECENCY self vs user: conf-use interaction with origin at identical turn structure
 5. Bidirectional value: retain-correct vs accept-valid-correction (net revision quality)
"""
import json, glob, os, random, sys
from collections import defaultdict

random.seed(0)

def rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None

def boot(rows, pred, ca, cb, reps=500):
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
    T = [json.loads(l) for l in open(os.path.join(d, "cells.jsonl"))]
    T = [t for t in T if t["outcome"] not in ("unparsed", "ambiguous")]
    ab = lambda t: t["outcome"] in ("switch_alt", "switch_other")
    S = [t for t in T if t["origin"] == "self"]
    U = [t for t in T if t["origin"] == "user_recency"]
    O = {"dir": os.path.basename(d), "n": len(T)}
    k = lambda t, kk: t["kind"] == kk
    # 1 & 2: source decomposition (on self-origin, pooled conf)
    O["abandon_by_kind"] = {kk: rate([ab(t) for t in S if k(t, kk)])
                            for kk in ("counter_src", "counter_bare", "src_only", "pressure")}
    O["source_effect_content_matched"] = {"delta": None if O["abandon_by_kind"]["counter_src"] is None else
                                          round(O["abandon_by_kind"]["counter_src"] - O["abandon_by_kind"]["counter_bare"], 3),
                                          "ci": boot(S, ab, lambda t: k(t, "counter_src"), lambda t: k(t, "counter_bare"))}
    O["source_effect_no_content"] = {"delta": None if O["abandon_by_kind"]["src_only"] is None else
                                     round(O["abandon_by_kind"]["src_only"] - O["abandon_by_kind"]["pressure"], 3),
                                     "ci": boot(S, ab, lambda t: k(t, "src_only"), lambda t: k(t, "pressure"))}
    # 3: self-conf use per kind
    O["conf_use_self_by_kind"] = {}
    for kk in ("counter_src", "counter_bare", "src_only", "pressure"):
        K = [t for t in S if k(t, kk)]
        lo = rate([ab(t) for t in K if t["conf"] == "low"]); hi = rate([ab(t) for t in K if t["conf"] == "high"])
        O["conf_use_self_by_kind"][kk] = {"delta": None if lo is None or hi is None else round(lo - hi, 3),
                                          "ci": boot(K, ab, lambda t: t["conf"] == "low", lambda t: t["conf"] == "high")}
    # 4: matched-recency interaction (counter_src only)
    out = {}
    for origin, G in (("self", S), ("user_recency", U)):
        K = [t for t in G if k(t, "counter_src")]
        lo = rate([ab(t) for t in K if t["conf"] == "low"]); hi = rate([ab(t) for t in K if t["conf"] == "high"])
        out[origin] = {"abandon_lo": lo, "abandon_hi": hi,
                       "delta": None if lo is None or hi is None else round(lo - hi, 3),
                       "ci": boot(K, ab, lambda t: t["conf"] == "low", lambda t: t["conf"] == "high")}
    out["interaction_STG_matched_recency"] = None
    if out["user_recency"]["delta"] is not None and out["self"]["delta"] is not None:
        out["interaction_STG_matched_recency"] = round(out["user_recency"]["delta"] - out["self"]["delta"], 3)
    O["matched_recency"] = out
    # 5: bidirectional value (counter_src, self origin)
    K = [t for t in S if k(t, "counter_src")]
    O["bidirectional"] = {"retain_correct": rate([not ab(t) for t in K if t["truth"]]),
                          "accept_valid_correction": rate([t["outcome"] == "switch_alt" for t in K if not t["truth"]]),
                          "reject_invalid_pressure": rate([not ab(t) for t in S if k(t, "pressure") and t["truth"]])}
    return O

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results_v17"
    res = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "cells.jsonl")):
            r = analyze(d); res.append(r)
            print(json.dumps(r, indent=1))
    json.dump(res, open(os.path.join(base, "all.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
