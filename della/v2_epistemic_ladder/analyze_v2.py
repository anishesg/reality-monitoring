#!/usr/bin/env python3
"""Aggregate v2 results across models. Stdlib only. Run on login node.
Usage: python analyze_v2.py [results_dir]
Writes results_summary.md + all_results.json
"""
import json, glob, os, random, sys, re
from collections import defaultdict

random.seed(0)

def abandon(t): return t["outcome"] in ("switch_alt", "switch_other")
def switch_alt(t): return t["outcome"] == "switch_alt"

def rate(T, pred, cond):
    xs = [pred(t) for t in T if cond(t)]
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)

def boot(T, pred, ca, cb, reps=600):
    byq = defaultdict(list)
    for t in T: byq[t["qid"]].append(t)
    qids = list(byq); diffs = []
    for _ in range(reps):
        flat = [t for q in (random.choice(qids) for _ in qids) for t in byq[q]]
        diffs.append(rate(flat, pred, ca)[0] - rate(flat, pred, cb)[0])
    diffs = [d for d in diffs if d == d]
    diffs.sort()
    if not diffs: return [float("nan")] * 2
    return [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]]

def d3(x): return round(x, 3) if isinstance(x, float) else x

def analyze(path):
    T0 = [json.loads(l) for l in open(os.path.join(path, "trials.jsonl"))]
    know = {}
    fc = os.path.join(path, "baseline_fc.jsonl")
    if os.path.exists(fc):
        know = {t["qid"]: t["knows"] for t in (json.loads(l) for l in open(fc))}
    T = [t for t in T0 if t["outcome"] not in ("unparsed", "ambiguous")]
    for t in T: t["knows"] = know.get(t["qid"], None)
    inj = [t for t in T if t["origin"] != "self_genuine"]
    R = {"dir": path, "n_analyzed": len(T),
         "excluded_frac": 1 - len(T) / max(len(T0), 1),
         "baseline_fc_acc": sum(know.values()) / max(len(know), 1) if know else None}

    ctr = lambda t: t["chal"] == "counter"
    R["delta_revision"] = {}
    a = lambda t: ctr(t) and not t["truth"]; b = lambda t: ctr(t) and t["truth"]
    pa, na = rate(inj, switch_alt, a); pb, nb = rate(inj, switch_alt, b)
    R["delta_revision"]["all"] = {"sw_false": d3(pa), "sw_true": d3(pb), "delta": d3(pa - pb),
                                   "ci": [d3(x) for x in boot(inj, switch_alt, a, b)], "n": [na, nb]}
    # state-use contrasts per origin
    contrasts = [("low_vs_high", "low", "high"), ("n30_vs_n90", "num30", "num90"),
                 ("guess_vs_verified", "guess", "verified")]
    for origin in ("self", "user", "other_ai", "user_matched", "other_ai_matched"):
        for tv, tlab in ((None, ""), (True, "_harmful"), (False, "_beneficial")):   # pooled (legacy) + split by truth (audit issue 2)
            O = [t for t in inj if t["origin"] == origin and (tv is None or t["truth"] == tv)]
            if not O: continue
            ent = {}
            for name, s_lo, s_hi in contrasts:
                ca = lambda t, s=s_lo: t["conf"] == s
                cb = lambda t, s=s_hi: t["conf"] == s
                pl, nl = rate(O, abandon, ca); ph, nh = rate(O, abandon, cb)
                ent[name] = {"ab_lo": d3(pl), "ab_hi": d3(ph), "delta": d3(pl - ph),
                             "ci": [d3(x) for x in boot(O, abandon, ca, cb)], "n": [nl, nh]}
            R[f"state_use_{origin}{tlab}"] = ent
    # origin contrasts (injected only, matched cells)
    for oa, ob in (("self", "user"), ("self", "other_ai"), ("user", "other_ai"), ("self", "user_matched"), ("self", "other_ai_matched")):
        for tv, tlab in ((None, ""), (True, "_harmful")):
            I = [t for t in inj if tv is None or t["truth"] == tv]
            ca = lambda t, o=oa: t["origin"] == o
            cb = lambda t, o=ob: t["origin"] == o
            p1, n1 = rate(I, abandon, ca); p2, n2 = rate(I, abandon, cb)
            if not n1 or not n2: continue
            R[f"origin_{oa}_vs_{ob}{tlab}"] = {"a": d3(p1), "b": d3(p2), "delta": d3(p1 - p2),
                                               "ci": [d3(x) for x in boot(I, abandon, ca, cb)], "n": [n1, n2]}
    # genuine vs injected self (conf none)
    g = [t for t in T if t["origin"] == "self_genuine"]
    if g:
        si = lambda t: t["origin"] == "self" and t["conf"] == "none"
        sg = lambda t: t["origin"] == "self_genuine"
        p1, n1 = rate(T, abandon, sg); p2, n2 = rate(T, abandon, si)
        R["genuine_vs_injected_self"] = {"genuine": d3(p1), "injected": d3(p2), "n": [n1, n2]}
    # sycophancy: abandon TRUE claims under pressure, per origin
    R["pressure_abandon_true"] = {o: d3(rate(inj, abandon, lambda t, o=o: t["chal"] == "pressure" and t["truth"] and t["origin"] == o)[0])
                                  for o in ("self", "user", "other_ai")}
    # per-domain + held-out template + knowledge strat for the key self contrast
    for key, cond_extra in [("by_domain", lambda t, v: t["domain"] == v),
                            ("by_template", lambda t, v: t["tmpl"] == v)]:
        vals = sorted({t["domain" if key == "by_domain" else "tmpl"] for t in inj})
        ent = {}
        for v in vals:
            S = [t for t in inj if cond_extra(t, v) and t["origin"] == "self"]
            pl, _ = rate(S, abandon, lambda t: t["conf"] == "low")
            ph, _ = rate(S, abandon, lambda t: t["conf"] == "high")
            ent[v] = d3(pl - ph)
        R[f"conf_use_self_{key}"] = ent
    if know:
        for kv in (True, False):
            S = [t for t in inj if t["origin"] == "self" and t["knows"] == kv]
            pl, _ = rate(S, abandon, lambda t: t["conf"] == "low")
            ph, _ = rate(S, abandon, lambda t: t["conf"] == "high")
            R[f"conf_use_self_knows_{kv}"] = d3(pl - ph)
    return R

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "trials.jsonl")):
            print("analyzing", d, flush=True)
            out.append(analyze(d))
    with open("all_results.json", "w") as f:
        json.dump(out, f, indent=2)
    # markdown emergence table
    def size_of(name):
        m = re.findall(r"(\d+(?:p\d+)?)[bB]", name)
        return float(m[-1].replace("p", ".")) if m else 0
    rows = sorted(out, key=lambda r: size_of(r["dir"]))
    with open("results_summary.md", "w") as f:
        f.write("| model | fc_acc | Δrev | self lo-hi (harmful) | self 30-90 (harmful) | self guess-ver (harmful) | user lo-hi (harmful) | otherAI lo-hi (harmful) | self-vs-user (unmatched) | self-vs-user MATCHED (harmful) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            g = lambda *ks: (lambda d: d if not isinstance(d, dict) else d.get("delta"))(
                __import__("functools").reduce(lambda a, k: a.get(k, {}) if isinstance(a, dict) else {}, ks, r)) or ""
            f.write(f"| {os.path.basename(r['dir'])} | {r.get('baseline_fc_acc')} | "
                    f"{g('delta_revision','all')} | {g('state_use_self_harmful','low_vs_high')} | "
                    f"{g('state_use_self_harmful','n30_vs_n90')} | {g('state_use_self_harmful','guess_vs_verified')} | "
                    f"{g('state_use_user_harmful','low_vs_high')} | {g('state_use_other_ai_harmful','low_vs_high')} | "
                    f"{g('origin_self_vs_user')} | {g('origin_self_vs_user_matched_harmful')} |\n")
    print("wrote all_results.json, results_summary.md")

if __name__ == "__main__":
    main()
