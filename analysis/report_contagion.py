#!/usr/bin/env python3
"""Report for harness/contagion.py runs. Cluster-bootstrap CIs over questions; chain curve measured vs Markov-predicted.
  python3 analysis/report_contagion.py results_contagion/qwen7b [more runs...]   -> prints tables, writes <run>/summary.json
Markov prediction: p_ww = P(agent wrong | previous agent wrong), p_cw = P(agent wrong | previous agent correct), pooled over
hops >= 2 of the clean+contaminated chains. Predicted P(wrong at hop k) = recursion from the seed state. If the measured curve
sits inside the prediction's CI, propagation is compositional (the two-agent result explains the chain).
"""
import json, os, random, sys
from collections import defaultdict

def rate(xs): return sum(xs) / len(xs) if xs else None
def boot(rows, f, reps=500, seed=0):
    """rows: list of (qid, value). f: aggregate over values. Cluster bootstrap over qid."""
    by = defaultdict(list)
    for q, v in rows: by[q].append(v)
    qs = list(by); rng = random.Random(seed); vals = []
    for _ in range(reps):
        sample = [v for q in rng.choices(qs, k=len(qs)) for v in by[q]]
        vals.append(f(sample))
    vals = [v for v in vals if v is not None]; vals.sort()
    return (f([v for q in qs for v in by[q]]), vals[int(.025 * len(vals))] if vals else None, vals[int(.975 * len(vals)) - 1] if vals else None)
def fmt(t): return "—" if t[0] is None else f"{t[0]:.3f} [{t[1]:.2f},{t[2]:.2f}]"
ab = lambda o: o in ("switch_alt", "switch_other")

def report(run):
    R = [json.loads(l) for l in open(os.path.join(run, "contagion.jsonl"))]
    meta = json.load(open(os.path.join(run, "meta.json"))) if os.path.exists(os.path.join(run, "meta.json")) else {}
    S = {"run": run, "model": meta.get("model"), "pairwise": {}, "chain": {}, "n_rows": len(R)}
    print("=" * 78); print(run, "|", meta.get("model"), "| rows", len(R))
    P = [r for r in R if r["cell"] == "pairwise" and r["outcome"] not in ("unparsed", "ambiguous")]
    unp = rate([r["outcome"] in ("unparsed", "ambiguous") for r in R if r["cell"] == "pairwise"])
    print(f"-- PAIRWISE (fold = abandon correct answer when Agent B names the distractor; accept = adopt B's true answer) unparsed={unp:.3f}")
    print(f"{'peer kind':14s} {'fold (truth=T)':26s} {'accept (truth=F)':26s} {'n':>5}")
    for kind in sorted({r["kind"] for r in P}, key=lambda k: (k.startswith("gen"), k)):
        fold = boot([(r["qid"], ab(r["outcome"])) for r in P if r["kind"] == kind and r["truth"]], rate)
        acc = boot([(r["qid"], r["outcome"] == "switch_alt") for r in P if r["kind"] == kind and not r["truth"]], rate)
        n = sum(1 for r in P if r["kind"] == kind)
        S["pairwise"][kind] = {"fold": fold, "accept": acc, "n": n}
        print(f"{kind:14s} {fmt(fold):26s} {fmt(acc):26s} {n:>5}")
    if "mention" in S["pairwise"]:
        m = S["pairwise"]["mention"]["fold"][0]
        gens = {k: v["fold"][0] for k, v in S["pairwise"].items() if k.startswith("gen:") and v["fold"][0] is not None}
        if gens:
            spread = max(gens.values()) - min(gens.values())
            print(f"   identity spread across generated peers: {spread:.3f} (P1 predicts <= 0.10); scripted mention = {m:.3f}")
            S["pairwise"]["_identity_spread"] = spread
        for k in ("reason", "conf_hi", "conf_lo"):
            if k in S["pairwise"] and S["pairwise"][k]["fold"][0] is not None:
                print(f"   {k} - mention = {S['pairwise'][k]['fold'][0] - m:+.3f}")
    # ---- CHAIN
    C = [r for r in R if r["cell"] == "chain"]
    if C:
        K = max(r["hop"] for r in C)
        print(f"-- CHAIN  P(agent at hop k answers WRONG)   k=1..{K}   (unparsed counted as wrong)")
        prev = {}  # (chain,qid,hop) -> correct
        for r in C: prev[(r["chain"], r["qid"], r["hop"])] = bool(r["correct"])
        trans = {"ww": [], "cw": []}
        for r in C:
            if r["hop"] >= 2 and r["chain"] != "firewall":
                pc = prev.get((r["chain"], r["qid"], r["hop"] - 1))
                if pc is None: continue
                trans["cw" if pc else "ww"].append((r["qid"], not r["correct"]))
        p_ww = boot(trans["ww"], rate) if trans["ww"] else (None, None, None); p_cw = boot(trans["cw"], rate) if trans["cw"] else (None, None, None)
        print(f"   hop transitions: P(wrong|prev wrong)={fmt(p_ww)}  P(wrong|prev correct)={fmt(p_cw)}")
        S["chain"]["p_ww"], S["chain"]["p_cw"] = p_ww, p_cw
        for cname in sorted({r["chain"] for r in C}):
            meas = []
            for k in range(1, K + 1):
                rows = [(r["qid"], not r["correct"]) for r in C if r["chain"] == cname and r["hop"] == k]
                meas.append(boot(rows, rate) if rows else (None, None, None))
            # Markov prediction from seed (contaminated/firewall seed = wrong; clean seed = correct)
            pred = []; pw = 0.0 if cname == "clean" else 1.0
            for k in range(1, K + 1):
                if p_ww[0] is None or p_cw[0] is None: pred.append(None); continue
                pw = pw * p_ww[0] + (1 - pw) * p_cw[0]; pred.append(pw)
            S["chain"][cname] = {"measured": meas, "markov": pred}
            print(f"   {cname:13s} measured: " + " ".join(f"k{k+1}={m[0]:.2f}" if m[0] is not None else f"k{k+1}=—" for k, m in enumerate(meas)))
            print(f"   {'':13s} markov:   " + " ".join(f"k{k+1}={p:.2f}" if p is not None else f"k{k+1}=—" for k, p in enumerate(pred)))
        if "firewall" in S["chain"] and "contaminated" in S["chain"]:
            fp = meta.get("firewall_pos", 2)
            after = [(S["chain"]["contaminated"]["measured"][k][0], S["chain"]["firewall"]["measured"][k][0]) for k in range(fp, K)]
            after = [(c, f) for c, f in after if c is not None and f is not None]
            if after:
                red = sum(c - f for c, f in after) / len(after)
                S["chain"]["firewall_reduction_downstream"] = red
                print(f"   firewall at hop {fp}: mean reduction in P(wrong) downstream = {red:+.3f} (P3 predicts >= 0.40)")
    json.dump(S, open(os.path.join(run, "summary.json"), "w"), indent=1, default=float)
    return S

if __name__ == "__main__":
    for run in sys.argv[1:]: report(run)
