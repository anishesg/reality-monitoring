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
    jp = os.path.join(run, "judged.jsonl")  # LLM equivalence re-grade of switch_other/ambiguous rows (analysis/judge_equiv.py); frontier runs only
    if os.path.exists(jp) and os.environ.get("RM_GRADING", "judged") == "judged":
        J = {json.dumps(j["key"]): j["outcome_judged"] for j in (json.loads(l) for l in open(jp))}; nj = 0
        for r in R:
            k = json.dumps([r["cell"], r["kind"], r["qid"], r["truth"], r["hop"], r["chain"]])
            if k in J: r["outcome_string"] = r["outcome"]; r["outcome"] = J[k]; r["correct"] = J[k] == "retain"; nj += 1
        print(f"[judged grading: {nj} rows overlaid from judged.jsonl]")
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
    # generated peers: how often the peer's text literally named the alternative, and fold restricted to those trials
    pm = {}
    for q in (os.path.join(run, "peer_msgs.jsonl"), os.path.join(os.path.dirname(run.rstrip("/")), "_peers", "peer_msgs.jsonl")):
        for r in ([json.loads(l) for l in open(q)] if os.path.exists(q) else []): pm[(r["peer"], r["qid"], r["truth"])] = r["mentions_alt"]
    for kind in [k for k in S["pairwise"] if k.startswith("gen:")]:
        name = kind[4:]; rows = [r for r in P if r["kind"] == kind and pm.get((name, r["qid"], r["truth"])) is not None]
        if not rows: continue
        named = [r for r in rows if pm[(name, r["qid"], r["truth"])]]
        fold_named = boot([(r["qid"], ab(r["outcome"])) for r in named if r["truth"]], rate) if any(r["truth"] for r in named) else (None, None, None)
        S["pairwise"][kind]["mentions_alt_rate"] = len(named) / len(rows); S["pairwise"][kind]["fold_when_named"] = fold_named
        print(f"   {kind}: peer text named the alternative in {len(named)/len(rows):.2f} of trials; fold when named = {fmt(fold_named)}")
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
    # ---- GENUINE (model's own answer; fold = items it got right, accept = items it got wrong)
    G = [r for r in R if r["cell"] == "genuine" and r["kind"] != "_own" and r["outcome"] not in ("unparsed", "ambiguous", "error")]
    own = [r for r in R if r["cell"] == "genuine" and r["kind"] == "_own"]
    if own:
        acc = rate([r["truth"] is True for r in own]); excl = rate([r["truth"] is None for r in own])
        print(f"-- GENUINE (own generated answer; accuracy={acc:.3f}, excluded neither-candidate={excl:.3f})")
        print(f"{'peer kind':14s} {'fold (own right)':26s} {'accept (own wrong)':26s} {'n':>5}")
        S["genuine"] = {"own_accuracy": acc, "excluded": excl}
        for kind in sorted({r["kind"] for r in G}, key=lambda k: (k.startswith("gen"), k)):
            fold = boot([(r["qid"], ab(r["outcome"])) for r in G if r["kind"] == kind and r["truth"]], rate)
            accp = boot([(r["qid"], r["outcome"] == "switch_alt") for r in G if r["kind"] == kind and not r["truth"]], rate)
            S["genuine"][kind] = {"fold": fold, "accept": accp, "n": sum(1 for r in G if r["kind"] == kind)}
            print(f"{kind:14s} {fmt(fold):26s} {fmt(accp):26s} {S['genuine'][kind]['n']:>5}")
    # ---- CHAIN
    C = [r for r in R if r["cell"] == "chain"]
    if C:
        K = max(r["hop"] for r in C)
        cunp = rate([r["outcome"] == "unparsed" for r in C])
        print(f"-- CHAIN  P(agent at hop k answers WRONG)   k=1..{K}   (unparsed counted as wrong; chain unparsed rate = {cunp:.3f})")
        S["chain"]["unparsed_rate"] = cunp
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
        if p_ww[0] is not None and p_cw[0] is not None:
            lam = p_ww[0] - p_cw[0]; pi_inf = p_cw[0] / (p_cw[0] + 1 - p_ww[0]) if (p_cw[0] + 1 - p_ww[0]) > 0 else None
            half = (-0.6931 / __import__("math").log(lam)) if 0 < lam < 1 else None
            S["chain"]["lambda"], S["chain"]["pi_inf"], S["chain"]["halflife_hops"] = lam, pi_inf, half
            print(f"   closed form: pi_k = pi_inf + (pi_1 - pi_inf) * lambda^(k-1);  lambda = p_ww - p_cw = {lam:.3f};  pi_inf = p_cw/(p_cw+1-p_ww) = {pi_inf if pi_inf is None else round(pi_inf,3)};  half-life = {half if half is None else round(half,1)} hops")
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
            # post-hoc two-phase prediction: start from the MEASURED hop-1 state (the seed is a bare message, not a full agent
            # reply), then apply the hop>=2 transition rates. Reported next to the pre-registered version, never instead of it.
            pred2 = []; pw2 = meas[0][0] if meas and meas[0][0] is not None else None
            for k in range(1, K + 1):
                if pw2 is None or p_ww[0] is None or p_cw[0] is None: pred2.append(None); continue
                if k > 1: pw2 = pw2 * p_ww[0] + (1 - pw2) * p_cw[0]
                pred2.append(pw2)
            inside = sum(1 for m, p in zip(meas, pred) if m[0] is not None and p is not None and m[1] is not None and m[1] <= p <= m[2])
            inside2 = sum(1 for m, p in zip(meas, pred2) if m[0] is not None and p is not None and m[1] is not None and m[1] <= p <= m[2])
            S["chain"][cname] = {"measured": meas, "markov": pred, "markov_from_k1": pred2, "hops_inside_ci": inside, "hops_inside_ci_from_k1": inside2}
            print(f"   {cname:13s} measured: " + " ".join(f"k{k+1}={m[0]:.2f}" if m[0] is not None else f"k{k+1}=—" for k, m in enumerate(meas)))
            print(f"   {'':13s} markov:   " + " ".join(f"k{k+1}={p:.2f}" if p is not None else f"k{k+1}=—" for k, p in enumerate(pred)) + f"   (prereg P3: {inside}/{K} hops inside measured CI)")
            print(f"   {'':13s} from k1:  " + " ".join(f"k{k+1}={p:.2f}" if p is not None else f"k{k+1}=—" for k, p in enumerate(pred2)) + f"   (post hoc: {inside2}/{K} inside)")
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

# ------------------------------------------------------------------------------------------------- figures + optional W&B
def figure(S, run):
    try: import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception: return None
    pw = {k: v for k, v in S["pairwise"].items() if not k.startswith("_") and v.get("fold", (None,))[0] is not None}
    if not pw: return None
    chain_ok = any(isinstance(v, dict) and "measured" in v for v in S["chain"].values())
    fig, axes = plt.subplots(1, 2 if chain_ok else 1, figsize=(11 if chain_ok else 6, 4))
    ax = axes[0] if chain_ok else axes
    order = ["none", "mention", "reason", "conf_lo", "conf_hi"] + sorted(k for k in pw if k.startswith("gen:"))
    ks = [k for k in order if k in pw]; x = range(len(ks))
    f = [pw[k]["fold"][0] for k in ks]; lo = [pw[k]["fold"][0] - (pw[k]["fold"][1] or 0) for k in ks]; hi = [(pw[k]["fold"][2] or 0) - pw[k]["fold"][0] for k in ks]
    a = [pw[k]["accept"][0] or 0 for k in ks]
    ax.bar([i - 0.2 for i in x], f, 0.4, yerr=[lo, hi], color="#b91c1c", capsize=3, label="fold (abandon correct answer)")
    ax.bar([i + 0.2 for i in x], a, 0.4, color="#15803d", label="accept (adopt true alternative)")
    ax.set_xticks(list(x)); ax.set_xticklabels([k.replace("gen:", "peer:") for k in ks], rotation=30, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("rate"); ax.set_title(f"Pairwise: message from Agent B  ({os.path.basename(run)})"); ax.legend(fontsize=8, loc="lower right")
    if chain_ok:
        ax2 = axes[1]
        for cname, col in (("contaminated", "#b91c1c"), ("clean", "#15803d"), ("firewall", "#1d4ed8")):
            v = S["chain"].get(cname)
            if not isinstance(v, dict) or "measured" not in v: continue
            ks_ = [i + 1 for i, m in enumerate(v["measured"]) if m[0] is not None]
            ax2.plot(ks_, [m[0] for m in v["measured"] if m[0] is not None], "o-", color=col, label=f"{cname} (measured)")
            ax2.fill_between(ks_, [m[1] for m in v["measured"] if m[0] is not None], [m[2] for m in v["measured"] if m[0] is not None], color=col, alpha=.15)
            if any(p is not None for p in v["markov"]): ax2.plot(ks_, [p for p in v["markov"] if p is not None], "--", color=col, alpha=.7, label=f"{cname} (Markov prediction)")
        ax2.set_xlabel("agent position k in chain"); ax2.set_ylabel("P(agent k answers wrong)"); ax2.set_ylim(0, 1); ax2.set_title("Chain propagation"); ax2.legend(fontsize=7)
    fig.tight_layout(); out = os.path.join(run, "contagion.png"); fig.savefig(out, dpi=150); plt.close(fig); return out

def wandb_log(S, run, png):
    if not os.environ.get("WANDB_API_KEY"): return
    try:
        import wandb
        r = wandb.init(project=os.environ.get("WANDB_PROJECT", "reality-monitoring"), name=f"contagion-{os.path.basename(run)}", reinit=True, config={"model": S.get("model")})
        flat = {f"pairwise/{k}/fold": v["fold"][0] for k, v in S["pairwise"].items() if isinstance(v, dict) and v.get("fold", (None,))[0] is not None}
        flat.update({f"pairwise/{k}/accept": v["accept"][0] for k, v in S["pairwise"].items() if isinstance(v, dict) and v.get("accept", (None,))[0] is not None})
        for cname, v in S["chain"].items():
            if isinstance(v, dict) and "measured" in v:
                for i, m in enumerate(v["measured"]):
                    if m[0] is not None: flat[f"chain/{cname}/k{i+1}"] = m[0]
        if png: flat["figure"] = wandb.Image(png)
        r.log(flat); r.finish()
    except Exception as e: print("wandb skipped:", str(e)[:80])

if __name__ == "__main__" and len(sys.argv) > 1:
    for run in sys.argv[1:]:
        S = json.load(open(os.path.join(run, "summary.json"))); png = figure(S, run); wandb_log(S, run, png)
        if png: print("figure:", png)
