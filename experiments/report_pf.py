import json, glob, os

def auroc(pairs):
    pairs = [(s, y) for s, y in pairs if s is not None and y is not None]
    pos = [s for s, y in pairs if y]; neg = [s for s, y in pairs if not y]
    if not pos or not neg: return None
    wins = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)

for d in sorted(glob.glob("results_ident/*")):
    tag = os.path.basename(d)
    if not os.path.exists(d + "/probes.jsonl"): continue
    sig = {r["qid"]: r for r in (json.loads(l) for l in open(d + "/signals.jsonl"))}
    pr = {r["qid"]: r for r in (json.loads(l) for l in open(d + "/probes.jsonl"))}
    print("=" * 10, tag, "=" * 10)
    # debiased margin AUROC vs fc_correct
    y = lambda q: sig.get(q, {}).get("fc_correct")
    a = auroc([(pr[q].get("margin_debiased"), y(q)) for q in pr])
    nm = sum(1 for q in pr if pr[q].get("margin_debiased") is not None)
    # generative verdict accuracy on true answers (should say True)
    vs = [pr[q].get("verdict_true_answer") for q in pr]
    vn = [v for v in vs if v is not None]
    acc = sum(vn) / len(vn) if vn else float("nan")
    print("debiased margin AUROC %s (n=%d) | verdict(True ans)=True rate %.3f (n=%d)" % (a, nm, acc, len(vn)))
    # weak-cue cells
    rows = [json.loads(l) for l in open(d + "/weak.jsonl")]
    def rate(cell):
        xs = [r for r in rows if r["cell"] == cell and r["outcome"] in ("retain", "switch_alt", "switch_other")]
        return (sum(r["outcome"] != "retain" for r in xs) / len(xs), len(xs)) if xs else (float("nan"), 0)
    tf, ntf = rate("TF"); ft, _ = rate("FT"); ff, _ = rate("FF")
    print("WEAK cue: TF %.3f FT %.3f FF %.3f  (FF-FT gap %+.3f, claim effect %+.3f)" % (tf, ft, ff, ff - ft, ff - tf))
    # belief split within weak FF
    hi, lo = [], []
    for r in rows:
        if r["cell"] != "FF" or r["outcome"] not in ("retain", "switch_alt", "switch_other"): continue
        s = sig.get(r["qid"], {})
        if s.get("b_d1") is None or s.get("b_d2") is None: continue
        (hi if s["b_d2"] - s["b_d1"] > 0 else lo).append(r["outcome"] != "retain")
    if hi and lo:
        print("WEAK FF belief split: more %.3f (n=%d) vs less %.3f (n=%d)  delta %+.3f"
              % (sum(hi)/len(hi), len(hi), sum(lo)/len(lo), len(lo), sum(hi)/len(hi)-sum(lo)/len(lo)))
    # confidence split within weak TF
    import statistics
    cf = [(sig[r["qid"]].get("conf"), r["outcome"] != "retain") for r in rows
          if r["cell"] == "TF" and r["outcome"] in ("retain","switch_alt","switch_other")
          and sig.get(r["qid"], {}).get("conf") is not None]
    if len(cf) > 40:
        med = statistics.median([c for c, _ in cf])
        h = [s for c, s in cf if c > med]; l = [s for c, s in cf if c <= med]
        if h and l:
            print("WEAK TF conf split (med %.0f): high %.3f vs low %.3f  delta %+.3f"
                  % (med, sum(h)/len(h), sum(l)/len(l), sum(h)/len(h)-sum(l)/len(l)))
    print()
