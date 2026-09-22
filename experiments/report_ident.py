import json, glob, math, os, statistics

def auroc(pairs):
    pairs = [(s, y) for s, y in pairs if s is not None and y is not None]
    pos = [s for s, y in pairs if y]; neg = [s for s, y in pairs if not y]
    if not pos or not neg: return None
    wins = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)

for d in sorted(glob.glob("results_ident/*")):
    tag = os.path.basename(d)
    if not os.path.exists(d + "/ident.jsonl"): continue
    rows = [json.loads(l) for l in open(d + "/ident.jsonl")]
    sig = {r["qid"]: r for r in (json.loads(l) for l in open(d + "/signals.jsonl"))}
    print("=" * 14, tag, "=" * 14)
    def rate(sel):
        xs = [r for r in rows if sel(r) and r["outcome"] in ("retain", "switch_alt", "switch_other")]
        if not xs: return float("nan"), 0
        return sum(r["outcome"] != "retain" for r in xs) / len(xs), len(xs)
    line = []
    for cell in ("TF", "FT", "FF"):
        c, cn = rate(lambda r, x=cell: r["cell"] == x and r["chal"] == "counter")
        p, pn = rate(lambda r, x=cell: r["cell"] == x and r["chal"] == "pressure")
        line.append("%s ctr %.3f prs %.3f" % (cell, c, p))
    print(" | ".join(line))
    # belief-margin sensitivity within FF (counter)
    hi, lo = [], []
    for r in rows:
        if r["cell"] != "FF" or r["chal"] != "counter": continue
        if r["outcome"] not in ("retain", "switch_alt", "switch_other"): continue
        s = sig.get(r["qid"], {})
        if s.get("b_d1") is None or s.get("b_d2") is None: continue
        (hi if s["b_d2"] - s["b_d1"] > 0 else lo).append(r["outcome"] != "retain")
    if hi and lo:
        print("FF belief split: alt-more-believed %.3f (n=%d) vs less %.3f (n=%d)  delta %.3f"
              % (sum(hi)/len(hi), len(hi), sum(lo)/len(lo), len(lo), sum(hi)/len(hi) - sum(lo)/len(lo)))
    # own-confidence effect on TF counter
    cf = [(sig[r["qid"]].get("conf"), r["outcome"] != "retain") for r in rows
          if r["cell"] == "TF" and r["chal"] == "counter"
          and r["outcome"] in ("retain", "switch_alt", "switch_other")
          and sig.get(r["qid"], {}).get("conf") is not None]
    if len(cf) > 40:
        med = statistics.median([c for c, _ in cf])
        h = [s for c, s in cf if c > med]; l = [s for c, s in cf if c <= med]
        if h and l:
            print("TF conf split (med %.0f): high %.3f vs low %.3f  delta %.3f"
                  % (med, sum(h)/len(h), sum(l)/len(l), sum(h)/len(h) - sum(l)/len(l)))
    # signal AUROCs
    y = lambda r: r.get("fc_correct")
    srows = list(sig.values())
    parts = []
    for name, key in [("conf", "conf"), ("pTrue", "p_true"), ("margin", "margin_true_minus_d1"), ("consist", "consistency")]:
        parts.append("%s %s" % (name, auroc([(r.get(key), y(r)) for r in srows])))
    mt = [r for r in srows if r.get("b_true") is not None and r.get("b_d1") is not None]
    parts.append("bMargin %s" % auroc([(r["b_true"] - r["b_d1"], y(r)) for r in mt]))
    acc = [r["fc_correct"] for r in srows if r.get("fc_correct") is not None]
    parts.append("FCacc %.3f (n=%d)" % (sum(acc)/len(acc), len(acc)))
    print("AUROC: " + " | ".join(parts))
    # template spread
    ts = []
    for t in range(5):
        p, _ = rate(lambda r, t=t: r["tmpl"] == t and r["chal"] == "counter")
        ts.append(round(p, 3))
    print("template spread:", ts)
    print()
