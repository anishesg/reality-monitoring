import json, os, glob, collections
print("%-26s %10s %10s %6s %6s" % ("arm", "lo->switch", "hi->retain", "rule", "MMLU"))
rows = []
for d in sorted(glob.glob("results_r3/*")):
    tag = os.path.basename(d)
    try: ev = [json.loads(l) for l in open(d + "/eval.jsonl")]
    except Exception: continue
    ok = [r for r in ev if r["outcome"] in ("retain", "switch_alt", "switch_other")]
    lo = [r["outcome"] != "retain" for r in ok if r.get("p_eff") is not None and r["p_eff"] <= 40]
    hi = [r["outcome"] == "retain" for r in ok if r.get("p_eff") is not None and r["p_eff"] >= 80]
    los = sum(lo) / len(lo) if lo else float("nan")
    his = sum(hi) / len(hi) if hi else float("nan")
    rule = (los + his) / 2
    capf = d + "/mmlu_cap.json"
    cap = json.load(open(capf))["mmlu_fc_acc"] if os.path.exists(capf) else float("nan")
    rows.append((tag, los, his, rule, cap))
    print("%-26s %10.3f %10.3f %6.3f %6.3f" % (tag, los, his, rule, cap))
agg = collections.defaultdict(list)
for tag, los, his, rule, cap in rows:
    agg[tag.rsplit("_s", 1)[0]].append((rule, cap))
print()
print("%-26s %6s %6s" % ("config (mean over seeds)", "rule", "MMLU"))
for cfg, v in sorted(agg.items()):
    r = sum(x[0] for x in v) / len(v); c = sum(x[1] for x in v) / len(v)
    print("%-26s %6.3f %6.3f  (n=%d)" % (cfg, r, c, len(v)))
