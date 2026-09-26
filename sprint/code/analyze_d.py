import glob, json
keys = {"mmlu": "acc,none", "truthfulqa_mc2": "acc,none", "gsm8k": "exact_match,flexible-extract", "ifeval": "prompt_level_strict_acc,none"}
res = {}
for f in glob.glob("res_d/*/*/results_*.json"):
    m = f.split("/")[1].split("_")[0]
    R = json.load(open(f))["results"]
    for t, k in keys.items():
        if t in R and k in R[t]:
            res.setdefault(m, {})[t] = (R[t][k], R[t].get(k.replace(",", "_stderr,"), None))
print("model  " + "  ".join(f"{t:>16}" for t in keys))
for m in ["base", "conf", "ctrl", "eown", "eshuf"] + sorted(k for k in res if k not in ("base", "conf", "ctrl", "eown", "eshuf")):
    print(f"{m:6s} " + "  ".join(f"{res.get(m,{}).get(t,(float('nan'),0))[0]:16.4f}" for t in keys))
for a, b in (("conf", "base"), ("conf", "ctrl")):
    print(f"{a}-{b} " + "  ".join(f"{100*(res.get(a,{}).get(t,(float('nan'),))[0]-res.get(b,{}).get(t,(float('nan'),))[0]):+16.2f}" for t in keys))
json.dump(res, open("res_d/summary_d.json", "w"), indent=1)
