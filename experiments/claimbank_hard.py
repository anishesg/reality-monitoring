#!/usr/bin/env python3
"""Hard claim bank: MMLU-Pro + TruthfulQA mc1 via datasets-server. Stdlib only, login node."""
import json, random, re, sys, time, urllib.request, urllib.parse

rng = random.Random(1)

def fetch(dataset, config, split, offset, length=100):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset, safe='')}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    req = urllib.request.Request(url, headers={"User-Agent": "claimbank/0.3"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("rows", [])
        except Exception as e:
            print(f"retry {dataset}@{offset}: {e}", file=sys.stderr); time.sleep(4)
    return []

def ok(s, maxlen=160):
    s = (s or "").strip()
    return 2 <= len(s) <= maxlen and not re.search(r"_{2,}", s)

rows = []
off = 0
while len(rows) < 600 and off < 3000:
    for item in fetch("TIGER-Lab/MMLU-Pro", "default", "test", off):
        r = item["row"]
        q, opts, ai = (r.get("question") or "").strip(), r.get("options") or [], r.get("answer_index", -1)
        if not (ok(q, 500) and isinstance(opts, list) and 0 <= ai < len(opts)): continue
        a = opts[ai].strip()
        wrong = [o.strip() for i, o in enumerate(opts) if i != ai and ok(o)]
        if not (ok(a) and wrong): continue
        d = rng.choice(wrong)
        if a.lower() == d.lower(): continue
        rows.append({"domain": "mmlupro", "question": q, "true_answer": a, "distractor": d})
        if len(rows) >= 600: break
    off += 100; time.sleep(1.5)
n1 = len(rows)
off = 0
while len(rows) < n1 + 300 and off < 900:
    for item in fetch("truthfulqa/truthful_qa", "multiple_choice", "validation", off):
        r = item["row"]
        q = (r.get("question") or "").strip()
        t = r.get("mc1_targets") or {}
        chs, labs = t.get("choices") or [], t.get("labels") or []
        corr = [c for c, l in zip(chs, labs) if l == 1]
        wrong = [c for c, l in zip(chs, labs) if l == 0]
        if not (ok(q, 300) and corr and wrong): continue
        a = corr[0].strip().rstrip(".")
        d = rng.choice(wrong).strip().rstrip(".")
        if not (ok(a) and ok(d)) or a.lower() == d.lower(): continue
        rows.append({"domain": "truthfulqa", "question": q, "true_answer": a, "distractor": d})
        if len(rows) >= n1 + 300: break
    off += 100; time.sleep(1.5)
for i, r in enumerate(rows): r["qid"] = i
with open("claims_hard.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
from collections import Counter
print("wrote", len(rows), Counter(r["domain"] for r in rows))
