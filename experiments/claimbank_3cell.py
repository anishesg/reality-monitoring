#!/usr/bin/env python3
"""MMLU-Pro bank with TWO distractors per item, for the 3-cell identification design.
Run on login node (datasets-server API). Output: claims3.jsonl"""
import json, random, re, sys, time, urllib.request, urllib.parse

rng = random.Random(7)

def fetch(offset, length=100):
    url = ("https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMMLU-Pro"
           f"&config=default&split=test&offset={offset}&length={length}")
    req = urllib.request.Request(url, headers={"User-Agent": "cb3/0.1"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("rows", [])
        except Exception as e:
            print(f"retry@{offset}: {e}", file=sys.stderr); time.sleep(4)
    return []

def ok(s, maxlen=160):
    s = (s or "").strip()
    return 2 <= len(s) <= maxlen and not re.search(r"_{2,}", s)

rows, off = [], 1600   # offset past the earlier bank to reduce overlap
while len(rows) < 600 and off < 6000:
    for item in fetch(off):
        r = item["row"]
        q, opts, ai = (r.get("question") or "").strip(), r.get("options") or [], r.get("answer_index", -1)
        if not (ok(q, 500) and isinstance(opts, list) and 0 <= ai < len(opts)): continue
        a = opts[ai].strip()
        wrong = [o.strip() for i, o in enumerate(opts) if i != ai and ok(o) and o.strip().lower() != a.lower()]
        if not (ok(a) and len(wrong) >= 2): continue
        d1, d2 = rng.sample(wrong, 2)
        rows.append({"qid": len(rows), "question": q, "true_answer": a, "d1": d1, "d2": d2})
        if len(rows) >= 600: break
    off += 100; time.sleep(1.2)
with open("claims3.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("wrote", len(rows))
