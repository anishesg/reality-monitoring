#!/usr/bin/env python3
"""Login-node prep for D2: (1) real instruction replay bank from tulu-3-sft-mixture,
(2) external capability bank from MMLU. Stdlib only (datasets-server API)."""
import json, random, sys, time, urllib.request, urllib.parse

rng = random.Random(11)

def fetch(dataset, config, split, offset, length=100):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset, safe='')}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    req = urllib.request.Request(url, headers={"User-Agent": "d2prep/0.1"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("rows", [])
        except Exception as e:
            print(f"retry {dataset}@{offset}: {e}", file=sys.stderr); time.sleep(4)
    return []

# ---- replay bank: single-turn tulu examples, modest length ----
replay, off = [], 0
while len(replay) < 4000 and off < 40000:
    for item in fetch("allenai/tulu-3-sft-mixture", "default", "train", off):
        m = item["row"].get("messages") or []
        if len(m) == 2 and m[0].get("role") == "user" and m[1].get("role") == "assistant":
            u, a = m[0]["content"].strip(), m[1]["content"].strip()
            if 10 < len(u) < 1200 and 5 < len(a) < 1200:
                replay.append({"user": u, "assistant": a})
        if len(replay) >= 4000: break
    off += 100
    if off % 2000 == 0: print(f"replay {len(replay)} @ {off}", file=sys.stderr)
    time.sleep(0.8)
with open("replay_tulu.jsonl", "w") as f:
    for r in replay: f.write(json.dumps(r) + "\n")
print("replay:", len(replay))

# ---- capability bank: MMLU test, 300 FC items ----
cap, off = [], 0
while len(cap) < 300 and off < 3000:
    for item in fetch("cais/mmlu", "all", "test", off):
        r = item["row"]
        q, ch, ai = (r.get("question") or "").strip(), r.get("choices") or [], r.get("answer", -1)
        if not (10 < len(q) < 400 and isinstance(ch, list) and 0 <= ai < len(ch)): continue
        a = ch[ai].strip()
        wrong = [c.strip() for i, c in enumerate(ch) if i != ai and 1 < len(c.strip()) < 120]
        if not (1 < len(a) < 120 and wrong): continue
        cap.append({"qid": len(cap), "question": q, "true_answer": a, "distractor": rng.choice(wrong)})
        if len(cap) >= 300: break
    off += 100; time.sleep(0.8)
with open("mmlu_cap.jsonl", "w") as f:
    for r in cap: f.write(json.dumps(r) + "\n")
print("cap:", len(cap))
