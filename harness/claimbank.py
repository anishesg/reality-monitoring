#!/usr/bin/env python3
"""Build claim bank from SciQ via HF datasets-server API. Stdlib only. Run on login node (tiny I/O)."""
import json, re, sys, time, urllib.request, urllib.parse

OUT = "claims.jsonl"
TARGET = 600
BASE = "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fsciq&config=default&split=train"

def ok_text(s, maxlen):
    if not s: return False
    s = s.strip()
    if not (2 <= len(s) <= maxlen): return False
    if re.search(r"_{2,}|\bfigure\b|\bdiagram\b", s.lower()): return False
    return True

def main():
    rows, offset = [], 0
    while len(rows) < TARGET and offset < 5000:
        url = f"{BASE}&offset={offset}&length=100"
        req = urllib.request.Request(url, headers={"User-Agent": "claimbank/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            print(f"fetch error at {offset}: {e}; retrying once", file=sys.stderr)
            time.sleep(5)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        batch = data.get("rows", [])
        if not batch: break
        for item in batch:
            row = item["row"]
            q = (row.get("question") or "").strip()
            ans = (row.get("correct_answer") or "").strip()
            dis = (row.get("distractor1") or "").strip()
            if dis.lower() == ans.lower():
                dis = (row.get("distractor2") or "").strip()
            if not (ok_text(q, 300) and ok_text(ans, 60) and ok_text(dis, 60)): continue
            if ans.lower() == dis.lower(): continue
            rows.append({"qid": len(rows), "question": q, "true_answer": ans, "distractor": dis})
        offset += 100
        time.sleep(1)
    with open(OUT, "w") as f:
        for r in rows[:TARGET]:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {min(len(rows), TARGET)} claims to {OUT}")

if __name__ == "__main__":
    main()
