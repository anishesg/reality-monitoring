#!/usr/bin/env python3
"""Build multi-domain claim bank. RUN ON LOGIN NODE (needs internet).
Domains: sciq (science QA), mmlu (mixed academic), truthfulqa (adversarial misconceptions).
Output: claims.jsonl with {qid, domain, question, true_answer, distractor}
"""
import json, random, re

rng = random.Random(0)

def ok(s, maxlen=80):
    s = (s or "").strip()
    return 2 <= len(s) <= maxlen and not re.search(r"_{2,}", s)

def from_sciq(n=600):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    out = []
    for r in ds:
        if len(out) >= n: break
        q, a, d = r["question"].strip(), r["correct_answer"].strip(), r["distractor1"].strip()
        if d.lower() == a.lower(): d = r["distractor2"].strip()
        if ok(q, 300) and ok(a) and ok(d) and a.lower() != d.lower():
            out.append({"domain": "sciq", "question": q, "true_answer": a, "distractor": d})
    return out

def from_mmlu(n=500):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="validation")
    out = []
    for r in ds:
        if len(out) >= n: break
        q = r["question"].strip()
        ch = [c.strip() for c in r["choices"]]
        a = ch[r["answer"]]
        wrong = [c for i, c in enumerate(ch) if i != r["answer"] and ok(c)]
        if not wrong: continue
        d = rng.choice(wrong)
        if ok(q, 400) and ok(a) and a.lower() != d.lower():
            out.append({"domain": "mmlu", "question": q, "true_answer": a, "distractor": d})
    return out

def from_truthfulqa(n=400):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for r in ds:
        if len(out) >= n: break
        q = r["question"].strip()
        t = r["mc1_targets"]
        correct = [c for c, l in zip(t["choices"], t["labels"]) if l == 1]
        wrong = [c for c, l in zip(t["choices"], t["labels"]) if l == 0]
        if not correct or not wrong: continue
        a, d = correct[0].strip().rstrip("."), rng.choice(wrong).strip().rstrip(".")
        if ok(q, 300) and ok(a, 120) and ok(d, 120) and a.lower() != d.lower():
            out.append({"domain": "truthfulqa", "question": q, "true_answer": a, "distractor": d})
    return out

def main():
    rows = from_sciq() + from_mmlu() + from_truthfulqa()
    for i, r in enumerate(rows):
        r["qid"] = i
    with open("claims.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print("wrote", len(rows), Counter(r["domain"] for r in rows))

if __name__ == "__main__":
    main()
