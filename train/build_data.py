#!/usr/bin/env python3
"""Build training data for the causal post-training ladder.

Modes
  injected : v17-style dialogues with an injected assistant claim (self origin, 4 challenge kinds;
             user_recency origin, 2 kinds) x conf {low,high} x truth {T,F}. Chosen = correct final
             action (retain if the claim is true, switch to the alternative if it is false).
  elicited : first sample the model's OWN forced-choice answer (+ verbalized confidence, v16 cell A),
             then challenge that actual claim (v16 cell D). Chosen = keep if it was right, else the
             other option. Needs --model.
  generic  : subsample N pairs from allenai/llama-3.1-tulu-3-8b-preference-mixture (arm A1 control).

Output JSONL rows: {"prompt": [messages], "chosen": str, "rejected": str, "meta": {...}}
The same rows serve GRPO (prompt + meta.claim/alt/truth are the verifiable labels).
Training bank is disjoint from the v17 eval set (hard-bank qids < 450 are never used).
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (KINDS, USER_KINDS, train_claims, self_dialogue, user_recency_dialogue,
                    elicited_dialogue, write_jsonl, parse_final, outcome, norm, Generator)

RETAIN_TXT = ["I've reconsidered and my original answer still stands; the alternative does not fit the question.",
              "After thinking it over, I stand by my answer, since it is the one supported by the facts.",
              "I have re-checked this and see no reason to change: my previous answer is correct."]
SWITCH_TXT = ["On reflection I was mistaken; the correct answer is the other one.",
              "Having reconsidered, my earlier answer was wrong and I am revising it.",
              "You are right to push back; the correct answer is different from what I gave."]

def resp(kind_txt, final, rng):
    return f"{rng.choice(kind_txt)} FINAL: {final}"

def build_injected(recs, rng, conf_lvls=("low", "high")):
    rows = []
    for r in recs:
        for truth in (True, False):
            claim = r["true_answer"] if truth else r["distractor"]
            alt = r["distractor"] if truth else r["true_answer"]
            correct = claim if truth else alt
            wrong = alt if truth else claim
            for conf in conf_lvls:
                cells = [("self", k, self_dialogue(r["question"], claim, conf, k, alt)) for k in KINDS]
                cells += [("user_recency", k, user_recency_dialogue(r["question"], claim, conf, k, alt)) for k in USER_KINDS]
                for origin, kind, msgs in cells:
                    chosen = resp(RETAIN_TXT if truth else SWITCH_TXT, correct, rng)
                    rejected = resp(SWITCH_TXT if truth else RETAIN_TXT, wrong, rng)
                    rows.append({"prompt": msgs, "chosen": chosen, "rejected": rejected,
                                 "meta": {"qid": r["qid"], "bank": r["bank"], "domain": r.get("domain"),
                                          "origin": origin, "kind": kind, "conf": conf, "truth": truth,
                                          "claim": claim, "alt": alt, "correct": correct}})
    return rows

def fc_prompt(r, rng):
    a, b = r["true_answer"], r["distractor"]
    if random.Random(r["qid"]).random() < 0.5: a, b = b, a
    return (f"Question: {r['question']}\nWhich answer is correct: \"{a}\" or \"{b}\"?\n"
            f"Give your choice AND your confidence that it is correct as a percentage (50-100)."
            f" End your reply with two lines:\nFINAL: <answer>\nCONFIDENCE: <number>%")

def parse_conf(text):
    import re
    h = re.findall(r"CONFIDENCE\s*[:\-]\s*(\d{1,3})", text, flags=re.IGNORECASE)
    if h: return max(0, min(100, int(h[-1])))
    h = re.findall(r"(\d{1,3})\s*%", text)
    return max(0, min(100, int(h[-1]))) if h else None

def build_elicited(recs, model, rng, kinds=KINDS):
    gen = Generator(model)
    prompts = [fc_prompt(r, rng) for r in recs]
    msgs = [[{"role": "system", "content": __import__("common").SYS}, {"role": "user", "content": p}] for p in prompts]
    texts = gen.chat(msgs, max_tokens=96)
    rows, n_unres = [], 0
    for r, p, t in zip(recs, prompts, texts):
        ans, conf = parse_final(t), parse_conf(t)
        if ans is None: n_unres += 1; continue
        res = outcome(ans, r["true_answer"], r["distractor"])
        if res not in ("retain", "switch_alt"): n_unres += 1; continue
        truth = res == "retain"
        alt = r["distractor"] if truth else r["true_answer"]
        correct = ans if truth else alt
        for kind in kinds:
            m = elicited_dialogue(r["question"], p, ans, conf, kind, alt)
            rows.append({"prompt": m,
                         "chosen": resp(RETAIN_TXT if truth else SWITCH_TXT, correct, rng),
                         "rejected": resp(SWITCH_TXT if truth else RETAIN_TXT, alt if truth else ans, rng),
                         "meta": {"qid": r["qid"], "bank": r["bank"], "domain": r.get("domain"), "origin": "elicited",
                                  "kind": kind, "conf": conf, "truth": truth, "claim": ans, "alt": alt, "correct": correct}})
    print(f"[elicited] backend={gen.backend} resolved={len(recs) - n_unres}/{len(recs)} rows={len(rows)}", file=sys.stderr)
    return rows

def build_generic(n, rng, name="allenai/llama-3.1-tulu-3-8b-preference-mixture"):
    from datasets import load_dataset
    ds = load_dataset(name, split="train", streaming=True).shuffle(seed=rng.randint(0, 10**6), buffer_size=5000)
    rows = []
    for ex in ds:
        ch, rj = ex.get("chosen"), ex.get("rejected")
        if not (isinstance(ch, list) and isinstance(rj, list) and ch and rj): continue
        if ch[-1].get("role") != "assistant" or rj[-1].get("role") != "assistant": continue
        prompt = ch[:-1]
        if [m["content"] for m in prompt] != [m["content"] for m in rj[:-1]]: continue
        if sum(len(m["content"]) for m in prompt) + len(ch[-1]["content"]) + len(rj[-1]["content"]) > 6000: continue
        rows.append({"prompt": prompt, "chosen": ch[-1]["content"], "rejected": rj[-1]["content"],
                     "meta": {"source": name, "origin": "generic"}})
        if len(rows) >= n: break
    return rows

def balance(rows, per_cell, rng):
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        m = r["meta"]; by[(m.get("origin"), m.get("kind"), m.get("truth"))].append(r)
    out = []
    for k in sorted(by, key=str):
        rng.shuffle(by[k]); out += by[k][:per_cell]
    rng.shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("injected", "elicited", "generic"), default="injected")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="required for --mode elicited")
    ap.add_argument("--n-questions", type=int, default=0, help="0 = all training questions")
    ap.add_argument("--no-sciq", action="store_true")
    ap.add_argument("--per-cell", type=int, default=0, help="cap rows per origin x kind x truth cell (0 = no cap)")
    ap.add_argument("--n-generic", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.mode == "generic":
        rows = build_generic(args.n_generic, rng)
    else:
        recs = train_claims(include_sciq=not args.no_sciq)
        rng.shuffle(recs)
        if args.n_questions: recs = recs[: args.n_questions]
        assert all(not (r["bank"] == "hard" and r["qid"] < 450) for r in recs), "eval leakage"
        if args.mode == "injected": rows = build_injected(recs, rng)
        else:
            assert args.model, "--model required"
            rows = build_elicited(recs, args.model, rng)
        if args.per_cell: rows = balance(rows, args.per_cell, rng)
    write_jsonl(args.out, rows)
    from collections import Counter
    c = Counter((r["meta"].get("origin"), r["meta"].get("kind"), r["meta"].get("truth")) for r in rows)
    print(json.dumps({"mode": args.mode, "rows": len(rows), "cells": len(c), "out": args.out}, indent=1))

if __name__ == "__main__":
    main()
