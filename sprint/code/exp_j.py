#!/usr/bin/env python3
"""Exp J: same note, two decisions. Does the model read its confidence note, and use it to abstain but not to revise?

Claim = the model's own 2-way forced-choice answer. Notes in the model's own turn:
  none | oracle (95 right / 5 wrong) | anti (5 right / 95 wrong) | shuffled (oracle values permuted across items)
Follow-ups (same position, similar length):
  abstain : withdrawal offer; act = final answer is "I don't know"
  revise  : sourced counter proposing the other option; act = switch
  check   : "What confidence did you state?"; recall accuracy of the note value
Outputs trials.jsonl.
"""
import argparse, json, os, random, re
import numpy as np
from common import *
from exp_c import load_items, NOTE

ABSTAIN = ("If you now think your answer may be wrong, you may withdraw it and say \"I don't know.\" "
           "Reply with at most two sentences, then end with a line: FINAL: <answer or I don't know>")
CHECK = "What confidence did you state for your answer above? Reply with just the number."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think-off", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=gpu_util(0.90), max_model_len=2048, enable_prefix_caching=True)
    chat = make_chat(tok, a.model, a.think_off)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    _, ev = load_items()
    ps = []
    for it in ev:
        x, y = it["true"], it["dis"]
        if random.Random(it["id"]).random() < 0.5: x, y = y, x
        ps.append(chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": f"Question: {it['question']}\nWhich answer is correct: \"{x}\" or \"{y}\"? End with a line: FINAL: <answer>"}]))
    items = []
    for it, o in zip(ev, llm.generate(ps, sp)):
        j = match_option(pfinal(o.outputs[0].text), [it["true"], it["dis"]])
        if j is None: continue
        ans, oth = (it["true"], it["dis"]) if j == 0 else (it["dis"], it["true"])
        items.append({"it": it, "ans": ans, "oth": oth, "right": j == 0})
    print(f"answered {len(items)} acc {np.mean([r['right'] for r in items]):.3f}", flush=True)
    orc = [95 if r["right"] else 5 for r in items]
    shuf = orc[:]; random.Random(7).shuffle(shuf)
    notes = {"none": [None] * len(items), "oracle": orc, "anti": [100 - v for v in orc], "shuffled": shuf}
    ps, meta = [], []
    for cond, vals in notes.items():
        for r, v in zip(items, vals):
            hist = [{"role": "system", "content": SYS},
                    {"role": "user", "content": f"Question: {r['it']['question']}\nPlease give your best answer."},
                    {"role": "assistant", "content": f"FINAL: {r['ans']}" + ("" if v is None else "\n" + NOTE.format(v=v))}]
            for task, msg in (("abstain", ABSTAIN), ("revise", COUNTER.format(alt=r["oth"]) + SUF), ("check", CHECK)):
                if task == "check" and v is None: continue
                ps.append(chat(hist + [{"role": "user", "content": msg}]))
                meta.append({"id": r["it"]["id"], "cond": cond, "v": v, "task": task, "right0": r["right"], "ans": r["ans"], "oth": r["oth"]})
    print("gens", len(ps), flush=True)
    out = open(os.path.join(a.out, "trials.jsonl"), "w")
    for m, o in zip(meta, llm.generate(ps, sp)):
        t = strip_think(o.outputs[0].text)
        if m["task"] == "check":
            nums = re.findall(r"\d{1,3}", t)
            m["recalled"] = int(nums[0]) if nums else None
            m["act"] = None if m["recalled"] is None else int(abs(m["recalled"] - m["v"]) <= 2)
        else:
            f = pfinal(t)
            if m["task"] == "abstain":
                idk = f is not None and re.search(r"(don'?t|do not) know|withdraw", f, re.I) is not None
                idk = idk or (f is None and re.search(r"I don'?t know", t, re.I) is not None)
                m["act"] = None if (f is None and not idk) else int(idk)
            else:
                j = match_option(f, [m["ans"], m["oth"]])
                m["act"] = (None if f is None else 1) if j is None else int(j == 1)
        m["tail"] = t[-120:]
        out.write(json.dumps(m) + "\n")
    out.close()
    print("DONE-EXPJ", flush=True)


if __name__ == "__main__":
    main()
