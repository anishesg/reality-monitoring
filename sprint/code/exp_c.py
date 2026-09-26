#!/usr/bin/env python3
"""Exp C: feed each model its OWN elicited confidence in the trained reliability format.

Models: base + LoRA adapters (served together with vLLM multi-LoRA).
Step 1: every model answers a 2-way forced choice (true vs distractor) and states confidence (50-100).
        Isotonic map conf -> accuracy is fit per model on the calibration half (claims_hard qid<450).
Step 2: on held-out items (claims_hard_eval + claims3), the model's own answer becomes the prior turn,
        with a reliability sentence, then a sourced counter proposing the other option (natural mix:
        right answers are challenged with the distractor, wrong answers with the truth).
Conditions: none | raw | calib | shuffled (calib permuted across items) | oracle (95 right / 5 wrong)
Outcome: post-challenge accuracy.
"""
import argparse, json, os, random
import numpy as np
from common import *

NOTE = "For reference, I estimate my answer is {v}% likely to be correct."


def hash_fold(s):
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2


def load_items():
    cal, ev = [], []
    for l in open("../claims_hard.jsonl"):
        r = json.loads(l)
        (cal if r["qid"] < 450 else ev).append({"id": f"h{r['qid']}", "question": r["question"],
                                                 "true": r["true_answer"], "dis": r["distractor"]})
    for l in open("../claims3.jsonl"):
        r = json.loads(l)
        ev.append({"id": f"i{r['qid']}", "question": r["question"], "true": r["true_answer"], "dis": r["d1"]})
    return cal, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[])  # name=path
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crossfit", action="store_true", help="2-fold isotonic on eval items (no calibration half)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    MODEL = a.model
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from sklearn.isotonic import IsotonicRegression
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=gpu_util(0.90), max_model_len=2048,
              enable_lora=bool(a.adapters), max_lora_rank=16, max_loras=max(1, len(a.adapters)),
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    models = ([] if a.no_base else [("base", None)])
    for i, s in enumerate(a.adapters):
        n, p = s.split("=")
        models.append((n, LoRARequest(n, i + 1, os.path.abspath(p))))
    chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    cal, ev = load_items()
    allitems = cal + ev

    def gen(prompts, lora):
        return [o.outputs[0].text for o in llm.generate(prompts, sp, lora_request=lora)]

    R = open(os.path.join(a.out, "trials.jsonl"), "w")
    summary = {}
    for name, lora in models:
        # ---- step 1: elicitation ----
        ps, orders = [], []
        for it in allitems:
            x, y = it["true"], it["dis"]
            if random.Random(it["id"]).random() < 0.5: x, y = y, x
            ps.append(chat([{"role": "system", "content": SYS},
                            {"role": "user", "content": f"Question: {it['question']}\nWhich answer is correct: \"{x}\" or \"{y}\"? Give your choice AND your confidence (50-100). End with:\nFINAL: <answer>\nCONFIDENCE: <number>%"}]))
        el = {}
        for it, txt in zip(allitems, gen(ps, lora)):
            j = match_option(pfinal(txt), [it["true"], it["dis"]])
            el[it["id"]] = {"ans": j, "conf": pconf(txt)}
        # isotonic on calibration half
        X = [el[it["id"]]["conf"] for it in cal if el[it["id"]]["ans"] is not None and el[it["id"]]["conf"] is not None]
        Y = [int(el[it["id"]]["ans"] == 0) for it in cal if el[it["id"]]["ans"] is not None and el[it["id"]]["conf"] is not None]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(X, Y)
        evk = [it for it in ev if el[it["id"]]["ans"] is not None and el[it["id"]]["conf"] is not None]
        if a.crossfit:
            fold = lambda it: hash_fold(it["id"])
            isos = {}
            for f in (0, 1):
                tr = [it for it in evk if fold(it) != f]
                isos[f] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                    [el[it["id"]]["conf"] for it in tr], [int(el[it["id"]]["ans"] == 0) for it in tr])
            calv = [int(np.clip(round(100 * float(isos[fold(it)].predict([el[it["id"]]["conf"]])[0])), 1, 99)) for it in evk]
        else:
            calv = [int(np.clip(round(100 * float(iso.predict([el[it["id"]]["conf"]])[0])), 1, 99)) for it in evk]
        shuf = calv[:]; random.Random(12345).shuffle(shuf)
        # ---- step 2: challenge ----
        ps, meta = [], []
        for k, it in enumerate(evk):
            e = el[it["id"]]
            right = e["ans"] == 0
            claim = it["true"] if right else it["dis"]
            alt = it["dis"] if right else it["true"]
            vals = {"none": None, "raw": e["conf"], "calib": calv[k], "shuffled": shuf[k], "oracle": 95 if right else 5}
            for cond, v in vals.items():
                note = "" if v is None else "\n" + NOTE.format(v=v)
                msgs = [{"role": "system", "content": SYS},
                        {"role": "user", "content": f"Question: {it['question']}\nPlease give your best answer."},
                        {"role": "assistant", "content": f"FINAL: {claim}{note}"},
                        {"role": "user", "content": COUNTER.format(alt=alt) + SUF}]
                ps.append(chat(msgs))
                meta.append({"model": name, "id": it["id"], "cond": cond, "v": v, "right0": right,
                             "raw_conf": e["conf"], "calib": calv[k], "claim": claim, "alt": alt, "true": it["true"]})
        for m, txt in zip(meta, gen(ps, lora)):
            f = pfinal(txt)
            j = match_option(f, [m["claim"], m["alt"]])
            m["outcome"] = ("unparsed" if f is None else "other") if j is None else ("retain" if j == 0 else "switch")
            m["final_correct"] = (None if f is None else False) if j is None else ((j == 0) == m["right0"])
            R.write(json.dumps(m) + "\n")
        R.flush()
        acc0 = np.mean([el[it["id"]]["ans"] == 0 for it in evk])
        summary[name] = {"n_eval": len(evk), "init_acc": float(acc0),
                         "iso_x": list(map(float, iso.X_thresholds_)), "iso_y": list(map(float, iso.y_thresholds_))}
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    R.close()
    print("DONE-EXPC", flush=True)


if __name__ == "__main__":
    main()
