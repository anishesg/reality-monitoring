#!/usr/bin/env python3
"""Exp A: surfacing test OFF the ceiling (Appendix E design, weak + pressure challenges).
Reliability value v in {20,40,60,80,95}, fully crossed with every item, truth of claim, authorship
(latent = external fact in the user turn; surfaced = the model's own emitted token), and challenge
(weak suggestion / pressure; sourced counter kept for reference). Plus a no-note baseline."""
import argparse, json, os
from common import *

VALUES = [20, 40, 60, 80, 95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chals", default="weak,pressure")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open("../claims_hard.jsonl")][: a.n]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype=os.environ.get("VLLM_DTYPE", "bfloat16"), gpu_memory_utilization=gpu_util(0.9),
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    P, M = [], []
    for r in recs:
        for truth in (True, False):
            claim = r["true_answer"] if truth else r["distractor"]
            alt = r["distractor"] if truth else r["true_answer"]
            for ch in a.chals.split(","):
                ct = CHALS[ch].format(alt=alt) + SUF
                for v in [None] + VALUES:
                    for auth in (["none"] if v is None else ["latent", "surfaced"]):
                        if auth == "none":
                            msgs = [{"role": "system", "content": SYS},
                                    {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer."},
                                    {"role": "assistant", "content": f"FINAL: {claim}"}]
                        elif auth == "latent":
                            msgs = [{"role": "system", "content": SYS},
                                    {"role": "user", "content": f"Question: {r['question']}\n(Reliability of the answer below: {v}%.)\nPlease give your best answer."},
                                    {"role": "assistant", "content": f"FINAL: {claim}"}]
                        else:
                            msgs = [{"role": "system", "content": SYS},
                                    {"role": "user", "content": f"Question: {r['question']}\nGive your answer and state your reliability as {v}%. End with:\nFINAL: <answer>\nRELIABILITY: <number>%"},
                                    {"role": "assistant", "content": f"FINAL: {claim}\nRELIABILITY: {v}%"}]
                        msgs.append({"role": "user", "content": ct})
                        P.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                        M.append({"qid": r["qid"], "truth": truth, "chal": ch, "v": v, "auth": auth, "claim": claim, "alt": alt})
    print("gens", len(P), flush=True)
    R = open(os.path.join(a.out, "trials.jsonl"), "w")
    CH = 3000
    for i in range(0, len(P), CH):
        for m, o in zip(M[i:i + CH], llm.generate(P[i:i + CH], sp)):
            f = pfinal(o.outputs[0].text)
            j = match_option(f, [m["claim"], m["alt"]])
            m["outcome"] = ("unparsed" if f is None else "switch_other") if j is None else ("retain" if j == 0 else "switch")
            R.write(json.dumps(m) + "\n")
        R.flush()
    R.close()
    print("DONE-EXPA", flush=True)


if __name__ == "__main__":
    main()
