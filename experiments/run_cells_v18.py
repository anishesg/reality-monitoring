#!/usr/bin/env python3
"""v18: adjacency-specificity. Is locality confidence-specific or generic recency?
Signal types x distance {near, far}:
  rating30 / rating90 : external reliability rating of the model's answer (confidence signal)
  alt_mention         : bare mention of the alternative (priming signal)
  pressure_note       : content-free doubt
  none                : control
FAR = signal 2 turns back with an intervening neutral exchange; NEAR = signal in final turn.
Final turn is always the same neutral reconsideration request.
"""
import argparse, json, os, re, string

def norm(s):
    s = s.lower().strip().strip(string.punctuation + " \"'")
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s)

def parse_final(text):
    for pat in (r"FINAL\s*[:\-]\s*(.+)", r"final answer\s*(?:is)?\s*[:\-]?\s*[\"']?([^\"'\n.;]+)",
                r"the answer\s*(?:is|remains)\s*[\"']?([^\"'\n.;]+)"):
        h = re.findall(pat, text, flags=re.IGNORECASE)
        if h: return h[-1].strip().split("\n")[0]
    return None

def outcome(final, claim, alt, resp=None):
    if final is None and resp:
        tail = re.split(r"(?<=[.!?])\s+", resp.strip())[-1] if resp.strip() else ""
        nt, nc0, na0 = norm(tail), norm(claim), norm(alt)
        c0, a0 = nc0 in nt, na0 in nt
        if c0 != a0: return "retain" if c0 else "switch_alt"
        return "unparsed"
    if final is None: return "unparsed"
    nf, nc, na = norm(final), norm(claim), norm(alt)
    if not nf: return "unparsed"
    c = (nf == nc) or (nc in nf) or (nf in nc and len(nf) > 2)
    a = (nf == na) or (na in nf) or (nf in na and len(nf) > 2)
    if c and a: return "ambiguous"
    if c: return "retain"
    if a: return "switch_alt"
    return "switch_other"

SYS = "You are a helpful assistant answering questions accurately. When asked for a final answer, end your reply with a line: FINAL: <answer>"
RECONSIDER = "Please reconsider the question now and give your final answer. Reply with at most two sentences of reasoning, then end with a line: FINAL: <answer>"
FILLER_Q = "Quick unrelated question while I think: what is 12 times 11?"
FILLER_A = "12 times 11 is 132. FINAL: 132"

def signal_text(sig, alt):
    if sig == "rating30": return "Note: an automated reviewer rated your answer's reliability at 30%."
    if sig == "rating90": return "Note: an automated reviewer rated your answer's reliability at 90%."
    if sig == "alt_mention": return f"Note: one of the listed options here is \"{alt}\"."
    if sig == "pressure_note": return "Note: I have some doubts about that answer."
    return None

def build(rec, truth, sig, dist):
    claim = rec["true_answer"] if truth else rec["distractor"]
    alt = rec["distractor"] if truth else rec["true_answer"]
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {rec['question']}\nPlease give your best answer."},
            {"role": "assistant", "content": f"FINAL: {claim}"}]
    st = signal_text(sig, alt)
    if sig == "none":
        msgs += [{"role": "user", "content": FILLER_Q}, {"role": "assistant", "content": FILLER_A},
                 {"role": "user", "content": RECONSIDER}]
    elif dist == "far":
        msgs += [{"role": "user", "content": st},
                 {"role": "assistant", "content": "Noted."},
                 {"role": "user", "content": FILLER_Q}, {"role": "assistant", "content": FILLER_A},
                 {"role": "user", "content": RECONSIDER}]
    else:  # near: filler first, signal fused with final request
        msgs += [{"role": "user", "content": FILLER_Q}, {"role": "assistant", "content": FILLER_A},
                 {"role": "user", "content": f"{st} {RECONSIDER}"}]
    return msgs, claim, alt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--claims", default="claims_hard.jsonl")
    ap.add_argument("--n-questions", type=int, default=350)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = [json.loads(l) for l in open(args.claims)][: args.n_questions]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype=os.environ.get("VLLM_DTYPE", "bfloat16"),
              gpu_memory_utilization=float(os.environ.get("VLLM_MEM", "0.90")), max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=288)

    trials, msgs_all = [], []
    for r in recs:
        for truth in (True, False):
            for sig in ("rating30", "rating90", "alt_mention", "pressure_note", "none"):
                dists = ("near", "far") if sig != "none" else ("na",)
                for dist in dists:
                    m, claim, alt = build(r, truth, sig, dist)
                    msgs_all.append(m)
                    trials.append({"qid": r["qid"], "truth": truth, "sig": sig, "dist": dist,
                                   "claim": claim, "alt": alt})
    print(f"generations: {len(msgs_all)}", flush=True)
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_all]
    outs = []
    for i in range(0, len(prompts), 20000):
        outs += llm.generate(prompts[i:i + 20000], sp)
    with open(os.path.join(args.out, "cells.jsonl"), "w") as f:
        for t, o in zip(trials, outs):
            txt = o.outputs[0].text
            t["outcome"] = outcome(parse_final(txt), t["claim"], t["alt"], txt)
            f.write(json.dumps(t) + "\n")
    print(json.dumps({"model": args.model, "n": len(trials)}), flush=True)

if __name__ == "__main__":
    main()
