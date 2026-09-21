#!/usr/bin/env python3
"""External capability eval of a merged model dir: MMLU-slice forced choice."""
import argparse, json, os, random, re, string

def norm(s):
    s = s.lower().strip().strip(string.punctuation + " \"'")
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s)

def pfinal(t):
    for p in (r"FINAL\s*[:\-]\s*(.+)", r"final answer\s*(?:is)?\s*[:\-]?\s*[\"']?([^\"'\n.;]+)"):
        h = re.findall(p, t, re.I)
        if h: return h[-1].strip().split("\n")[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdir", required=True)
    ap.add_argument("--capfile", default="mmlu_cap.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.capfile)]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.mdir)
    llm = LLM(model=a.mdir, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    SYS = "You are a helpful assistant answering questions accurately. When asked for a final answer, end your reply with a line: FINAL: <answer>"
    prompts = []
    for r in recs:
        x, y = r["true_answer"], r["distractor"]
        if random.Random(r["qid"]).random() < 0.5: x, y = y, x
        prompts.append(tok.apply_chat_template([{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {r['question']}\nWhich answer is correct: \"{x}\" or \"{y}\"? End with a line: FINAL: <answer>"}],
            tokenize=False, add_generation_prompt=True))
    ok = tot = 0
    for r, o in zip(recs, llm.generate(prompts, sp)):
        f = pfinal(o.outputs[0].text)
        if f is None: continue
        nf, ng, nd = norm(f), norm(r["true_answer"]), norm(r["distractor"])
        g = ng in nf or nf == ng
        d = nd in nf or nf == nd
        if g == d: continue
        tot += 1; ok += g
    res = {"mmlu_fc_acc": round(ok / max(tot, 1), 4), "n": tot}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
