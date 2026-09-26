#!/usr/bin/env python3
"""Elicited three-cell identification (2026-09-22). Same bank, templates, cells and grading as run_identification.py, but the
challenged claim is the model's OWN generated answer (its full reply stays in the transcript) instead of an injected "FINAL: x".
Closes the limitation stated in the paper: "the identification experiments inject claims rather than eliciting them".

Cells are assigned from what the model actually answered:
  own answer == true_answer  -> TF  (alt = d1)                      harmful switch
  own answer == d1           -> FT  (alt = true_answer), FF (alt = d2)
  own answer == d2           -> FT  (alt = true_answer), FF (alt = d1)
  own answer == neither      -> excluded (rate reported)
Outputs: <out>/own.jsonl (every generated answer, classified), <out>/ident.jsonl (same fields as the injected run plus
claim_src="elicited", own_correct, resp) so experiments/report_ident.py and the pooled regression run unchanged.
  python experiments/run_identification_elicited.py --model Qwen/Qwen2.5-7B-Instruct --claims claims3.jsonl --out results_ident_elicited/e_qwen7b
"""
import argparse, importlib.util, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ident", os.path.join(HERE, "run_identification.py")); ident = importlib.util.module_from_spec(spec); spec.loader.exec_module(ident)
norm, pfinal, outc, SYS, SUF, COUNTERS, PRESSURE = ident.norm, ident.pfinal, ident.outc, ident.SYS, ident.SUF, ident.COUNTERS, ident.PRESSURE

def classify(final, r):
    """Which candidate did the model give? true / d1 / d2 / None."""
    if final is None: return None
    nf = norm(final)
    for key in ("true_answer", "d1", "d2"):
        nc = norm(r[key])
        if nf == nc or nc in nf or (nf in nc and len(nf) > 2): return key
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--claims", default="claims3.jsonl")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--out", required=True); ap.add_argument("--maxtok", type=int, default=220)
    ap.add_argument("--backend", choices=("vllm", "api"), default="vllm", help="api: any OpenAI-compatible server (llama-server on the CPU VM), via harness/contagion.py ApiBackend")
    ap.add_argument("--api-base", default=None); ap.add_argument("--api-key-env", default="LLAMA_KEY"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open(a.claims)][: a.n]
    if a.backend == "api":
        import types
        cspec = importlib.util.spec_from_file_location("ctg", os.path.join(HERE, "..", "harness", "contagion.py")); ctg = importlib.util.module_from_spec(cspec); cspec.loader.exec_module(ctg)
        be = ctg.ApiBackend(types.SimpleNamespace(provider="openai", api_base=a.api_base, api_key_env=a.api_key_env, model=a.model, max_tokens=a.maxtok,
                                                  effort=None, seed=0, dry_run=False, workers=a.workers))
        gen = lambda msgs: be.generate(msgs)          # list of message lists -> list of texts (server applies the chat template)
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(a.model)
        llm = LLM(model=a.model, dtype=os.environ.get("VLLM_DTYPE", "bfloat16"), gpu_memory_utilization=float(os.environ.get("VLLM_MEM", "0.90")),
                  max_model_len=int(os.environ.get("VLLM_LEN", "3072")))
        spg = SamplingParams(temperature=0.0, max_tokens=a.maxtok)
        chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        gen = lambda msgs: [o.outputs[0].text for o in llm.generate([chat(m) for m in msgs], spg)]
    # ---- phase 0: the model's own answer, full reply kept
    q_msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer.{SUF}"}] for r in recs]
    own = {}
    with open(os.path.join(a.out, "own.jsonl"), "w") as f:
        for r, txt in zip(recs, gen(q_msgs)):
            fin = pfinal(txt); key = classify(fin, r)
            own[r["qid"]] = {"key": key, "resp": txt.strip(), "final": fin}
            f.write(json.dumps({"qid": r["qid"], "final": fin, "answered": key, "own_correct": key == "true_answer", "resp": txt.strip()}) + "\n")
    n_ex = sum(1 for v in own.values() if v["key"] is None); acc = sum(1 for v in own.values() if v["key"] == "true_answer") / len(recs)
    print(f"own answers: accuracy={acc:.3f} excluded(neither candidate)={n_ex}/{len(recs)}", flush=True)
    # ---- phase 1: cells from the own answer
    trials, prompts = [], []
    for r in recs:
        o = own[r["qid"]]
        if o["key"] is None: continue
        if o["key"] == "true_answer": cells = [("TF", r["d1"])]
        elif o["key"] == "d1": cells = [("FT", r["true_answer"]), ("FF", r["d2"])]
        else: cells = [("FT", r["true_answer"]), ("FF", r["d1"])]
        claim = r[o["key"]]
        base = [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer.{SUF}"},
                {"role": "assistant", "content": o["resp"]}]
        for cell, alt in cells:
            for ti, tmpl in enumerate(COUNTERS):
                prompts.append(base + [{"role": "user", "content": tmpl.format(alt=alt) + SUF}])
                trials.append({"qid": r["qid"], "cell": cell, "tmpl": ti, "chal": "counter", "claim": claim, "alt": alt, "claim_src": "elicited", "own_correct": o["key"] == "true_answer"})
            prompts.append(base + [{"role": "user", "content": PRESSURE + SUF}])
            trials.append({"qid": r["qid"], "cell": cell, "tmpl": -1, "chal": "pressure", "claim": claim, "alt": alt, "claim_src": "elicited", "own_correct": o["key"] == "true_answer"})
    print(f"challenge gens: {len(prompts)}", flush=True)
    with open(os.path.join(a.out, "ident.jsonl"), "w") as R:
        CH = 20000 if a.backend == "vllm" else 512
        for i in range(0, len(prompts), CH):
            for t, txt in zip(trials[i:i + CH], gen(prompts[i:i + CH])):
                t["outcome"] = outc(pfinal(txt), t["claim"], t["alt"], txt); t["resp"] = txt[-300:]
                R.write(json.dumps(t) + "\n")
            print(f"  {min(i + CH, len(prompts))}/{len(prompts)}", flush=True)
    json.dump({"model": a.model, "backend": a.backend, "n": len(recs), "own_accuracy": acc, "excluded": n_ex, "trials": len(trials)}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE-IDENT-ELICITED", a.out, flush=True)

if __name__ == "__main__":
    main()
