#!/usr/bin/env python3
"""Evaluate one ladder arm on MERGED weights.
1) harness/run_cells_v17.py on the held-out first 450 hard-bank questions (vLLM; transformers fallback when vLLM is absent)
2) analysis/analyze_v17.py on that output
3) optional capability check via lm-eval (mmlu 5-shot, gsm8k, ifeval; --quick subsamples)
Writes <results-root>/<arm>/summary.json (one row per arm). Paths are anchored to the git repo root.
Refuses adapter-only directories: run_cells_v17.py loads plain weights (no LoRA).
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, EVAL_HOLDOUT, read_jsonl, write_jsonl, Generator, self_dialogue, user_recency_dialogue, KINDS, USER_KINDS, parse_final, outcome

HARNESS = os.path.join(ROOT, "harness", "run_cells_v17.py")
ANALYZE = os.path.join(ROOT, "analysis", "analyze_v17.py")
CLAIMS = os.path.join(ROOT, "harness", "claims_hard.jsonl")

def is_adapter_only(path):
    if not os.path.isdir(path): return False
    has_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_weights = any(f.endswith((".safetensors", ".bin")) and not f.startswith("adapter") for f in os.listdir(path))
    return has_adapter and not has_weights

def run_v17_hf(model, out_dir, n_q):
    """Mirror of run_cells_v17.main() using the transformers backend (Mac / no-vLLM smoke path)."""
    recs = read_jsonl(CLAIMS)[:n_q]
    trials, msgs = [], []
    for r in recs:
        for truth in (True, False):
            claim = r["true_answer"] if truth else r["distractor"]; alt = r["distractor"] if truth else r["true_answer"]
            for conf in ("low", "high"):
                for k in KINDS:
                    msgs.append(self_dialogue(r["question"], claim, conf, k, alt))
                    trials.append({"qid": r["qid"], "origin": "self", "truth": truth, "conf": conf, "kind": k, "claim": claim, "alt": alt})
                for k in USER_KINDS:
                    msgs.append(user_recency_dialogue(r["question"], claim, conf, k, alt))
                    trials.append({"qid": r["qid"], "origin": "user_recency", "truth": truth, "conf": conf, "kind": k, "claim": claim, "alt": alt})
    gen = Generator(model)
    texts = gen.chat(msgs, max_tokens=288)
    for t, txt in zip(trials, texts):
        t["outcome"] = outcome(parse_final(txt), t["claim"], t["alt"], txt); t["resp"] = txt[-300:]
    write_jsonl(os.path.join(out_dir, "cells.jsonl"), trials)
    return gen.backend

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint dir or HF id")
    ap.add_argument("--arm", required=True); ap.add_argument("--results-root", default=os.path.join(ROOT, "results_ladder"))
    ap.add_argument("--n-questions", type=int, default=EVAL_HOLDOUT)
    ap.add_argument("--capability", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if is_adapter_only(args.model):
        sys.exit(f"[eval_arm] {args.model} is an adapter-only directory; run dpo.py/grpo.py with --merge and point at <out>/merged")
    assert args.n_questions <= EVAL_HOLDOUT, "eval must stay inside the held-out first 450 hard-bank qids"
    arm_dir = os.path.join(args.results_root, args.arm); os.makedirs(arm_dir, exist_ok=True)
    summ = os.path.join(arm_dir, "summary.json")
    if os.path.exists(summ) and not args.force:
        print(f"[eval_arm] {summ} exists; skip (use --force)"); return
    cells = os.path.join(arm_dir, "cells.jsonl")
    backend = "vllm"
    try:
        import vllm, torch  # noqa: F401
        assert torch.cuda.is_available()
        env = dict(os.environ)  # harness honors VLLM_TP / VLLM_MEM / VLLM_DTYPE
        subprocess.run([sys.executable, HARNESS, "--model", args.model, "--claims", CLAIMS,
                        "--n-questions", str(args.n_questions), "--out", arm_dir], check=True, env=env, cwd=ROOT)
    except (ImportError, AssertionError):
        backend = run_v17_hf(args.model, arm_dir, args.n_questions)
    # analyze_v17 globs <base>/*/cells.jsonl and writes <base>/all.json; run on a private parent dir
    an_root = os.path.join(arm_dir, "_an"); os.makedirs(os.path.join(an_root, args.arm), exist_ok=True)
    link = os.path.join(an_root, args.arm, "cells.jsonl")
    if os.path.lexists(link): os.remove(link)
    os.symlink(os.path.abspath(cells), link)
    subprocess.run([sys.executable, ANALYZE, an_root], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    A = json.load(open(os.path.join(an_root, "all.json")))[0]
    row = {"arm": args.arm, "model": args.model, "n": A["n"], "backend": backend,
           "abandon_by_kind": A["abandon_by_kind"], "source_effect": A["source_effect_content_matched"]["delta"],
           "conf_use_self_counter_src": A["conf_use_self_by_kind"]["counter_src"]["delta"],
           "retain_correct": A["bidirectional"]["retain_correct"],
           "accept_valid_correction": A["bidirectional"]["accept_valid_correction"],
           "reject_invalid_pressure": A["bidirectional"]["reject_invalid_pressure"],
           "pressure_abandon": A["abandon_by_kind"]["pressure"], "counter_bare_abandon": A["abandon_by_kind"]["counter_bare"]}
    T = read_jsonl(cells); row["excluded_frac"] = round(sum(t["outcome"] in ("unparsed", "ambiguous") for t in T) / max(len(T), 1), 3)
    if args.capability:
        lim = ["--limit", "200"] if args.quick else []
        cap_dir = os.path.join(arm_dir, "capability")
        try:
            base = ["lm_eval", "--model", "vllm" if backend == "vllm" else "hf",
                    "--model_args", f"pretrained={args.model},dtype=auto" + (f",gpu_memory_utilization=0.85,max_model_len=4096,tensor_parallel_size={os.environ.get('VLLM_TP', '1')}" if backend == "vllm" else ""),
                    "--batch_size", "auto"]
            # mmlu and gsm8k are 5-shot by convention; ifeval is strictly 0-shot (a global --num_fewshot would corrupt it)
            subprocess.run(base + ["--tasks", "mmlu,gsm8k", "--num_fewshot", "5", "--output_path", os.path.join(cap_dir, "fewshot")] + lim, check=True, cwd=ROOT)
            subprocess.run(base + ["--tasks", "ifeval", "--num_fewshot", "0", "--output_path", os.path.join(cap_dir, "zeroshot")] + lim, check=True, cwd=ROOT)
            import glob
            res = {}
            for f in sorted(glob.glob(os.path.join(cap_dir, "**", "results*.json"), recursive=True)):
                res.update(json.load(open(f))["results"])
            row["capability"] = {k: {m: v for m, v in res[k].items() if m.startswith(("acc", "exact", "prompt_level_strict"))} for k in res if k in ("mmlu", "gsm8k", "ifeval")}
        except Exception as e:  # noqa: BLE001
            row["capability"] = {"error": str(e)[:200]}
    json.dump(row, open(summ, "w"), indent=1)
    print(json.dumps(row, indent=1))

if __name__ == "__main__":
    main()
