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

def build_trials(recs, paraphrase=False):
    """Same order and content as harness/run_cells_v17.py (self: 4 kinds; user_recency: 2 kinds). With paraphrase=True the
    self-origin challenges use the held-out PARAPHRASE wordings (never in training data); user cells are skipped."""
    from common import paraphrase_text
    trials, msgs = [], []
    for r in recs:
        for truth in (True, False):
            claim = r["true_answer"] if truth else r["distractor"]; alt = r["distractor"] if truth else r["true_answer"]
            for conf in ("low", "high"):
                for j, k in enumerate(KINDS):
                    m = self_dialogue(r["question"], claim, conf, k, alt)
                    if paraphrase: m[-1]["content"] = paraphrase_text(k, alt, r["qid"] + j)
                    msgs.append(m)
                    trials.append({"qid": r["qid"], "origin": "self", "truth": truth, "conf": conf, "kind": k, "claim": claim, "alt": alt})
                if not paraphrase:
                    for k in USER_KINDS:
                        msgs.append(user_recency_dialogue(r["question"], claim, conf, k, alt))
                        trials.append({"qid": r["qid"], "origin": "user_recency", "truth": truth, "conf": conf, "kind": k, "claim": claim, "alt": alt})
    return trials, msgs

def run_v17_resumable(model, out_dir, n_q, claims_path=None, chunk=1200, paraphrase=False, gen=None):
    """Generate in chunks and APPEND to <out_dir>/cells.jsonl with a trial index, so a preempted job resumes where it stopped.
    Returns (backend, generator) so the generator can be reused for the paraphrase pass."""
    recs = read_jsonl(claims_path or CLAIMS)[:n_q]
    trials, msgs = build_trials(recs, paraphrase)
    os.makedirs(out_dir, exist_ok=True)
    cells = os.path.join(out_dir, "cells.jsonl")
    done = set()
    if os.path.exists(cells):
        rows = [json.loads(l) for l in open(cells) if l.strip()]
        done = {r["i"] for r in rows if "i" in r}
        if rows and "i" not in rows[0]:  # legacy complete file from the harness subprocess path
            return "existing", gen
        print(f"[eval_arm] resuming {out_dir}: {len(done)}/{len(trials)} trials done", flush=True)
    todo = [i for i in range(len(trials)) if i not in done]
    if todo:
        gen = gen or Generator(model, tp=int(os.environ.get("VLLM_TP", "1")))
        with open(cells, "a") as f:
            for s0 in range(0, len(todo), chunk):
                idx = todo[s0:s0 + chunk]
                texts = gen.chat([msgs[i] for i in idx], max_tokens=288)
                for i, txt in zip(idx, texts):
                    t = dict(trials[i]); t["i"] = i
                    t["outcome"] = outcome(parse_final(txt), t["claim"], t["alt"], txt); t["resp"] = txt[-300:]
                    f.write(json.dumps(t) + "\n")
                f.flush(); os.fsync(f.fileno())
                print(f"[eval_arm] {min(s0 + chunk, len(todo))}/{len(todo)} generated", flush=True)
    return (gen.backend if gen else "existing"), gen

def summarize(arm, model, arm_dir, cells, backend, split, extra=None):
    an_root = os.path.join(arm_dir, "_an"); os.makedirs(os.path.join(an_root, arm), exist_ok=True)
    link = os.path.join(an_root, arm, "cells.jsonl")
    if os.path.lexists(link): os.remove(link)
    os.symlink(os.path.abspath(cells), link)
    subprocess.run([sys.executable, ANALYZE, an_root], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    A = json.load(open(os.path.join(an_root, "all.json")))[0]
    row = {"arm": arm, "model": model, "n": A["n"], "backend": backend, "split": split,
           "abandon_by_kind": A["abandon_by_kind"], "source_effect": A["source_effect_content_matched"]["delta"],
           "conf_use_self_counter_src": A["conf_use_self_by_kind"]["counter_src"]["delta"],
           "retain_correct": A["bidirectional"]["retain_correct"],
           "accept_valid_correction": A["bidirectional"]["accept_valid_correction"],
           "reject_invalid_pressure": A["bidirectional"]["reject_invalid_pressure"],
           "pressure_abandon": A["abandon_by_kind"]["pressure"], "counter_bare_abandon": A["abandon_by_kind"]["counter_bare"]}
    T = read_jsonl(cells)
    row["excluded_frac"] = round(sum(t["outcome"] in ("unparsed", "ambiguous") for t in T) / max(len(T), 1), 3)
    # exclusion-robust variants: an unparsed/ambiguous reply counts as NOT retained and NOT a valid switch (guards against a
    # trained arm looking better only because it formats FINAL: more reliably)
    tr = lambda t: t["truth"] in (True, "True", 1)
    K = [t for t in T if t["origin"] == "self" and t["kind"] == "counter_src"]
    row["retain_correct_all_trials"] = round(sum(t["outcome"] == "retain" for t in K if tr(t)) / max(sum(1 for t in K if tr(t)), 1), 3)
    row["accept_valid_all_trials"] = round(sum(t["outcome"] == "switch_alt" for t in K if not tr(t)) / max(sum(1 for t in K if not tr(t)), 1), 3)
    row["switch_other_frac"] = round(sum(t["outcome"] == "switch_other" for t in T) / max(len(T), 1), 3)
    if extra: row.update(extra)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint dir or HF id")
    ap.add_argument("--arm", required=True); ap.add_argument("--results-root", default=os.path.join(ROOT, "results_ladder"))
    ap.add_argument("--n-questions", type=int, default=EVAL_HOLDOUT)
    ap.add_argument("--capability", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", choices=("test", "dev"), default="test", help="test = held-out qids 0-449 (paper); dev = qids 800-899 (hyperparameter selection only)")
    ap.add_argument("--chunk", type=int, default=1200, help="trials per generation chunk (each chunk is appended to cells.jsonl; resume granularity)")
    ap.add_argument("--paraphrase", action="store_true", help="also evaluate the held-out challenge paraphrases -> <arm>/paraphrase/, summary key paraphrase")
    ap.add_argument("--harness-subprocess", action="store_true", help="legacy one-shot path through harness/run_cells_v17.py (not resumable)")
    ap.add_argument("--no-refresh", action="store_true", help="do not regenerate results/scale_table.csv + figures/scale after writing the summary")
    args = ap.parse_args()
    if is_adapter_only(args.model):
        sys.exit(f"[eval_arm] {args.model} is an adapter-only directory; run dpo.py/grpo.py with --merge and point at <out>/merged")
    claims_path = CLAIMS
    if args.split == "dev":
        dev = [r for r in read_jsonl(CLAIMS) if 800 <= r["qid"] <= 899]
        claims_path = os.path.join(args.results_root, "dev_claims.jsonl"); write_jsonl(claims_path, dev)
        args.n_questions = min(args.n_questions, len(dev)); args.results_root = os.path.join(args.results_root, "dev")
    else:
        assert args.n_questions <= EVAL_HOLDOUT, "test eval must stay inside the held-out first 450 hard-bank qids"
    arm_dir = os.path.join(args.results_root, args.arm); os.makedirs(arm_dir, exist_ok=True)
    summ = os.path.join(arm_dir, "summary.json")
    if os.path.exists(summ) and not args.force:
        print(f"[eval_arm] {summ} exists; skip (use --force)"); return
    cells = os.path.join(arm_dir, "cells.jsonl")
    gen = None
    if args.harness_subprocess:
        backend = "vllm"
        subprocess.run([sys.executable, HARNESS, "--model", args.model, "--claims", claims_path,
                        "--n-questions", str(args.n_questions), "--out", arm_dir], check=True, env=dict(os.environ), cwd=ROOT)
    else:
        backend, gen = run_v17_resumable(args.model, arm_dir, args.n_questions, claims_path, chunk=args.chunk)
    extra = {}
    if args.paraphrase:
        pdir = os.path.join(arm_dir, "paraphrase")
        pb, gen = run_v17_resumable(args.model, pdir, args.n_questions, claims_path, chunk=args.chunk, paraphrase=True, gen=gen)
        prow = summarize(args.arm, args.model, pdir, os.path.join(pdir, "cells.jsonl"), pb, args.split)
        extra["paraphrase"] = {k: prow[k] for k in ("retain_correct", "accept_valid_correction", "pressure_abandon", "counter_bare_abandon",
                                                    "excluded_frac", "retain_correct_all_trials", "accept_valid_all_trials")}
    if gen is not None:
        del gen
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    row = summarize(args.arm, args.model, arm_dir, cells, backend, args.split, extra)
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
    if not args.no_refresh:  # running record: table + figures are regenerated after every finished arm
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "scale_compare.py")], cwd=ROOT, check=False)

if __name__ == "__main__":
    main()
