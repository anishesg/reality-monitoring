# Della Experiment Spec: Confidence Without Control (oral-version experiments)

These are the experiments that exceed ionic's 48GB cards or benefit from A100/H100 scale.
Everything here reuses the harness in `experiments/` (repo: reality-monitoring). Compute
estimates assume A100-80GB. All jobs follow the della pattern already in `della/`: downloads
on the login node (compute nodes have no internet), HF cache on /scratch/gpfs.

## D1. Scale ladder, unquantized, single protocol (closes the scaling objection)
Run `run_identification.py --claims claims3.jsonl` (the 3-cell identification + 4-signal
triangulation script) on, all bf16 unquantized:
  Qwen2.5-32B-Instruct (1x A100), Qwen2.5-72B-Instruct (2x A100 TP2),
  Qwen3-32B (1x A100), Llama-3.1-70B-Instruct (2x A100 TP2, needs HF token with license),
  plus one current frontier-adjacent open model of choice (e.g. Qwen3-30B-A3B).
Purpose: the confidence-use gap curve (4 signals AUROC vs behavioral use) extended to 70B
on ONE protocol. This is Figure panel A at full range.
Est: ~10 A100-hours total.

## D2. Repair that works (the method contribution)
Goal: conditioning on OWN elicited confidence with <= 2 points external-capability cost.
Script: extend `train_solution_v2.py` (see experiments/) with:
  - replay from a real instruction mixture (allenai/tulu-3-sft-mixture, 30-50% of examples)
  - training questions from BOTH SciQ and a disjoint MMLU-Pro slice (diversity)
  - trajectories that use the model's own elicited confidence statements, not only injected
  - arms: {conditioning + real-replay} x lr {5e-5, 1e-4} x replay {30%, 50%} x 3 seeds
  - control arm (confidence sentences deleted) x 3 seeds
Eval per arm: (a) held-out conditioning curve (unseen values/phrasings), (b) post-challenge
accuracy with own elicited confidence, (c) EXTERNAL capability: 500-item MMLU slice + GSM8K
200 + our FC bank. The deliverable figure: conditioning strength vs capability cost frontier
across arms; the claim is the point nearest (full conditioning, zero cost).
Est: 24 jobs x ~1.5 A100-hours = ~36 A100-hours. This is the oral-maker; do not shrink it.

## D3. Open-ended format replication (kills the MCQ objection)
Core cells (3-cell identification, no visible options; answers scored by normalized match
+ an NLI/judge fallback) on 4 anchor models + Qwen3-32B.
Est: ~6 A100-hours.

## D4. Legitimate re-ask experiment (replaces the tautological 1.00)
Paraphrased re-ask (5 paraphrases per question, GPT-generated bank committed in repo) x
3 sampling seeds x ALL items (not conditioned on initial correctness), vs in-context
reconsideration under the same seeds. Report accuracy distributions, not a single number.
Est: ~4 A100-hours.

## Priorities if compute-limited
D2 > D1 > D3 > D4. D2 alone upgrades the paper class.

## What ionic is already running (do not duplicate)
3-cell identification + 4-signal triangulation on Qwen2.5-{7B,14B}, Llama-3.1-8B,
OLMo-2-7B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B (jobs ID_* of Sep 20).
