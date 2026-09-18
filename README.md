# Reality Monitoring: Epistemic-State Use in LLM Revision

Code + pre-registrations + aggregate results for the study of how LLMs use declared
epistemic state (confidence, verification status, source) when revising claims under challenge.

## Key findings (as of this snapshot)
1. Revision is governed by **proximity and priming, not epistemic state**: declared confidence
   (own or other's) only influences revision when adjacent to the decision (v17 matched-recency
   control); the mere mention of an alternative triggers ~90% capitulation in some families.
2. **Scissors**: verbalized self-confidence informativeness grows with scale (AUROC .49->.70)
   while behavioral use stays ~flat (~.05).
3. **Capitulation plateau**: models abandon ~50% of their own correct answers to a bare false
   counter; no improvement 7B->14B.
4. **Stage ladder** (OLMo-2 / Tulu-3 / Zephyr, pre-registered): DPO-stage descendants show more
   challenge-damage in 3/3 lineages; RLVR-stage recovers robustness only.
5. Pre-registered "self-discounting" (P2) FAILED its matched-recency control -> honest reframe.
   Receipts in prereg/ (md5-timestamped before data).

## Layout
- harness/   claim banks + experiment generators (vLLM) + slurm job templates (Princeton ionic)
- analysis/  stdlib analyzers, bootstrap CIs, GEE + random-effects meta (statsmodels)
- mech/      probe + activation-steering (transformers hooks) - null result, appendix
- della/     scaled v2 package for the della cluster (scale ladder 0.5B-72B)
- prereg/    frozen predictions with hashes, BEFORE data
- results/   aggregate JSONs only (per-trial jsonl too large; on cluster)

## Reproduce
claims: python3 harness/claimbank_hard.py
run:    sbatch --gres=gpu:a5000:1 --export=ALL,MODEL=...,TAG=... harness/jobf.sh
analyze: python3 analysis/analyze_v17.py results_v17

Secrets (HF token) intentionally excluded; jobs read ~/.hftok on the cluster.
