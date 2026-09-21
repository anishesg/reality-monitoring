# GPU queue for Section C part 2 (STAND) and the agent-chain replication — exact commands for della

Prereqs (once, login node): `python -m venv .venv && source .venv/bin/activate && pip install -r train/requirements.txt`;
`HF_HOME=/scratch/gpfs/$USER/hf bash slurm/prefetch.sh olmo tulu` and `... prefetch.sh contagion`. Find the partition:
`sinfo -o "%P %G %l %D"`. Everything below is resumable; re-submitting skips finished arms.

## 1. STAND (A3) and its DPO control (A2), 3 seeds x 2 backbones  — ~70 GPU-h, 24 jobs
    for S in 0 1 2; do
      PARTITION=<p> GRES=gpu:1 CONSTRAINT=<a100-80g> SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A2 A3"
    done
Output: results_ladder/<tag>/A3/summary.json (retain_correct, accept_valid_correction, pressure_abandon, counter_bare_abandon,
capability). Success criterion (prereg H2/H3): retain_correct >= 0.6 with accept_valid_correction >= 0.7 and MMLU/GSM8K/IFEval
within 1 point of A0. Compare to FIRM (doubt-only coverage, -12 to -21 pts).

## 2. Contagion, bf16 replication of the CPU Q8 runs  — ~2 GPU-h, 5 jobs
    PARTITION=<p> GRES=gpu:1 bash slurm/submit_contagion.sh
(uses the merged-turn chain protocol and the genuine cells by default now.)

## 3. Firewall: STAND agent inside the chain  — 0.5 GPU-h, after step 1
    FIREWALL=checkpoints/olmo/A3/merged bash slurm/submit_contagion.sh olmo_fw
Delivers q = P(wrong | prev wrong) for the trained agent and the reset of the chain (plan/AGENT_CHAIN_THEORY.md §3).

## 4. Real-recipe DPO (A1), only if the training-history claim stays in the paper  — ~18 GPU-h, 12 jobs
    for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"; done

## 5. Scale rung (rebuttal)  — ~12 GPU-h eval-only, then 32B A2
    bash slurm/measure_stages.sh olmo13 olmo32
    bash slurm/submit_ladder.sh --backbones olmo32 --arms "A0 A2"

Order if only one A100 is free at a time: 2 (20 min each) -> 1 seed-0 of A3 on olmo (~3 h) -> the rest of 1 -> 3 -> 4.
