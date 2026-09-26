#!/usr/bin/env bash
# The CORE queue for the ICLR submission, in priority order, with Slurm dependencies. One command; safe to re-run (finished work is skipped).
#   CONSTRAINT=gpu80 bash slurm/submit_core.sh            # A100 80GB nodes (constraint name is cluster-specific)
#   SEEDS="0" GPU=a6000 bash slurm/submit_core.sh         # one seed of everything on 48 GB cards
#   PARTITION=<p> GRES=gpu:1 CONSTRAINT=<c> ACCOUNT=<a> bash slurm/submit_core.sh
# Order: 1) A1 (generic DPO) on both backbones, 3 seeds  -> the origin claim (H1)      ~16 GPU-h per seed incl. evals
#        2) A3 (STAND, GRPO) + A2 (revision-DPO control) on OLMo, 3 seeds -> the fix (H2/H3/H5)   ~60 GPU-h
#        3) the same on Tulu                                                              ~60 GPU-h
#        4) firewall chain, after the seed-0 OLMo A3 training job                        ~1 GPU-h
# Every train job is followed by its eval job (challenge cells + MMLU/GSM8K/IFEval capability suite) via --dependency=afterok.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p logs
export GRES="${GRES:-gpu:1}" GPU="${GPU:-a100}"; SEEDS="${SEEDS:-0 1 2}"   # SEEDS="0" runs one seed of everything (about a third of the GPU-hours)
if [ "$GPU" = a6000 ]; then   # 48 GB cards, ~2.5x slower than A100-80GB: longer wall-time caps, GRPO on 2 GPUs (slurm/train_grpo_2gpu.sbatch)
  export TIME_DPO="${TIME_DPO:-08:00:00}" TIME_GRPO="${TIME_GRPO:-30:00:00}" TIME_EVAL="${TIME_EVAL:-04:00:00}" VLLM_MEM="${VLLM_MEM:-0.45}"; FW_TIME=08:00:00
else FW_TIME=04:00:00; fi
SBX=(); [ -n "${PARTITION:-}" ] && SBX+=(--partition="$PARTITION"); [ -n "${CONSTRAINT:-}" ] && SBX+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SBX+=(--account="$ACCOUNT")
say() { printf '\n== %s\n' "$*"; }
[ -d .venv ] || { echo "FATAL: .venv missing. Run: bash RUN_ME.sh"; exit 1; }
[ -f data/olmo/generic.jsonl ] && [ -f data/tulu/generic.jsonl ] && [ -f data/olmo/revision.jsonl ] && [ -f data/tulu/revision.jsonl ] || { echo "FATAL: training data missing. Run: bash RUN_ME.sh (it prefetches models and builds data on the login node)"; exit 1; }
say "1. A1 generic DPO, both backbones, seeds 0 1 2  [origin claim, H1]"
for S in $SEEDS; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"; done
say "2. A3 STAND + A2 control on OLMo, seeds 0 1 2  [the fix, H2/H3/H5]"
for S in $SEEDS; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo" --arms "A3 A2"; done
say "3. A3 STAND + A2 control on Tulu, seeds 0 1 2"
for S in $SEEDS; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "tulu" --arms "A3 A2"; done
say "4. firewall chain, queued after the seed-0 OLMo A3 training job  [P4]"
A3=$(squeue -u "$USER" -h -n rm-grpo-olmo-A3 -o %i | head -1)
if [ -n "$A3" ] && [ ! -f results_contagion/olmo_fw/summary.json ]; then
  sbatch --parsable "${SBX[@]}" --gres="$GRES" --mem=80G --time=$FW_TIME --dependency=afterok:"$A3" --job-name=rm-ctg-olmo_fw \
    --export="ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,TAG=olmo_fw,PEERS=weak=Qwen/Qwen2.5-1.5B-Instruct,same=allenai/OLMo-2-1124-7B-Instruct,strong=Qwen/Qwen2.5-14B-Instruct,N=300,K=8,FIREWALL=$ROOT/checkpoints/olmo/A3/merged,FIREWALL_POS=2,PEER_MSGS=results_contagion/olmo/peer_msgs.jsonl" slurm/contagion.sbatch
else echo "   (no OLMo A3 job in queue or firewall already done; later: FIREWALL=checkpoints/olmo/A3/merged bash slurm/submit_contagion.sh olmo_fw)"; fi
say "queued. Watch with: squeue -u \$USER -o '%.10i %.22j %.2t %.10M %R'"
squeue -u "$USER" -o "%.10i %.22j %.2t %.10M %R" | head -60
