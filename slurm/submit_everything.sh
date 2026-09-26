#!/usr/bin/env bash
# Submit the ENTIRE Slurm queue in priority order with dependencies, in one command. Resumable: every launcher skips finished
# work, so re-running this after failures only re-queues what is missing.
#   PARTITION=<p> GRES=gpu:1 CONSTRAINT=<a100-80g> bash slurm/submit_everything.sh [--core-only]
# --core-only stops after step 4 (what the paper needs by the deadline). Steps 5-9 are the strengthening set.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p logs
CORE_ONLY=0; [ "${1:-}" = "--core-only" ] && CORE_ONLY=1
export PARTITION="${PARTITION:-}" GRES="${GRES:-gpu:1}"
SBX=(); [ -n "$PARTITION" ] && SBX+=(--partition="$PARTITION"); [ -n "${CONSTRAINT:-}" ] && SBX+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SBX+=(--account="$ACCOUNT")
say() { printf '\n== %s\n' "$*"; }
[ -f data/olmo/revision.jsonl ] && [ -f data/tulu/revision.jsonl ] || { echo "run 'bash slurm/prefetch.sh olmo tulu' on the login node first"; exit 1; }

say "1. contagion replication (5 jobs)";              bash slurm/submit_contagion.sh
say "2. real-recipe DPO (A1), 3 seeds x 2 backbones  [makes the stage-ladder claim causal]";  for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"; done
say "3. STAND + DPO control, 3 seeds x 2 backbones";  for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A2 A3"; done
say "4. firewall, queued after the seed-0 olmo A3 training job"
A3=$(squeue -u "$USER" -h -n rm-grpo-olmo-A3 -o %i | head -1)
if [ -n "$A3" ]; then
  sbatch --parsable "${SBX[@]}" --gres="$GRES" --dependency=afterok:"$A3" --job-name=rm-ctg-olmo_fw \
    --export="ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,TAG=olmo_fw,PEERS=weak=Qwen/Qwen2.5-1.5B-Instruct,same=allenai/OLMo-2-1124-7B-Instruct,strong=Qwen/Qwen2.5-14B-Instruct,N=300,K=8,FIREWALL=$ROOT/checkpoints/olmo/A3/merged,FIREWALL_POS=2,PEER_MSGS=results_contagion/olmo/peer_msgs.jsonl" slurm/contagion.sbatch
else echo "   (no olmo A3 job in queue; run later: FIREWALL=checkpoints/olmo/A3/merged bash slurm/submit_contagion.sh olmo_fw)"; fi
[ $CORE_ONLY = 1 ] && { say "core queue submitted; strengthening set skipped (--core-only)"; squeue -u "$USER" -o "%.10i %.22j %.2t %.8M" | head -40; exit 0; }

say "5. 13B/32B stage checkpoints, eval only";         bash slurm/measure_stages.sh olmo13 olmo32
say "6. contagion at 32B / 72B (needs weights prefetched; see plan/DELLA_PACKAGE.md step 6)"
for spec in "Qwen/Qwen2.5-32B-Instruct|qwen32b|1|120G|1" "Qwen/Qwen2.5-72B-Instruct|qwen72b|2|200G|2"; do
  IFS="|" read -r M T G MEM TP <<<"$spec"; [ -f "results_contagion/$T/meta.json" ] && { echo "   $T done"; continue; }
  sbatch --parsable "${SBX[@]}" --gres=gpu:$G --mem=$MEM --time=04:00:00 --job-name=rm-ctg-$T \
    --export="ALL,VLLM_TP=$TP,MODEL=$M,TAG=$T,PEERS=weak=Qwen/Qwen2.5-1.5B-Instruct,same=$M,strong=Qwen/Qwen2.5-72B-Instruct,N=300,K=8" slurm/contagion.sbatch || true
done
say "7. STAND at 32B, one seed";                       [ -f data/olmo32/revision.jsonl ] && bash slurm/submit_ladder.sh --backbones olmo32 --arms "A0 A2 A3" || echo "   (prefetch olmo32 first: bash slurm/prefetch.sh olmo32)"
say "8. A2 sweep on the DEV split";                    [ -f results_ladder/olmo/sweep/sweep_selection.json ] || sbatch --parsable "${SBX[@]}" --gres="$GRES" --mem=80G --time=10:00:00 --job-name=rm-sweep --wrap="source slurm/_common.sh; bash train/sweep_a2.sh"
say "9. v2 epistemic ladder: run separately; not auto-submitted"
say "queued. squeue:"; squeue -u "$USER" -o "%.10i %.22j %.2t %.8M %R" | head -60
