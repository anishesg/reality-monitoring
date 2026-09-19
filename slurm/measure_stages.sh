#!/usr/bin/env bash
# Measurement-only stage ladder at scale: evaluate published SFT/DPO/RLVR checkpoints (13B, 32B) with the frozen v17 harness.
#   bash slurm/measure_stages.sh [olmo13 olmo32]        (env: PARTITION, CONSTRAINT, ACCOUNT; CAP=0 to skip capability suite)
# Results: results_ladder/stage_<tag>/<STAGE>/summary.json  (retain_correct, accept_valid_correction, pressure_abandon, ...)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p logs
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(olmo13 olmo32)
for TAG in "${TAGS[@]}"; do
  case $TAG in
    olmo13) PFX=allenai/OLMo-2-1124-13B; TP=1; GRES=gpu:1; MEM=80G;;
    olmo32) PFX=allenai/OLMo-2-0325-32B; TP=2; GRES=gpu:2; MEM=160G;;
    *) echo "unknown tag $TAG"; exit 1;;
  esac
  RES="$ROOT/results_ladder/stage_$TAG"; mkdir -p "$RES"
  for STAGE in SFT DPO RLVR; do
    [ -f "$RES/$STAGE/summary.json" ] && { echo "$TAG $STAGE done, skip"; continue; }
    SUF=$STAGE; [ "$STAGE" = RLVR ] && SUF=Instruct
    SB=(--partition="${PARTITION:-gpu}" --gres="$GRES" --mem="$MEM" --time=03:00:00); [ -n "${CONSTRAINT:-}" ] && SB+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SB+=(--account="$ACCOUNT")
    J=$(sbatch --parsable "${SB[@]}" --job-name=rm-stage-$TAG-$STAGE --export=ALL,MODEL=$PFX-$SUF,ARM=$STAGE,RES=$RES,VLLM_TP=$TP,CAP=${CAP:-1} slurm/eval_arm.sbatch | cut -d';' -f1)
    echo "$TAG $STAGE -> job $J ($PFX-$SUF, tp=$TP)"
  done
done
echo "watch: squeue -u \$USER ; summaries in results_ladder/stage_<tag>/<STAGE>/summary.json"
