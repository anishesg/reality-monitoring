#!/bin/bash
#SBATCH --job-name=rm-grpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# env: BACKBONE, DATA, OUT, EXTRA (e.g. --conf for A4); VLLM=0 disables colocated rollouts
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
VL="--vllm"; [ "${VLLM:-1}" = "0" ] && VL=""
python train/grpo.py --model "$BACKBONE" --data "$DATA" --out "$OUT" --merge $VL ${EXTRA:-}
echo "DONE-GRPO $OUT"
