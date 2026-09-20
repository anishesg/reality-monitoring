#!/bin/bash
#SBATCH --job-name=rm-dpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x_%j.out
# env: BACKBONE (HF id or merged dir), DATA (jsonl), OUT (ckpt dir), EXTRA (extra dpo.py flags)
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
python train/dpo.py --model "$BACKBONE" --data "$DATA" --out "$OUT" --merge --save-steps "${SAVE_STEPS:-50}" ${EXTRA:-} &
wait $!
echo "DONE-DPO $OUT"
