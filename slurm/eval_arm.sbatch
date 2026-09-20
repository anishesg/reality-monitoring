#!/bin/bash
#SBATCH --job-name=rm-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x_%j.out
# env: MODEL (merged dir or HF id), ARM, RES (results root), CAP=1 to add capability suite
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
C=""; [ "${CAP:-1}" = "1" ] && C="--capability"
P=""; [ "${PARA:-1}" = "1" ] && P="--paraphrase"
python train/eval_arm.py --model "$MODEL" --arm "$ARM" --results-root "$RES" $C $P &
wait $!
echo "DONE-EVAL $ARM"
