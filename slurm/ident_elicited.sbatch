#!/bin/bash
#SBATCH --job-name=rm-ident-el
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x_%j.out
# env: MODEL (HF id), TAG, CLAIMS (path to claims3.jsonl built by experiments/claimbank_3cell.py on the login node), N (500)
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
python experiments/run_identification_elicited.py --model "${MODEL:?}" --claims "${CLAIMS:-claims3.jsonl}" --n "${N:-500}" --out "results_ident_elicited/${TAG:?}" &
wait $!
echo "DONE-IDENT-ELICITED $TAG"
