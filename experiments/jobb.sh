#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/u/ak8686/reality-monitoring/logs/%x_%j.out
set -euo pipefail
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q vllm hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /u/ak8686/reality-monitoring
python run_cells_v17.py --model "$MODEL" --claims claims_hard_eval.jsonl --n-questions 300 --out "results_v19/base_v17eval"
python run_cells_v18.py --model "$MODEL" --claims claims_hard_eval.jsonl --n-questions 250 --out "results_v19/base_v18eval"
echo "DONE-BASE"
