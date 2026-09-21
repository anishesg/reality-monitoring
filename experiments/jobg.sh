#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:55:00
#SBATCH --output=/u/ak8686/reality-monitoring/logs/%x_%j.out
set -euo pipefail
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
if [ -f /u/ak8686/reality-monitoring/.hftok ]; then export HF_TOKEN=$(cat /u/ak8686/reality-monitoring/.hftok); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; fi
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q vllm hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /u/ak8686/reality-monitoring
python run_cells_v18.py --model "$MODEL" --claims claims_hard.jsonl --out "results_v18/$TAG"
echo "DONE-V16 $TAG"
