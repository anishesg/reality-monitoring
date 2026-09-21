#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:20:00
#SBATCH --output=/u/ak8686/reality-monitoring/logs/%x_%j.out
set -euo pipefail
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q torch transformers peft accelerate hf_transfer vllm
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /u/ak8686/reality-monitoring
python train_solution_v2.py --model "Qwen/Qwen2.5-7B-Instruct" --arm "$ARM" --seed "$SEED" --phase eval \
  --out "results_v21/${ARM}_s${SEED}" --adapter-dir "adapters/${ARM}_s${SEED}"
echo "DONE-S2 ${ARM}_s${SEED}"
