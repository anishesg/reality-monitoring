#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:45:00
#SBATCH --output=$HOME/reality-monitoring/logs/%x_%j.out
set -euo pipefail
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q torch transformers peft accelerate hf_transfer vllm
export HF_HUB_ENABLE_HF_TRANSFER=1
cd $HOME/reality-monitoring
python train_solution_v2.py --model "Qwen/Qwen2.5-7B-Instruct" --arm conf2 --seed "$SEED" --phase train \
  --lr 4e-5 --epochs 1 --replay 2.0 \
  --out "results_v22/conf3_s${SEED}" --adapter-dir "adapters/conf3_s${SEED}"
python train_solution_v2.py --model "Qwen/Qwen2.5-7B-Instruct" --arm conf2 --seed "$SEED" --phase eval \
  --out "results_v22/conf3_s${SEED}" --adapter-dir "adapters/conf3_s${SEED}"
echo "DONE-S3 conf3_s${SEED}"
