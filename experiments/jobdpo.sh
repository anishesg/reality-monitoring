#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:15:00
#SBATCH --output=$HOME/reality-monitoring/logs/%x_%j.out
set -euo pipefail
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q torch transformers datasets trl peft accelerate hf_transfer vllm
export HF_HUB_ENABLE_HF_TRANSFER=1
cd $HOME/reality-monitoring
python dpo_experiment.py --arm "$ARM" --seed "$SEED" --phase train \
  --out "results_v20/${ARM}_s${SEED}" --n-pairs 12000 --adapter-root "adapters/dpo_${ARM}_s${SEED}"
find "adapters/dpo_${ARM}_s${SEED}" -name "optimizer.pt" -delete 2>/dev/null || true
find "adapters/dpo_${ARM}_s${SEED}" -name "*.safetensors" -path "*global_step*" -delete 2>/dev/null || true
python dpo_experiment.py --arm "$ARM" --seed "$SEED" --phase eval \
  --out "results_v20/${ARM}_s${SEED}" --n-pairs 12000 --adapter-root "adapters/dpo_${ARM}_s${SEED}"
echo "DONE-V20 ${ARM}_s${SEED}"
