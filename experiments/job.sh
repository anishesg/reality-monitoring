#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=$HOME/reality-monitoring/logs/%x_%j.out
set -euo pipefail
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL tag=$TAG nq=${NQ:-500}"
WORK=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh
module load anaconda3/2024.02
python -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install -q --upgrade pip
pip install -q vllm hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd $HOME/reality-monitoring
python run_experiment.py --model "$MODEL" --claims claims.jsonl \
  --n-questions "${NQ:-500}" --out "results/$TAG"
echo "DONE $TAG"
