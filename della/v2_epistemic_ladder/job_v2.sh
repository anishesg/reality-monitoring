#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/gpfs/%u/reality-monitoring/logs/%x_%j.out
# NOTE: --gres / --time / --mem / --partition are overridden at submit time.
set -euo pipefail
PROJ=${PROJ:-/scratch/gpfs/$USER/reality-monitoring}
cd "$PROJ"
module load anaconda3/2024.10 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$PROJ/venv/bin/activate"
export HF_HOME="$PROJ/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python run_experiment_v2.py --model "$MODEL" --tp "${GPUS:-1}" \
  --n-questions "${NQ:-1500}" ${LITE:+--lite} --own-claims "${OWN:-300}" \
  --claims claims.jsonl --out "results/$TAG"
echo "DONE $TAG"
