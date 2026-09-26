#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=$HOME/reality-monitoring/logs/%x_%j.out
set -euo pipefail
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$WORK/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q torch transformers peft accelerate hf_transfer vllm
export HF_HUB_ENABLE_HF_TRANSFER=1
cd $HOME/reality-monitoring
python train_solution.py --model "$MODEL" --arm "$ARM" --out "results_v19/$TAG" --merged-dir "$WORK/merged"
python run_cells_v17.py --model "$WORK/merged" --claims claims_hard_eval.jsonl --n-questions 300 --out "results_v19/${TAG}_v17eval"
python run_cells_v18.py --model "$WORK/merged" --claims claims_hard_eval.jsonl --n-questions 250 --out "results_v19/${TAG}_v18eval"
echo "DONE-SOLUTION $TAG"
