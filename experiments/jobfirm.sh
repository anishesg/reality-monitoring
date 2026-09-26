#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=$HOME/reality-monitoring/logs/%x_%j.out
set -euo pipefail
SCR=/n/fs/scratch/$USER; WORK=$SCR/tmp/${SLURM_JOB_ID}; mkdir -p "$WORK"; trap 'rm -rf $WORK' EXIT
export HF_HOME="$SCR/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK"
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q torch transformers peft accelerate hf_transfer vllm
cd $HOME/reality-monitoring
python train_firmness.py --arm "$ARM" --seed "$SEED" --out "results_firm/${ARM}_s${SEED}"
echo "DONE-FIRM-JOB ${ARM}_s${SEED}"
