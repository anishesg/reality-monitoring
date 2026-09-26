#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=$HOME/reality-monitoring/logs/%x_%j.out
set -euo pipefail
SCR=/n/fs/scratch/$USER
WORK=$SCR/tmp/${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$SCR/hf" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK" HF_HUB_ENABLE_HF_TRANSFER=1
source /etc/profile.d/modules.sh; module load anaconda3/2024.02
python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"
pip install -q --upgrade pip; pip install -q vllm hf_transfer
cd $HOME/reality-monitoring
python forced_conf2.py --model "$MODEL" --out "results_forced2/$TAG"
echo "DONE-FORCED-JOB $TAG"
