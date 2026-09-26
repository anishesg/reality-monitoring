#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/reality-monitoring/sprint/logs/%x_%j.out
set -eo pipefail
SCR=/n/fs/scratch/$USER; WORK=$SCR/tmp/${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="$SCR/hf" TMPDIR="$WORK" HF_HUB_ENABLE_HF_TRANSFER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_TOKEN=$(cat $HOME/reality-monitoring/.hftok 2>/dev/null || true)
source $SCRATCH/venv_sprint/bin/activate
cd $HOME/reality-monitoring/sprint
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CMD: $CMD"
eval "$CMD"
echo "DONE-JOB $SLURM_JOB_NAME"
