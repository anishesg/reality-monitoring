#!/bin/bash
# Smoke test: 0.5B model, 60 questions, lite factorial (~2.2k generations, <15 min).
set -euo pipefail
PROJ=${PROJ:-/scratch/gpfs/$USER/reality-monitoring}
cd "$PROJ"
sbatch --gres=gpu:1 --time=00:59:00 --mem=48G --job-name=rm_smoke \
  --export=ALL,PROJ="$PROJ",MODEL=Qwen/Qwen2.5-0.5B-Instruct,TAG=smoke,NQ=60,LITE=1,OWN=0 job_v2.sh
echo "Smoke submitted. Check: results/smoke/summary.json (want unparsed_frac < 0.10)"
