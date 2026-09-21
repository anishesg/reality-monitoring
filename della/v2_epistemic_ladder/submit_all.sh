#!/bin/bash
# Full ladder. Format: model|tag|gpus|time|mem
set -euo pipefail
PROJ=${PROJ:-/scratch/gpfs/$USER/reality-monitoring}
cd "$PROJ"
source ./secrets.env || true

JOBS=(
  "Qwen/Qwen2.5-0.5B-Instruct|qwen_0p5b|1|03:00:00|64G"
  "Qwen/Qwen2.5-1.5B-Instruct|qwen_1p5b|1|03:00:00|64G"
  "Qwen/Qwen2.5-3B-Instruct|qwen_3b|1|04:00:00|64G"
  "Qwen/Qwen2.5-7B-Instruct|qwen_7b|1|05:00:00|64G"
  "Qwen/Qwen2.5-14B-Instruct|qwen_14b|1|06:00:00|80G"
  "allenai/OLMo-2-1124-7B-Instruct|olmo2_7b|1|05:00:00|64G"
  "allenai/OLMo-2-1124-13B-Instruct|olmo2_13b|1|06:00:00|80G"
  "mistralai/Mistral-7B-Instruct-v0.3|mistral_7b|1|05:00:00|64G"
)
[ "${SKIP_BIG:-0}" != "1" ] && JOBS+=(
  "Qwen/Qwen2.5-32B-Instruct|qwen_32b|1|09:00:00|100G"
  "Qwen/Qwen2.5-72B-Instruct|qwen_72b|2|12:00:00|160G"
)
[ -n "${HF_TOKEN:-}" ] && JOBS+=(
  "meta-llama/Llama-3.1-8B-Instruct|llama31_8b|1|05:00:00|64G"
  "meta-llama/Llama-3.1-70B-Instruct|llama31_70b|2|12:00:00|160G"
  "google/gemma-2-9b-it|gemma2_9b|1|05:00:00|64G"
  "google/gemma-2-27b-it|gemma2_27b|1|08:00:00|100G"
)

for spec in "${JOBS[@]}"; do
  IFS="|" read -r model tag gpus time mem <<< "$spec"
  sbatch --gres=gpu:"$gpus" --time="$time" --mem="$mem" --job-name="rm_$tag" \
    --export=ALL,PROJ="$PROJ",MODEL="$model",TAG="$tag",GPUS="$gpus",NQ=1500,OWN=300 job_v2.sh
done
squeue -u "$USER" -o "%.10i %.14j %.2t %.8M %R"
