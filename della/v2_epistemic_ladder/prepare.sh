#!/bin/bash
# RUN ON THE DELLA *LOGIN* NODE (compute nodes have NO internet).
# Sets up env, builds claim bank, pre-downloads all models to /scratch/gpfs.
set -euo pipefail
export PROJ=${PROJ:-/scratch/gpfs/$USER/reality-monitoring}
mkdir -p "$PROJ"/{results,logs}
cp -f ./*.py ./*.sh ./secrets.env "$PROJ"/ 2>/dev/null || true
cd "$PROJ"
source ./secrets.env || true

module load anaconda3/2024.10 2>/dev/null || module load anaconda3 2>/dev/null || true
python3 -m venv "$PROJ/venv"
source "$PROJ/venv/bin/activate"
pip install -q --upgrade pip
pip install -q vllm datasets hf_transfer "huggingface_hub[cli]"

export HF_HOME="$PROJ/hf" HF_HUB_ENABLE_HF_TRANSFER=1
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "== building claim bank"
python claimbank_v2.py

CORE=(
  Qwen/Qwen2.5-0.5B-Instruct
  Qwen/Qwen2.5-1.5B-Instruct
  Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  Qwen/Qwen2.5-14B-Instruct
  allenai/OLMo-2-1124-7B-Instruct
  allenai/OLMo-2-1124-13B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
BIG=( Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-72B-Instruct )
GATED=( meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct google/gemma-2-9b-it google/gemma-2-27b-it )

DL=("${CORE[@]}")
[ "${SKIP_BIG:-0}" != "1" ] && DL+=("${BIG[@]}")
[ -n "${HF_TOKEN:-}" ] && DL+=("${GATED[@]}")

for m in "${DL[@]}"; do
  echo "== downloading $m"
  huggingface-cli download "$m" --exclude "original/*" --exclude "*.pth" >/dev/null || \
    echo "WARN: download failed for $m (gated without access? skip it in submit_all.sh)"
done
echo "== prepare done. Disk usage:"; du -sh "$PROJ/hf" 2>/dev/null | tail -1
echo "Next: sbatch submit via ./submit_smoke.sh, then ./submit_all.sh"
