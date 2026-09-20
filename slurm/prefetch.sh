#!/usr/bin/env bash
# Run ONCE on a login node (internet): download every model and dataset the ladder needs into HF_HOME on scratch, and build the
# injected/generic training data, so compute nodes can run fully offline. Idempotent.
#   HF_HOME=/scratch/gpfs/$USER/hf bash slurm/prefetch.sh [olmo tulu olmo13 olmo32]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export HF_HOME="${HF_HOME:-/scratch/gpfs/$USER/hf}"; mkdir -p "$HF_HOME"; export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f .hftok ] && export HF_TOKEN=$(cat .hftok)
[ -d .venv ] && source .venv/bin/activate
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(olmo tulu)
declare -A HF=([olmo]=allenai/OLMo-2-1124-7B-SFT [tulu]=allenai/Llama-3.1-Tulu-3-8B-SFT [olmo13]=allenai/OLMo-2-1124-13B-SFT [olmo32]=allenai/OLMo-2-0325-32B-SFT)
for T in "${TAGS[@]}"; do
  python - "${HF[$T]}" <<'PY'
import sys; from huggingface_hub import snapshot_download
print("prefetch", sys.argv[1], "->", snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]))
PY
  mkdir -p data/$T
  [ -f data/$T/revision.jsonl ] || python train/build_data.py --mode injected --out data/$T/revision.jsonl --per-cell 600
  [ -f data/$T/generic.jsonl ]  || python train/build_data.py --mode generic  --out data/$T/generic.jsonl  --n-generic 10000
done
# capability suite datasets (lm-eval downloads on first use): warm the cache here, on the login node
python - <<'PY'
from datasets import load_dataset
for name, cfg in (("cais/mmlu", "all"), ("openai/gsm8k", "main"), ("google/IFEval", None)):
    try: load_dataset(name, cfg) if cfg else load_dataset(name); print("cached", name)
    except Exception as e: print("WARN could not cache", name, str(e)[:120])
PY
echo "prefetch done: HF_HOME=$HF_HOME"
