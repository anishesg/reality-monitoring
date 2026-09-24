#!/usr/bin/env bash
# ONE-TIME SETUP on a LOGIN node (needs internet), then submits the whole core queue. Idempotent: re-run after any failure.
#   bash RUN_ME.sh                         # della defaults: CONSTRAINT=gpu80 (A100 80GB), HF cache on /scratch/gpfs/$USER/hf
#   GPU=a6000 CONSTRAINT=<a6000 nodes> bash RUN_ME.sh     # 48 GB cards (A6000/L40/A40): GRPO uses 2 GPUs per job, longer time caps
#   CONSTRAINT=... PARTITION=... ACCOUNT=... bash RUN_ME.sh     # other clusters
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HOME="${HF_HOME:-/scratch/gpfs/$USER/hf}"; mkdir -p "$HF_HOME" logs
export GPU="${GPU:-a100}"; if [ "$GPU" = a100 ]; then export CONSTRAINT="${CONSTRAINT-gpu80}"; else export CONSTRAINT="${CONSTRAINT-}"; fi
echo "== root=$(pwd)  GPU=$GPU  HF_HOME=$HF_HOME  CONSTRAINT=${CONSTRAINT:-none} PARTITION=${PARTITION:-none} ACCOUNT=${ACCOUNT:-none}"
# 1. python environment (once, ~10 min)
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; { module load anaconda3/2024.02 || module load anaconda3 || true; } 2>/dev/null
if [ ! -d .venv ]; then python -m venv .venv; fi
source .venv/bin/activate
python -c "import trl, vllm, peft, lm_eval" 2>/dev/null || pip install -q -r train/requirements.txt
python -c "import trl, vllm, peft, lm_eval, torch; print('env ok: torch', torch.__version__, 'cuda', torch.version.cuda)"
# 2. models + datasets + training data (once, ~20 min; ~40 GB on scratch). No HF token needed: every model here is ungated.
bash slurm/prefetch.sh olmo tulu
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "allenai/OLMo-2-1124-7B-Instruct"):   # firewall-chain peers
    print("prefetch", m, "->", snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]))
PY
# 3. submit everything with dependencies
bash slurm/submit_core.sh
echo; echo "== submitted. Progress: bash STATUS.sh    Results to send back: bash COLLECT_RESULTS.sh"
