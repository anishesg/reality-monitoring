#!/usr/bin/env bash
# Runs ON the CPU VM (inside tmux). Elicited three-cell identification on the six open models with llama-server (Q8_0, CPU),
# driving experiments/run_identification_elicited.py --backend api. Same bank/templates/grading as the GPU (vLLM) path.
#   bash azure/cpu_run_ident_elicited.sh qwen7b olmo llama8b qwen14b qwen3_8b r1_7b      (env: N=500 SLOTS=16)
set -euo pipefail
cd /mnt/work/reality-monitoring; export HF_HOME=/mnt/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 LLAMA_KEY=local; source /mnt/work/venv/bin/activate
SRV=/mnt/work/llama.cpp/build/bin/llama-server; N=${N:-500}; SLOTS=${SLOTS:-16}; PORT=8080
declare -A REPO=( [qwen7b]=Qwen/Qwen2.5-7B-Instruct-GGUF [qwen14b]=Qwen/Qwen2.5-14B-Instruct-GGUF [olmo]=allenai/OLMo-2-1124-7B-Instruct-GGUF
                  [llama8b]=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF [qwen3_8b]=Qwen/Qwen3-8B-GGUF [r1_7b]=bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF )
declare -A FILE=( [qwen7b]=qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf [qwen14b]=qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf
                  [olmo]=olmo-2-1124-7B-instruct-Q8_0.gguf [llama8b]=Meta-Llama-3.1-8B-Instruct-Q8_0.gguf [qwen3_8b]=Qwen3-8B-Q8_0.gguf [r1_7b]=DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf )
declare -A HFID=( [qwen7b]=Qwen/Qwen2.5-7B-Instruct [qwen14b]=Qwen/Qwen2.5-14B-Instruct [olmo]=allenai/OLMo-2-1124-7B-Instruct
                  [llama8b]=meta-llama/Llama-3.1-8B-Instruct [qwen3_8b]=Qwen/Qwen3-8B [r1_7b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B )
fetch() { python - "${REPO[$1]}" <<'PY'
import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*q8_0*.gguf", "*Q8_0*.gguf"]))
PY
}
serve() {
  local dir; dir=$(fetch "$1" | tail -1); local f="$dir/${FILE[$1]}"; [ -f "$f" ] || f=$(ls "$dir"/*.gguf | head -1)
  local PC; PC=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s+0}'); [ "$PC" -gt 0 ] || PC=$(nproc)
  local EXTRA=""; [ "$1" = qwen3_8b ] && EXTRA="--reasoning-budget 0"   # Qwen3 non-thinking mode, matching the vLLM run's 220-token budget
  "$SRV" -m "$f" --host 127.0.0.1 --port $PORT -np $SLOTS -c $((2048 * SLOTS)) -t "$PC" -tb "$PC" --numa distribute --temp 0 --no-warmup $EXTRA > "logs/llama_e_$1.log" 2>&1 &
  echo $! > /tmp/srv.pid
  for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0; sleep 5; done
  echo "server for $1 did not come up; see logs/llama_e_$1.log"; return 1
}
stop() { [ -f /tmp/srv.pid ] && { kill "$(cat /tmp/srv.pid)" 2>/dev/null || true; rm -f /tmp/srv.pid; sleep 3; }; }
mkdir -p logs results_ident_elicited
for TAG in "$@"; do
  OUT=results_ident_elicited/e_$TAG
  [ -f "$OUT/summary.json" ] && { echo "$TAG done"; continue; }
  serve $TAG || { echo "SKIP $TAG (server failed)"; stop; continue; }
  python experiments/run_identification_elicited.py --backend api --api-base "http://127.0.0.1:$PORT/v1" --api-key-env LLAMA_KEY --workers $SLOTS \
    --model "${HFID[$TAG]}" --n $N --out "$OUT" 2>&1 | tee "logs/e_$TAG.log"
  stop
done
touch /mnt/work/ELICITED_DONE; echo "ALL DONE"
