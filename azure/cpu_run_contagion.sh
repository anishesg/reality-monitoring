#!/usr/bin/env bash
# Runs ON the CPU VM (inside tmux). Serves each Q8 GGUF with llama-server and drives harness/contagion.py through its API backend.
#   bash azure/cpu_run_contagion.sh qwen7b llama8b mistral olmo [qwen14b]      (env: N=300 K=8 SLOTS=16)
# Peer messages for weak (1.5B) and strong (14B) are generated once into results_contagion/_peers/ and reused for every model.
set -euo pipefail
cd /mnt/work/reality-monitoring; export HF_HOME=/mnt/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 LLAMA_KEY=local; source /mnt/work/venv/bin/activate
SRV=/mnt/work/llama.cpp/build/bin/llama-server; N=${N:-300}; K=${K:-8}; SLOTS=${SLOTS:-16}; PORT=8080
declare -A REPO=( [qwen15]=Qwen/Qwen2.5-1.5B-Instruct-GGUF [qwen7b]=Qwen/Qwen2.5-7B-Instruct-GGUF [qwen14b]=Qwen/Qwen2.5-14B-Instruct-GGUF
                  [olmo]=allenai/OLMo-2-1124-7B-Instruct-GGUF [llama8b]=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF [mistral]=bartowski/Mistral-7B-Instruct-v0.3-GGUF )
declare -A FILE=( [qwen15]=qwen2.5-1.5b-instruct-q8_0.gguf [qwen7b]=qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf [qwen14b]=qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf
                  [olmo]=olmo-2-1124-7B-instruct-Q8_0.gguf [llama8b]=Meta-Llama-3.1-8B-Instruct-Q8_0.gguf [mistral]=Mistral-7B-Instruct-v0.3-Q8_0.gguf )
declare -A HFID=( [qwen15]=Qwen/Qwen2.5-1.5B-Instruct [qwen7b]=Qwen/Qwen2.5-7B-Instruct [qwen14b]=Qwen/Qwen2.5-14B-Instruct
                  [olmo]=allenai/OLMo-2-1124-7B-Instruct [llama8b]=meta-llama/Llama-3.1-8B-Instruct [mistral]=mistralai/Mistral-7B-Instruct-v0.3 )
fetch() { python - "${REPO[$1]}" <<'PY'
import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*q8_0*.gguf", "*Q8_0*.gguf"]))
PY
}
serve() {  # serve <tag>: start llama-server on $PORT, wait for health
  local dir; dir=$(fetch "$1" | tail -1); local f="$dir/${FILE[$1]}"; [ -f "$f" ] || f=$(ls "$dir"/*.gguf | head -1)
  local PC; PC=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s+0}'); [ "$PC" -gt 0 ] || PC=$(nproc)
  "$SRV" -m "$f" --host 127.0.0.1 --port $PORT -np $SLOTS -c $((2048 * SLOTS)) -t "$PC" -tb "$PC" --numa distribute --temp 0 --no-warmup > "logs/llama_$1.log" 2>&1 &
  echo $! > /tmp/srv.pid
  for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0; sleep 5; done
  echo "server for $1 did not come up; see logs/llama_$1.log"; return 1
}
stop() { [ -f /tmp/srv.pid ] && { kill "$(cat /tmp/srv.pid)" 2>/dev/null || true; rm -f /tmp/srv.pid; sleep 3; }; }
ctg() { python harness/contagion.py --backend api --provider openai --api-base "http://127.0.0.1:$PORT/v1" --api-key-env LLAMA_KEY --workers $SLOTS --n $N --k $K "$@"; }
mkdir -p logs results_contagion/_peers
# shared peers (weak, strong) once
for P in qwen15 qwen14b; do
  NAME=weak; [ $P = qwen14b ] && NAME=strong
  grep -q "\"peer\": \"$NAME\"" results_contagion/_peers/peer_msgs.jsonl 2>/dev/null && { echo "peers $NAME done"; continue; }
  serve $P; ctg --phase peers --model "${HFID[$P]}" --out results_contagion/_peers --peers "$NAME=${HFID[$P]}"; stop
done
for TAG in "$@"; do
  OUT=results_contagion/$TAG; mkdir -p "$OUT"
  [ -f "$OUT/meta.json" ] && { echo "$TAG done"; continue; }
  serve $TAG
  [ -f "$OUT/peer_msgs.jsonl" ] || cp results_contagion/_peers/peer_msgs.jsonl "$OUT/peer_msgs.jsonl"
  ctg --phase peers --model "${HFID[$TAG]}" --out "$OUT" --peers "same=${HFID[$TAG]}"
  ctg --phase main  --model "${HFID[$TAG]}" --out "$OUT" 2>&1 | tee -a "logs/ctg_$TAG.log"
  python - "$OUT" <<'PY'
import json,sys,os; m=json.load(open(os.path.join(sys.argv[1],"meta.json"))); m["backend"]="llama.cpp Q8_0 (CPU)"; json.dump(m,open(os.path.join(sys.argv[1],"meta.json"),"w"),indent=1)
PY
  stop; python analysis/report_contagion.py "$OUT" | tee "$OUT/report.txt"
done
touch /mnt/work/CONTAGION_DONE; echo "ALL DONE"
