#!/usr/bin/env bash
# Runs ON the VM after the chain re-run: GENUINE-answer pairwise cells for every model into results_contagion/<tag>_genuine.
# Waits for /mnt/work/CONTAGION_DONE2. Touches CONTAGION_DONE3 at the end.
set -euo pipefail
cd /mnt/work/reality-monitoring; export HF_HOME=/mnt/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 LLAMA_KEY=local; source /mnt/work/venv/bin/activate
while [ ! -f /mnt/work/CONTAGION_DONE2 ]; do sleep 120; done
source <(sed -n '/^declare -A REPO/,/^stop()/p' azure/cpu_run_contagion.sh)
SRV=/mnt/work/llama.cpp/build/bin/llama-server; N=${N:-300}; SLOTS=${SLOTS:-16}; PORT=8080
for TAG in "$@"; do
  OUT=results_contagion/${TAG}_genuine; mkdir -p "$OUT"; [ -f "$OUT/meta.json" ] && { echo "$TAG genuine done"; continue; }
  cp -n results_contagion/$TAG/peer_msgs.jsonl "$OUT/peer_msgs.jsonl" 2>/dev/null || true
  serve $TAG
  python harness/contagion.py --backend api --provider openai --api-base "http://127.0.0.1:$PORT/v1" --api-key-env LLAMA_KEY --workers $SLOTS \
    --n $N --cells genuine --model "${HFID[$TAG]}" --out "$OUT" 2>&1 | tee -a "logs/ctg3_$TAG.log"
  stop; python analysis/report_contagion.py "$OUT" | tee "$OUT/report.txt"
done
touch /mnt/work/CONTAGION_DONE3; echo "GENUINE ALL DONE"
