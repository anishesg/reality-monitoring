#!/usr/bin/env bash
# Runs ON the VM after the first pass: re-run the CHAIN cells for every model with the v2 merged-turn protocol into
# results_contagion/<tag>_chain2. Waits for /mnt/work/CONTAGION_DONE (first pass) before starting. Touches CONTAGION_DONE2 at the end.
set -euo pipefail
cd /mnt/work/reality-monitoring; export HF_HOME=/mnt/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 LLAMA_KEY=local; source /mnt/work/venv/bin/activate
while [ ! -f /mnt/work/CONTAGION_DONE ]; do sleep 120; done
source <(sed -n '/^declare -A REPO/,/^stop()/p' azure/cpu_run_contagion.sh)   # reuse REPO/FILE/HFID tables, fetch(), serve(), stop()
SRV=/mnt/work/llama.cpp/build/bin/llama-server; N=${N:-300}; K=${K:-8}; SLOTS=${SLOTS:-16}; PORT=8080
for TAG in "$@"; do
  OUT=results_contagion/${TAG}_chain2; mkdir -p "$OUT"; [ -f "$OUT/meta.json" ] && { echo "$TAG chain2 done"; continue; }
  serve $TAG
  python harness/contagion.py --backend api --provider openai --api-base "http://127.0.0.1:$PORT/v1" --api-key-env LLAMA_KEY --workers $SLOTS \
    --n $N --k $K --cells chain --chain-turns merged --model "${HFID[$TAG]}" --out "$OUT" 2>&1 | tee -a "logs/ctg2_$TAG.log"
  stop; python analysis/report_contagion.py "$OUT" | tee "$OUT/report.txt"
done
touch /mnt/work/CONTAGION_DONE2; echo "CHAIN2 ALL DONE"
