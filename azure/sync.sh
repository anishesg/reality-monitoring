#!/usr/bin/env bash
# rsync the git-layout repo up (code, claim banks, prereg) and results/checkpoint summaries down.
#   bash azure/sync.sh up|down [ip]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; IP="${2:-$(cat "$ROOT/azure/.last_ip")}"; DEST=/mnt/work/reality-monitoring
case "${1:?up|down}" in
  up)   rsync -az --delete --exclude .git --exclude results_ladder --exclude checkpoints --exclude checkpoints_meta --exclude data --exclude docs --exclude figures --exclude .venv --exclude logs \
              "$ROOT/" "azureuser@$IP:$DEST/" && echo "synced -> $IP:$DEST (harness/, analysis/, results/, train/, slurm/, prereg/, azure/)";;
  down) rsync -az "azureuser@$IP:$DEST/results_ladder/" "$ROOT/results_ladder/" && rsync -az --include '*/' --include 'train_meta.json' --include 'provenance.json' --exclude '*' "azureuser@$IP:$DEST/checkpoints/" "$ROOT/checkpoints_meta/" && echo "pulled results_ladder/ and checkpoint metadata";;
esac
