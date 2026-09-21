#!/usr/bin/env bash
# Resume-safe poller: pull results every 10 min; when the VM reports CONTAGION_DONE, pull once more and delete the VM.
set -u; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; IP=$(cat "$ROOT/azure/.last_ip_cpu"); RG=${RG:-rm-cpu}
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 azureuser@$IP"
while ! $SSH 'test -f /mnt/work/CONTAGION_DONE' 2>/dev/null; do rsync -az "azureuser@$IP:/mnt/work/reality-monitoring/results_contagion/" "$ROOT/results_contagion/" 2>/dev/null; sleep 600; done
rsync -az "azureuser@$IP:/mnt/work/reality-monitoring/results_contagion/" "$ROOT/results_contagion/"; echo "done; deleting $RG"; az group delete -n "$RG" --yes --no-wait
