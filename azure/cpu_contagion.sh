#!/usr/bin/env bash
# Run the answer-contagion experiment on an Azure CPU VM (no GPU quota needed; credit-funded). Creates the VM, waits for
# cloud-init, syncs the repo, launches azure/cpu_run_contagion.sh in tmux, pulls results back every 10 min, deletes the VM at
# the end. Safe to re-run: the VM script skips finished models and contagion.py resumes per row.
#   bash azure/cpu_contagion.sh --size Standard_D32s_v5 --region eastus --tags "qwen7b llama8b mistral olmo" --yes
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SIZE="Standard_D64s_v7 Standard_D48s_v7 Standard_D64s_v6 Standard_E64s_v7 Standard_D64s_v5 Standard_F64s_v2 Standard_D32s_v7 Standard_D32s_v5 Standard_D16s_v7 Standard_D16s_v5"; REGION=eastus; TAGS="qwen7b llama8b mistral olmo"; YES=0; RG=rm-cpu; NAME=rm-cpu1; N=300
while [ $# -gt 0 ]; do case "$1" in --size) SIZE=$2; shift;; --region) REGION=$2; shift;; --tags) TAGS=$2; shift;; --n) N=$2; shift;;
  --name) NAME=$2; shift;; --rg) RG=$2; shift;; --yes) YES=1;; *) echo "unknown $1"; exit 1;; esac; shift; done
PRICE=$(curl -s "https://prices.azure.com/api/retail/prices?\$filter=armSkuName%20eq%20'${SIZE%% *}'%20and%20armRegionName%20eq%20'$REGION'%20and%20priceType%20eq%20'Consumption'" | python3 -c "import json,sys; it=[i for i in json.load(sys.stdin)['Items'] if 'Windows' not in i['productName'] and 'Spot' not in i['skuName'] and 'Low Priority' not in i['skuName']]; print(it[0]['retailPrice'] if it else 'unknown')")
echo "CPU VM $NAME sizes=[$SIZE] region=$REGION first-choice price=\$$PRICE/h  models: $TAGS  (est. 6-9 h per 7B model at N=$N)"
[ $YES = 1 ] || { echo "dry run; add --yes"; exit 0; }
az group create -n "$RG" -l "$REGION" -o none
OK=0
for SZ in $SIZE; do
  echo "trying $SZ ..."
  az vm create -g "$RG" -n "$NAME" -l "$REGION" --size "$SZ" --image Ubuntu2204 --admin-username azureuser --generate-ssh-keys \
    --os-disk-size-gb 64 --data-disk-sizes-gb 256 --storage-sku Premium_LRS --custom-data azure/cloud-init-cpu.yaml --public-ip-sku Standard --nsg-rule SSH \
    --tags project=reality-monitoring killswitch=yes -o table 2>&1 | grep -o -E 'SkuNotAvailable|QuotaExceeded|"powerState"[^,]*|VM running' | head -1 || true
  az vm show -g "$RG" -n "$NAME" -o none 2>/dev/null && { OK=1; echo "VM present ($SZ)"; break; }
done
[ $OK = 1 ] || { echo "no CPU size could be created in $REGION"; exit 1; }
IP=$(az vm show -d -g "$RG" -n "$NAME" --query publicIps -o tsv); echo "$IP" > azure/.last_ip_cpu; echo "VM up at $IP; waiting for cloud-init (llama.cpp build ~5 min)"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 azureuser@$IP"
for i in $(seq 1 80); do $SSH 'test -f /mnt/work/READY' 2>/dev/null && break; sleep 15; done; $SSH 'test -f /mnt/work/READY' || { echo "cloud-init did not finish"; exit 1; }
$SSH 'sudo chown -R azureuser:azureuser /mnt/work && mkdir -p /mnt/work/reality-monitoring'
rsync -az --delete --exclude .git --exclude results --exclude results_ladder --exclude checkpoints --exclude data --exclude docs --exclude figures --exclude .venv --exclude logs --exclude wandb \
  "$ROOT/" "azureuser@$IP:/mnt/work/reality-monitoring/" && echo "repo synced"
$SSH "cd /mnt/work/reality-monitoring && mkdir -p logs && tmux new -d -s ctg \"N=$N bash azure/cpu_run_contagion.sh $TAGS 2>&1 | tee logs/cpu_run.log\""
echo "launched in tmux 'ctg' on $IP (ssh azureuser@$IP; tmux attach -t ctg). Polling every 10 min; results land in results_contagion/."
while ! $SSH 'test -f /mnt/work/CONTAGION_DONE' 2>/dev/null; do
  rsync -az "azureuser@$IP:/mnt/work/reality-monitoring/results_contagion/" "$ROOT/results_contagion/" 2>/dev/null || true
  $SSH 'tail -1 /mnt/work/reality-monitoring/logs/cpu_run.log 2>/dev/null' 2>/dev/null | cut -c1-120; sleep 600; done
rsync -az "azureuser@$IP:/mnt/work/reality-monitoring/results_contagion/" "$ROOT/results_contagion/"
echo "finished; deleting VM (billing stops)"; az group delete -n "$RG" --yes --no-wait
