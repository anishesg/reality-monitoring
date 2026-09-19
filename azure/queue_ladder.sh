#!/usr/bin/env bash
# "Queue" for Azure, which has no scheduler: retry VM creation until quota+capacity allow it, then run the ladder
# unattended in tmux on the VM, pull results back, and tear the VM down. Safe to leave running on a laptop or a
# tiny always-on box. Nothing is billed until a VM actually starts.
#   bash azure/queue_ladder.sh --backbones "olmo tulu" [--size Standard_NC24ads_A100_v4] [--region eastus2] [--spot] [--interval 600] --yes
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZE=Standard_NC24ads_A100_v4; REGION=eastus2; SPOT=""; INTERVAL=600; BACKBONES="olmo tulu"; YES=0
while [ $# -gt 0 ]; do case "$1" in
  --size) SIZE=$2; shift;; --region) REGION=$2; shift;; --spot) SPOT="--spot";; --interval) INTERVAL=$2; shift;;
  --backbones) BACKBONES=$2; shift;; --yes) YES=1;; *) echo "unknown arg $1"; exit 1;; esac; shift; done
[ $YES = 1 ] || { echo "dry run. Would retry '$SIZE' in $REGION every ${INTERVAL}s until a VM starts, then run the ladder for: $BACKBONES. Add --yes to arm."; exit 0; }
[ -f "$ROOT/.hftok" ] || { echo "note: no $ROOT/.hftok; fine for the ungated OLMo-2/Tulu-3 backbones, needed only for gated models"; touch "$ROOT/.hftok"; }
n=0
while true; do
  n=$((n+1)); echo "[$(date -u +%FT%TZ)] attempt $n: $SIZE in $REGION"
  if bash "$ROOT/azure/launch_vm.sh" --size "$SIZE" --region "$REGION" $SPOT --yes; then break; fi
  echo "  not available yet (quota or capacity); retrying in ${INTERVAL}s"; sleep "$INTERVAL"
done
IP=$(cat "$ROOT/azure/.last_ip")
echo "VM up at $IP; waiting for cloud-init"; for i in $(seq 1 60); do ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 azureuser@$IP 'test -f /mnt/work/.cloud-init-done' 2>/dev/null && break; sleep 30; done
bash "$ROOT/azure/sync.sh" up
ssh -o StrictHostKeyChecking=no azureuser@$IP "cd /mnt/work/reality-monitoring && cp .hftok .hftok 2>/dev/null; tmux new -d -s ladder 'source /mnt/work/venv/bin/activate; export HF_HOME=/mnt/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 USE_TF=0 HF_TOKEN=\$(cat .hftok); for T in $BACKBONES; do case \$T in olmo) B=allenai/OLMo-2-1124-7B-SFT;; tulu) B=allenai/Llama-3.1-Tulu-3-8B-SFT;; olmo13) B=allenai/OLMo-2-1124-13B-SFT;; olmo32) B=allenai/OLMo-2-0325-32B-SFT;; *) B=\$T;; esac; bash train/run_ladder.sh --reuse-a0 \$B \$T 2>&1 | tee -a /mnt/work/ladder_\$T.log; done; touch /mnt/work/LADDER_DONE'"
echo "ladder launched in tmux session 'ladder' on $IP. Polling for completion every 10 min; results sync back as arms finish."
while ! ssh -o StrictHostKeyChecking=no azureuser@$IP 'test -f /mnt/work/LADDER_DONE' 2>/dev/null; do
  bash "$ROOT/azure/sync.sh" down >/dev/null 2>&1 || true; sleep 600; done
bash "$ROOT/azure/sync.sh" down
echo "ladder finished; tearing down VM to stop billing"
bash "$ROOT/azure/teardown.sh" --yes
