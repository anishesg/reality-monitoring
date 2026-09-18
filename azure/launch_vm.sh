#!/usr/bin/env bash
# Launch a GPU VM for the ladder. Refuses to run without --yes; prints the live hourly price first.
#   bash azure/launch_vm.sh [--size Standard_NC24ads_A100_v4|Standard_NC96ads_A100_v4] [--spot] [--region eastus2] --yes
set -euo pipefail
SIZE=Standard_NC24ads_A100_v4; REGION=eastus2; SPOT=0; YES=0; RG=rm-gpu; NAME=rm-gpu1; DISK_GB=512
while [ $# -gt 0 ]; do case "$1" in --size) SIZE=$2; shift;; --region) REGION=$2; shift;; --spot) SPOT=1;; --yes) YES=1;; --name) NAME=$2; shift;; --rg) RG=$2; shift;; *) echo "unknown $1"; exit 1;; esac; shift; done
PT=Consumption; [ $SPOT = 1 ] && PT=Spot
PRICE=$(curl -s "https://prices.azure.com/api/retail/prices?\$filter=armSkuName%20eq%20'$SIZE'%20and%20armRegionName%20eq%20'$REGION'%20and%20priceType%20eq%20'$PT'" | python3 -c "import json,sys; it=[i for i in json.load(sys.stdin)['Items'] if 'Windows' not in i['productName'] and ('Spot' in i['skuName'])==($SPOT==1)]; print(it[0]['retailPrice'] if it else 'unknown')")
echo "VM $NAME  size=$SIZE  region=$REGION  spot=$SPOT  price=\$$PRICE/hour  (+ ~\$0.10/h for the ${DISK_GB}GB premium disk)"
echo "Quota check:"; az vm list-usage --location "$REGION" -o json | python3 -c "
import json,sys; fam={'Standard_NC24ads_A100_v4':'standardNCADSA100v4Family','Standard_NC48ads_A100_v4':'standardNCADSA100v4Family','Standard_NC96ads_A100_v4':'standardNCADSA100v4Family','Standard_NC40ads_H100_v5':'standardNCadsH100v5Family'}.get('$SIZE')
need={'Standard_NC24ads_A100_v4':24,'Standard_NC48ads_A100_v4':48,'Standard_NC96ads_A100_v4':96,'Standard_NC40ads_H100_v5':40}.get('$SIZE',0)
for u in json.load(sys.stdin):
    if u['name']['value'] in (fam,'lowPriorityCores'): print('  ',u['name']['value'],'limit=',u['limit'],'used=',u['currentValue'], '(need',need,')')"
[ $YES = 1 ] || { echo "Dry run. Re-run with --yes to create the VM (this starts billing)."; exit 0; }
az group create -n "$RG" -l "$REGION" -o none
EXTRA=(); [ $SPOT = 1 ] && EXTRA=(--priority Spot --eviction-policy Deallocate --max-price -1)
az vm create -g "$RG" -n "$NAME" -l "$REGION" --size "$SIZE" --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --admin-username azureuser --generate-ssh-keys --os-disk-size-gb 128 --storage-sku Premium_LRS \
  --data-disk-sizes-gb "$DISK_GB" --custom-data "$(dirname "$0")/cloud-init.yaml" --public-ip-sku Standard --nsg-rule SSH "${EXTRA[@]}" \
  --tags project=reality-monitoring killswitch=yes -o table
IP=$(az vm show -d -g "$RG" -n "$NAME" --query publicIps -o tsv); echo "$IP" > "$(dirname "$0")/.last_ip"
echo "ssh azureuser@$IP   (cloud-init takes ~10 min; wait for /mnt/work/READY). Billing started at \$$PRICE/h. Tear down with azure/teardown.sh --yes"
