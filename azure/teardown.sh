#!/usr/bin/env bash
# Deallocate AND delete the GPU VM, its disks, NIC and public IP (billing stops only on delete). Requires --yes.
set -euo pipefail
RG=rm-gpu; NAME=rm-gpu1; YES=0; ALL=0
while [ $# -gt 0 ]; do case "$1" in --yes) YES=1;; --name) NAME=$2; shift;; --rg) RG=$2; shift;; --all) ALL=1;; esac; shift; done
echo "Current VMs in $RG:"; az vm list -g "$RG" -d -o table 2>/dev/null || { echo "(no resource group $RG)"; exit 0; }
[ $YES = 1 ] || { echo "Dry run. Re-run with --yes to delete."; exit 0; }
if [ $ALL = 1 ]; then az group delete -n "$RG" --yes --no-wait; echo "deleting resource group $RG (all VMs, disks, IPs)"; exit 0; fi
az vm deallocate -g "$RG" -n "$NAME" -o none || true
az vm delete -g "$RG" -n "$NAME" --yes -o none
for d in $(az disk list -g "$RG" --query "[?managedBy==null].name" -o tsv); do az disk delete -g "$RG" -n "$d" --yes -o none; done
for r in nic publicip nsg; do for n in $(az network $r list -g "$RG" --query "[].name" -o tsv 2>/dev/null); do az network $r delete -g "$RG" -n "$n" -o none 2>/dev/null || true; done; done
echo "deleted $NAME and its disks/network in $RG. Remaining:"; az resource list -g "$RG" -o table
