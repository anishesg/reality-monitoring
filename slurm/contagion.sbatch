#!/bin/bash
#SBATCH --job-name=rm-contagion
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x_%j.out
# env: MODEL (HF id or merged dir), TAG (results_contagion/<TAG>), PEERS (name=model,...), FIREWALL (merged dir, optional),
#      FIREWALL_POS (default 2), N (default 300), K (default 8). Resumable; re-submitting the same job continues.
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
OUT="results_contagion/${TAG:?}"; mkdir -p "$OUT"
FW=""; [ -n "${FIREWALL:-}" ] && FW="--firewall-model $FIREWALL --firewall-pos ${FIREWALL_POS:-2}"
PM=""; [ -n "${PEER_MSGS:-}" ] && PM="--peer-msgs $PEER_MSGS"
python harness/contagion.py --backend vllm --model "${MODEL:?}" --out "$OUT" --n "${N:-300}" --k "${K:-8}" --peers "${PEERS:-}" $FW $PM &
wait $!
python analysis/report_contagion.py "$OUT"
echo "DONE-CONTAGION $TAG"
