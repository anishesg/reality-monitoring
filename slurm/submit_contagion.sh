#!/usr/bin/env bash
# Submit the answer-contagion experiment (prereg/PREREG_contagion.md) for the pre-registered open models on della.
#   PARTITION=gpu GRES=gpu:1 CONSTRAINT=a100-80g bash slurm/submit_contagion.sh            # C1: 5 open models, peers weak/same/strong
#   FIREWALL=checkpoints/qwen7b_firm/merged bash slurm/submit_contagion.sh qwen7b_fw      # C3: chain with a trained firewall agent
# Peer models (identity manipulation) are fixed for every model under test: weak = Qwen2.5-1.5B, strong = Qwen2.5-14B, same = the
# model under test. Prefetch all weights on a login node first: bash slurm/prefetch.sh (compute nodes are offline).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p logs results_contagion
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:1}"); [ -n "${CONSTRAINT:-}" ] && SB+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SB+=(--account="$ACCOUNT")
declare -A M=( [qwen7b]=Qwen/Qwen2.5-7B-Instruct [qwen14b]=Qwen/Qwen2.5-14B-Instruct [llama8b]=meta-llama/Llama-3.1-8B-Instruct
               [mistral]=mistralai/Mistral-7B-Instruct-v0.3 [olmo]=allenai/OLMo-2-1124-7B-Instruct )
WEAK=Qwen/Qwen2.5-1.5B-Instruct; STRONG=Qwen/Qwen2.5-14B-Instruct
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(qwen7b qwen14b llama8b mistral olmo)
for TAG in "${TAGS[@]}"; do
  BASE=${TAG%_fw}; MODEL=${M[$BASE]:?unknown tag $BASE}
  PEERS="weak=$WEAK,same=$MODEL,strong=$STRONG"
  EXP="ALL,MODEL=$MODEL,TAG=$TAG,PEERS=$PEERS,N=${N:-300},K=${K:-8}"
  [[ "$TAG" == *_fw ]] && { [ -n "${FIREWALL:-}" ] || { echo "set FIREWALL=<merged model dir> for $TAG"; exit 1; }; EXP="$EXP,FIREWALL=$FIREWALL,FIREWALL_POS=${FIREWALL_POS:-2},PEER_MSGS=results_contagion/$BASE/peer_msgs.jsonl"; }
  J=$(sbatch --parsable "${SB[@]}" --job-name=rm-ctg-$TAG --export="$EXP" slurm/contagion.sbatch | cut -d';' -f1)
  echo "$TAG -> job $J ($MODEL)"
done
echo "watch: squeue -u \$USER ; results: results_contagion/<tag>/summary.json ; report: python3 analysis/report_contagion.py results_contagion/*"
