#!/usr/bin/env bash
# Elicited three-cell identification on the six identification models (closes the "inject rather than elicit" limitation).
#   CLAIMS=/path/to/claims3.jsonl PARTITION=<p> bash slurm/submit_ident_elicited.sh        (build claims3.jsonl first: python experiments/claimbank_3cell.py on the login node)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p logs results_ident_elicited
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:1}"); [ -n "${CONSTRAINT:-}" ] && SB+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SB+=(--account="$ACCOUNT")
declare -A M=( [e_qwen7b]=Qwen/Qwen2.5-7B-Instruct [e_qwen14b]=Qwen/Qwen2.5-14B-Instruct [e_llama8b]=meta-llama/Llama-3.1-8B-Instruct [e_olmo]=allenai/OLMo-2-1124-7B-Instruct [e_qwen3_8b]=Qwen/Qwen3-8B [e_r1_7b]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B )
for TAG in "${!M[@]}"; do
  [ -f "results_ident_elicited/$TAG/summary.json" ] && { echo "$TAG done"; continue; }
  J=$(sbatch --parsable "${SB[@]}" --job-name=rm-ident-$TAG --export="ALL,MODEL=${M[$TAG]},TAG=$TAG,CLAIMS=${CLAIMS:-claims3.jsonl},N=${N:-500}" slurm/ident_elicited.sbatch | cut -d';' -f1)
  echo "$TAG -> job $J"
done
