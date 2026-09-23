#!/usr/bin/env bash
# Fable 5.1 frontier replication through the Anthropic Message Batches API (V, 2026-09-22). Sequential on purpose: if the
# prepaid balance runs out (the API refuses; no card on file), the earlier experiments are complete rather than all three partial.
# Order: three-cell identification (injected+elicited, 500 q) -> contagion (300 q, k=8) -> v17 decomposition (450 q, all cells).
set -uo pipefail; cd "$(dirname "$0")/.."; source ~/.rm_keys
export RM_BATCH_INFLIGHT=${RM_BATCH_INFLIGHT:-4}
M=claude-fable-5-1; E=low
RM_SPEND_TAG=ident_fable python3 experiments/run_identification_api.py --provider anthropic --model $M --effort $E --claims claims3.jsonl --n 500 --out results_ident_api/fable51 --batch || { echo "STOP after identification (exit $?)"; exit 1; }
mkdir -p results_contagion/fable51; [ -s results_contagion/fable51/peer_msgs.jsonl ] || cp results_contagion/qwen7b/peer_msgs.jsonl results_contagion/fable51/peer_msgs.jsonl
RM_SPEND_TAG=ctg_fable python3 harness/contagion.py --backend batch --provider anthropic --model $M --effort $E --n 300 --k 8 --peers fable=$M --out results_contagion/fable51 || { echo "STOP after contagion (exit $?)"; exit 1; }
RM_SPEND_TAG=v17_fable python3 harness/run_cells_v17_api.py --provider anthropic --model $M --effort $E --n-questions 450 --cells all --out results_api/fable51 --batch || { echo "STOP after decomposition (exit $?)"; exit 1; }
echo "DONE-FABLE-ALL"
