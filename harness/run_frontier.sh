#!/usr/bin/env bash
# Frontier replication from a laptop. Keys are read from ~/.rm_keys (chmod 600), lines like: export ANTHROPIC_API_KEY=...
#   bash harness/run_frontier.sh                 # dry-run cost estimate for both experiments on Fable 5.1 + GPT-6 Astra
#   bash harness/run_frontier.sh --yes           # run: v17 decomposition (150 q) + contagion pairwise/genuine/chain (150 q)
set -euo pipefail; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ -f ~/.rm_keys ] && source ~/.rm_keys; YES=${1:-}
: "${ANTHROPIC_API_KEY:?put 'export ANTHROPIC_API_KEY=...' in ~/.rm_keys}"; : "${OPENAI_API_KEY:?put 'export OPENAI_API_KEY=...' in ~/.rm_keys}"
DRY=""; [ "$YES" = "--yes" ] || DRY="--dry-run"
mkdir -p results_api results_contagion logs
python3 harness/run_cells_v17_api.py --provider anthropic --model claude-fable-5-1 --effort low --n-questions 150 --out results_api/fable51 $DRY
python3 harness/run_cells_v17_api.py --provider openai    --model gpt-6-astra      --effort low --n-questions 150 --out results_api/astra   $DRY
[ -n "$DRY" ] && { echo "dry run only; add --yes"; exit 0; }
PM=results_contagion/qwen7b/peer_msgs.jsonl
python3 harness/contagion.py --backend api --provider anthropic --model claude-fable-5-1 --effort low --n 150 --peers "same=claude-fable-5-1" --peer-msgs $PM --out results_contagion/fable51 --workers 8
python3 harness/contagion.py --backend api --provider openai    --model gpt-6-astra      --effort low --n 150 --peers "same=gpt-6-astra"      --peer-msgs $PM --out results_contagion/astra   --workers 8
python3 analysis/report_contagion.py results_contagion/fable51 results_contagion/astra
