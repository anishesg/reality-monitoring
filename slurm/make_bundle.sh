#!/usr/bin/env bash
# Build the standalone cluster bundle (zip) from the repo: code + claim banks + pre-registrations + the two reusable A0 evals + firewall peers.
#   bash slurm/make_bundle.sh [outdir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUTDIR="${1:-$HOME/Desktop}"; B="$OUTDIR/rm_ladder_bundle"; rm -rf "$B"; mkdir -p "$B"
cd "$ROOT"
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'run_fable*.sh' --exclude 'run_frontier.sh' harness/ "$B/harness/"
rsync -a --exclude '__pycache__' --include '*/' --include '*.py' --exclude '*' analysis/ "$B/analysis/"
rsync -a --exclude '__pycache__' --exclude 'results_server.py' train/ "$B/train/"
rsync -a --exclude 'make_bundle.sh' slurm/ "$B/slurm/"
rsync -a prereg/ "$B/prereg/"
mkdir -p "$B/results/results_v17_cells" "$B/results_contagion/olmo" "$B/experiments"
rsync -a results/results_v17_cells/c_olmo_sft results/results_v17_cells/c_tulu_sft "$B/results/results_v17_cells/"
cp results_contagion/olmo/peer_msgs.jsonl "$B/results_contagion/olmo/"
mkdir -p "$B/data/olmo"; cp data/olmo/generic.jsonl data/olmo/revision.jsonl "$B/data/olmo/"   # prebuilt OLMo training data (tulu is built by prefetch.sh); cp experiments/report_ladder.py "$B/experiments/"
cp bundle/README.md bundle/RUN_ME.sh bundle/STATUS.sh bundle/COLLECT_RESULTS.sh "$B/"
# checks: no secrets, scripts parse, python compiles, referenced files exist
! grep -rIl "sk-ant\|sk-proj\|hf_[A-Za-z0-9]\{20\}" "$B" >/dev/null || { echo "FATAL: secret-looking string in bundle"; grep -rIl "sk-ant\|sk-proj" "$B"; exit 1; }
for f in $(find "$B" -name '*.sh' -o -name '*.sbatch'); do bash -n "$f" || { echo "FATAL: $f does not parse"; exit 1; }; done
python3 -m compileall -q "$B" >/dev/null || { echo "FATAL: python compile error"; exit 1; }
for f in harness/claims_hard.jsonl harness/claims.jsonl harness/run_cells_v17.py harness/contagion.py analysis/analyze_v17.py analysis/report_contagion.py train/build_data.py train/dpo.py train/grpo.py train/eval_arm.py train/reward.py slurm/_common.sh slurm/submit_core.sh slurm/train_dpo.sbatch slurm/train_grpo.sbatch slurm/eval_arm.sbatch slurm/contagion.sbatch slurm/prefetch.sh results/results_v17_cells/c_olmo_sft/cells.jsonl results/results_v17_cells/c_tulu_sft/cells.jsonl results_contagion/olmo/peer_msgs.jsonl prereg/PREREG_causal_ladder.md; do
  [ -f "$B/$f" ] || { echo "FATAL: missing $f"; exit 1; }; done
find "$B" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
( cd "$OUTDIR" && rm -f rm_ladder_bundle.zip && zip -qr rm_ladder_bundle.zip rm_ladder_bundle )
echo "bundle: $OUTDIR/rm_ladder_bundle.zip ($(du -sh "$OUTDIR/rm_ladder_bundle.zip" | cut -f1)); files: $(find "$B" -type f | wc -l | tr -d ' ')"
