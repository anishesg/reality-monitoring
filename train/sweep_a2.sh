#!/usr/bin/env bash
# Pre-registered A2 (revision-DPO) sensitivity sweep on the DEV split (hard qids 800-899), OLMo-2-7B-SFT only.
# 5 runs: beta in {0.05, 0.1, 0.2} at lr 5e-6, plus lr in {2e-6, 1e-5} at beta 0.1. Selection rule: max min(retain_correct, accept_valid_correction) on DEV.
# The selected setting is then used for A2 on TEST and reported alongside all five dev rows (appendix). ~8 GPU-hours on one A100.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; BB="${1:-allenai/OLMo-2-1124-7B-SFT}"; TAG="${2:-olmo}"
PY="${PYTHON:-python3}"; export USE_TF=0 TRANSFORMERS_NO_TF=1
DATA="$ROOT/data/$TAG"; CK="$ROOT/checkpoints/$TAG/sweep"; RES="$ROOT/results_ladder/$TAG/sweep"; mkdir -p "$DATA" "$CK" "$RES"
[ -f "$DATA/revision.jsonl" ] || $PY "$ROOT/train/build_data.py" --mode injected --out "$DATA/revision.jsonl" --per-cell 600
for cfg in "0.05 5e-6" "0.1 5e-6" "0.2 5e-6" "0.1 2e-6" "0.1 1e-5"; do
  set -- $cfg; B=$1; LR=$2; NAME="A2_b${B}_lr${LR}"
  [ -f "$RES/dev/$NAME/summary.json" ] && { echo "$NAME done"; continue; }
  $PY "$ROOT/train/dpo.py" --model "$BB" --data "$DATA/revision.jsonl" --out "$CK/$NAME" --beta "$B" --lr "$LR" --merge ${TRAIN_EXTRA:-}
  $PY "$ROOT/train/eval_arm.py" --model "$CK/$NAME/merged" --arm "$NAME" --results-root "$RES" --split dev
done
$PY - "$RES/dev" <<'PY'
import json, glob, os, sys
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "A2_*", "summary.json")))]
for r in rows: r["_sel"] = min(r["retain_correct"] or 0, r["accept_valid_correction"] or 0)
rows.sort(key=lambda r: -r["_sel"])
print(f"{'setting':22s} {'retainC':>8} {'acceptFix':>9} {'pressAb':>8} {'min':>6}")
for r in rows: print(f"{r['arm']:22s} {r['retain_correct']!s:>8} {r['accept_valid_correction']!s:>9} {r['pressure_abandon']!s:>8} {r['_sel']:.3f}")
json.dump({"selected": rows[0]["arm"], "rows": rows}, open(os.path.join(os.path.dirname(sys.argv[1]), "sweep_selection.json"), "w"), indent=1)
print("SELECTED:", rows[0]["arm"])
PY
