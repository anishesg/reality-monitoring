#!/usr/bin/env bash
# Causal post-training ladder, one backbone, one node. Idempotent: an arm is skipped when its summary.json exists.
#   bash train/run_ladder.sh [--reuse-a0] allenai/OLMo-2-1124-7B-SFT olmo [A2 A1 ...]
# Default order puts the abstract-deciding number first: A2 (revision-DPO) -> A1 (generic-DPO) -> A3 -> A4 -> A5.
# --reuse-a0 copies the existing v17 eval of the SFT backbone (results/results_v17_cells/c_<tag>_sft) instead of re-running A0.
# Env: ELICITED=1 (build A2/A3 data on the model's own answers), QUICK=1 (tiny smoke), VLLM=0 (no colocated rollouts), NQ (eval questions)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REUSE=0; [ "${1:-}" = "--reuse-a0" ] && { REUSE=1; shift; }
BACKBONE="${1:?backbone HF id}"; TAG="${2:?short tag e.g. olmo}"; shift 2
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(A0 A2 A1 A3 A4 A5)
PY="${PYTHON:-python3}"; export USE_TF=0 TRANSFORMERS_NO_TF=1
CK="$ROOT/checkpoints/$TAG"; DATA="$ROOT/data/$TAG"; RES="$ROOT/results_ladder/$TAG"; mkdir -p "$CK" "$DATA" "$RES"
Q=""; [ "${QUICK:-0}" = "1" ] && Q="--limit 64 --max-steps 4"
VL="--vllm"; [ "${VLLM:-1}" = "0" ] && VL=""
CAP="--capability"; [ "${QUICK:-0}" = "1" ] && CAP="--capability --quick"
done_arm() { [ -f "$RES/$1/summary.json" ]; }
reuse_a0() { SRC="$ROOT/results/results_v17_cells/c_${TAG}_sft/cells.jsonl"; [ -f "$SRC" ] || { echo "no reusable A0 at $SRC"; return 1; }
  mkdir -p "$RES/A0/_an/A0"; cp "$SRC" "$RES/A0/cells.jsonl"; ln -sf "$RES/A0/cells.jsonl" "$RES/A0/_an/A0/cells.jsonl"
  $PY "$ROOT/analysis/analyze_v17.py" "$RES/A0/_an" >/dev/null
  $PY - "$RES/A0" "$BACKBONE" <<'PYEOF'
import json, sys, os
d, bb = sys.argv[1], sys.argv[2]; A = json.load(open(os.path.join(d, "_an", "all.json")))[0]
json.dump({"arm": "A0", "model": bb, "reused_from": "results/results_v17_cells", "n": A["n"], "abandon_by_kind": A["abandon_by_kind"],
  "source_effect": A["source_effect_content_matched"]["delta"], "conf_use_self_counter_src": A["conf_use_self_by_kind"]["counter_src"]["delta"],
  "retain_correct": A["bidirectional"]["retain_correct"], "accept_valid_correction": A["bidirectional"]["accept_valid_correction"],
  "reject_invalid_pressure": A["bidirectional"]["reject_invalid_pressure"], "pressure_abandon": A["abandon_by_kind"]["pressure"],
  "counter_bare_abandon": A["abandon_by_kind"]["counter_bare"], "excluded_frac": None}, open(os.path.join(d, "summary.json"), "w"), indent=1)
print("A0 reused from results_v17_cells (c_%s_sft)" % os.path.basename(os.path.dirname(d)))
PYEOF
}
echo "== ladder for $BACKBONE ($TAG); arms: ${ARMS[*]}"
printf "%-4s %-10s %s\n" arm est_hours note
printf "%-4s %-10s %s\n" A0 "0.3 (or 0)" "SFT baseline eval (0 with --reuse-a0)"
printf "%-4s %-10s %s\n" A2 "1.5+0.3" "revision-DPO (abstract number)"
printf "%-4s %-10s %s\n" A1 "1.5+0.3" "generic-DPO control (10k Tulu-3 pairs)"
printf "%-4s %-10s %s\n" A3 "6-9+0.3" "revision-GRPO, verifiable reward"
printf "%-4s %-10s %s\n" A4 "6-9+0.3" "GRPO + confidence reward"
printf "%-4s %-10s %s\n" A5 "6-9+0.3" "A1 merged -> revision-GRPO"
echo "(one NC24ads_A100_v4; +0.5h each with --capability; ETA total ~25-32h sequential)"
evalarm() { $PY "$ROOT/train/eval_arm.py" --model "$2" --arm "$1" --results-root "$RES" $CAP ${NQ:+--n-questions $NQ}; }

# data (built once per backbone)
[ -f "$DATA/revision.jsonl" ] || {
  if [ "${ELICITED:-0}" = "1" ]; then $PY "$ROOT/train/build_data.py" --mode elicited --model "$BACKBONE" --out "$DATA/revision.jsonl" --per-cell 600
  else $PY "$ROOT/train/build_data.py" --mode injected --out "$DATA/revision.jsonl" --per-cell 600; fi; }
[ -f "$DATA/generic.jsonl" ] || $PY "$ROOT/train/build_data.py" --mode generic --out "$DATA/generic.jsonl" --n-generic 10000

for ARM in "${ARMS[@]}"; do
  if done_arm "$ARM"; then echo "== $ARM done, skip"; continue; fi
  echo "== $ARM ($TAG) $(date -u +%FT%TZ)"
  case "$ARM" in
    A0) if [ $REUSE = 1 ] && reuse_a0; then :; else evalarm A0 "$BACKBONE"; fi ;;
    A1) $PY "$ROOT/train/dpo.py" --model "$BACKBONE" --data "$DATA/generic.jsonl" --out "$CK/A1" --merge $Q
        evalarm A1 "$CK/A1/merged" ;;
    A2) $PY "$ROOT/train/dpo.py" --model "$BACKBONE" --data "$DATA/revision.jsonl" --out "$CK/A2" --merge $Q
        evalarm A2 "$CK/A2/merged" ;;
    A3) $PY "$ROOT/train/grpo.py" --model "$BACKBONE" --data "$DATA/revision.jsonl" --out "$CK/A3" --merge $VL $Q
        evalarm A3 "$CK/A3/merged" ;;
    A4) $PY "$ROOT/train/grpo.py" --model "$BACKBONE" --data "$DATA/revision.jsonl" --out "$CK/A4" --merge --conf $VL $Q
        evalarm A4 "$CK/A4/merged" ;;
    A5) [ -d "$CK/A1/merged" ] || { echo "A5 needs A1 merged weights"; exit 1; }
        $PY "$ROOT/train/grpo.py" --model "$CK/A1/merged" --data "$DATA/revision.jsonl" --out "$CK/A5" --merge $VL $Q
        evalarm A5 "$CK/A5/merged" ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
done
$PY - "$RES" <<'PYEOF'
import json, glob, os, sys
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*", "summary.json")))]
print(f"{'arm':4s} {'retainC':>8} {'acceptFix':>9} {'pressAb':>8} {'bareAb':>7} {'excl':>5}")
for r in rows: print(f"{r['arm']:4s} {r['retain_correct']!s:>8} {r['accept_valid_correction']!s:>9} {r['pressure_abandon']!s:>8} {r['counter_bare_abandon']!s:>7} {r['excluded_frac']!s:>5}")
json.dump(rows, open(os.path.join(sys.argv[1], "ladder.json"), "w"), indent=1)
PYEOF
