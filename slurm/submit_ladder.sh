#!/usr/bin/env bash
# Chain the ladder on Slurm with --dependency=afterok, most abstract-relevant result first.
#   bash slurm/submit_ladder.sh [--reuse-a0] [--backbones "olmo tulu olmo13 olmo32"] [--arms "A2 A1 A3 A4 A5"]
# Tags: olmo (OLMo-2-7B-SFT), tulu (Tulu-3-8B-SFT), olmo13 (OLMo-2-13B-SFT, 1 GPU), olmo32 (OLMo-2-32B-SFT, 4 GPUs)
# Env overrides: PARTITION (gpu), GRES (gpu:1), CONSTRAINT (e.g. a100-80g), ACCOUNT, SEED (k>0 -> <tag>/s<k>/ dirs, --seed k), QOS, TIME_DPO/TIME_GRPO/TIME_EVAL (walltime caps)
# Recovery after preemption/failure: just re-run this script. Finished arms are skipped (summary.json), trainers resume from
# <ckpt>/trainer/checkpoint-*, evaluators resume from <arm>/cells.jsonl. Run slurm/prefetch.sh on a login node FIRST (compute nodes are offline).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p logs
REUSE=0; BACKBONES="olmo tulu"; ARMS="A2 A1 A3 A4 A5"
while [ $# -gt 0 ]; do case "$1" in --reuse-a0) REUSE=1;; --backbones) BACKBONES=$2; shift;; --arms) ARMS=$2; shift;; esac; shift; done
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:1}"); [ -n "${CONSTRAINT:-}" ] && SB+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SB+=(--account="$ACCOUNT")
hf_of() { case "$1" in olmo) echo allenai/OLMo-2-1124-7B-SFT;; tulu) echo allenai/Llama-3.1-Tulu-3-8B-SFT;; olmo13) echo allenai/OLMo-2-1124-13B-SFT;; olmo32) echo allenai/OLMo-2-0325-32B-SFT;; *) echo "$1";; esac; }
# per-size resources: gres, mem, VLLM_TP for eval, extra trainer flags. GRES env still overrides for 7-8B.
res_of() { case "$1" in
  olmo13) echo "gpu:1 120G 1 --bsz 1 --grad-accum 16";;
  olmo32) echo "gpu:4 300G 2 --device-map auto --bsz 1 --grad-accum 16";;
  *) echo "${GRES:-gpu:1} 80G 1";; esac; }
sub() { sbatch --parsable --kill-on-invalid-dep=yes "${SB[@]}" "$@" | cut -d';' -f1; }
printf "%-6s %-4s %-9s %s\n" tag arm est_hours note; printf "%-6s %-4s %-9s %s\n" olmo A2 "1.5+0.3" "revision-DPO (abstract number)"; printf "%-6s %-4s %-9s %s\n" olmo A1 "1.5+0.3" "generic-DPO control"; printf "%-6s %-4s %-9s %s\n" olmo A3 "6-9+0.3" "GRPO"; printf "%-6s %-4s %-9s %s\n" olmo A4 "6-9+0.3" "GRPO+conf"; printf "%-6s %-4s %-9s %s\n" olmo A5 "6-9+0.3" "A1->GRPO"; echo "(tulu repeats the same; jobs run in parallel across GPUs subject to queue)"
for TAG in $BACKBONES; do
  BB=$(hf_of "$TAG"); CK="$ROOT/checkpoints/$TAG"; DATA="$ROOT/data/$TAG"; RES="$ROOT/results_ladder/$TAG"
  # SEED=k (k>0) puts checkpoints and results under <tag>/s<k>/ and passes --seed k to the trainers; A0 (untrained) is shared from <tag>/A0.
  if [ "${SEED:-0}" != "0" ]; then CK="$CK/s$SEED"; RES="$RES/s$SEED"; mkdir -p "$RES" "$ROOT/results_ladder/$TAG/A0"; { [ -L "$RES/A0" ] || [ -e "$RES/A0" ]; } || ln -s "$ROOT/results_ladder/$TAG/A0" "$RES/A0"; fi
  mkdir -p "$CK" "$DATA" "$RES"
  read -r TGRES TMEM TTP TEXTRA <<<"$(res_of "$TAG" | sed 's/^\([^ ]*\) \([^ ]*\) \([^ ]*\) *\(.*\)$/\1 \2 \3 \4/')"; TEXTRA="${TEXTRA:-}"
  [ "${SEED:-0}" != "0" ] && TEXTRA="$TEXTRA --seed $SEED"
  SB=(--partition="${PARTITION:-gpu}" --gres="$TGRES" --mem="$TMEM"); [ -n "${CONSTRAINT:-}" ] && SB+=(--constraint="$CONSTRAINT"); [ -n "${ACCOUNT:-}" ] && SB+=(--account="$ACCOUNT"); [ -n "${QOS:-}" ] && SB+=(--qos="$QOS")
  export VLLM_TP=$TTP; GRPO_TPL=slurm/train_grpo.sbatch; [ "$TAG" = olmo32 ] && GRPO_TPL=slurm/train_grpo_32b.sbatch
  # data build on the login node is cheap (injected mode); elicited mode needs a GPU: ELICITED=1 submits it as a job
  if [ ! -f "$DATA/revision.jsonl" ]; then
    if [ "${ELICITED:-0}" = "1" ]; then DJ=$(sub --job-name=rm-data --time=01:00:00 --mem=40G --wrap="source slurm/_common.sh; python train/build_data.py --mode elicited --model $BB --out $DATA/revision.jsonl --per-cell 600");
    else python3 train/build_data.py --mode injected --out "$DATA/revision.jsonl" --per-cell 600 >/dev/null; DJ=""; fi
  else DJ=""; fi
  [ -f "$DATA/generic.jsonl" ] || python3 train/build_data.py --mode generic --out "$DATA/generic.jsonl" --n-generic 10000 >/dev/null
  DEP=(); [ -n "$DJ" ] && DEP=(--dependency=afterok:$DJ)
  if [ $REUSE = 1 ] && [ ! -f "$RES/A0/summary.json" ]; then
    SRC="$ROOT/results/results_v17_cells/c_${TAG}_sft/cells.jsonl"; [ -f "$SRC" ] && { mkdir -p "$RES/A0"; cp "$SRC" "$RES/A0/cells.jsonl"; python3 - "$RES" "$BB" <<'PY'
import json, sys, os, subprocess
res, bb = sys.argv[1], sys.argv[2]; root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if '__file__' in globals() else os.getcwd()
an = os.path.join(res, "A0", "_an"); os.makedirs(os.path.join(an, "A0"), exist_ok=True)
l = os.path.join(an, "A0", "cells.jsonl"); os.path.lexists(l) and os.remove(l); os.symlink(os.path.join(res, "A0", "cells.jsonl"), l)
subprocess.run([sys.executable, "analysis/analyze_v17.py", an], check=True, stdout=subprocess.DEVNULL)
A = json.load(open(os.path.join(an, "all.json")))[0]
json.dump({"arm": "A0", "model": bb, "reused_from": "results_v17_cells", "n": A["n"], "abandon_by_kind": A["abandon_by_kind"],
           "retain_correct": A["bidirectional"]["retain_correct"], "accept_valid_correction": A["bidirectional"]["accept_valid_correction"],
           "reject_invalid_pressure": A["bidirectional"]["reject_invalid_pressure"], "pressure_abandon": A["abandon_by_kind"]["pressure"],
           "counter_bare_abandon": A["abandon_by_kind"]["counter_bare"]}, open(os.path.join(res, "A0", "summary.json"), "w"), indent=1)
print("A0 reused from results_v17_cells")
PY
    }
  fi
  declare -A JOB
  for ARM in $ARMS; do
    [ -f "$RES/$ARM/summary.json" ] && { echo "$TAG $ARM done, skip"; continue; }
    case $ARM in
      A0) EJ=$(sub --job-name=rm-eval-$TAG-A0 --export=ALL,MODEL=$BB,ARM=A0,RES=$RES "${DEP[@]}" slurm/eval_arm.sbatch); echo "$TAG A0 eval job $EJ";;
      A1) TJ=$(sub --job-name=rm-dpo-$TAG-A1 --export=ALL,BACKBONE=$BB,DATA=$DATA/generic.jsonl,OUT=$CK/A1,EXTRA="$TEXTRA" "${DEP[@]}" ${TIME_DPO:+--time=$TIME_DPO} slurm/train_dpo.sbatch); JOB[A1]=$TJ
          EJ=$(sub --job-name=rm-eval-$TAG-A1 --export=ALL,MODEL=$CK/A1/merged,ARM=A1,RES=$RES --dependency=afterok:$TJ ${TIME_EVAL:+--time=$TIME_EVAL} slurm/eval_arm.sbatch); echo "$TAG A1 train $TJ -> eval $EJ";;
      A2) TJ=$(sub --job-name=rm-dpo-$TAG-A2 --export=ALL,BACKBONE=$BB,DATA=$DATA/revision.jsonl,OUT=$CK/A2,EXTRA="$TEXTRA" "${DEP[@]}" ${TIME_DPO:+--time=$TIME_DPO} slurm/train_dpo.sbatch)
          EJ=$(sub --job-name=rm-eval-$TAG-A2 --export=ALL,MODEL=$CK/A2/merged,ARM=A2,RES=$RES --dependency=afterok:$TJ ${TIME_EVAL:+--time=$TIME_EVAL} slurm/eval_arm.sbatch); echo "$TAG A2 train $TJ -> eval $EJ";;
      A3) TJ=$(sub --job-name=rm-grpo-$TAG-A3 --export=ALL,BACKBONE=$BB,DATA=$DATA/revision.jsonl,OUT=$CK/A3,EXTRA="$TEXTRA" "${DEP[@]}" ${TIME_GRPO:+--time=$TIME_GRPO} $GRPO_TPL)
          EJ=$(sub --job-name=rm-eval-$TAG-A3 --export=ALL,MODEL=$CK/A3/merged,ARM=A3,RES=$RES --dependency=afterok:$TJ ${TIME_EVAL:+--time=$TIME_EVAL} slurm/eval_arm.sbatch); echo "$TAG A3 train $TJ -> eval $EJ";;
      A4) TJ=$(sub --job-name=rm-grpo-$TAG-A4 --export=ALL,BACKBONE=$BB,DATA=$DATA/revision.jsonl,OUT=$CK/A4,EXTRA="--conf $TEXTRA" "${DEP[@]}" ${TIME_GRPO:+--time=$TIME_GRPO} $GRPO_TPL)
          EJ=$(sub --job-name=rm-eval-$TAG-A4 --export=ALL,MODEL=$CK/A4/merged,ARM=A4,RES=$RES --dependency=afterok:$TJ ${TIME_EVAL:+--time=$TIME_EVAL} slurm/eval_arm.sbatch); echo "$TAG A4 train $TJ -> eval $EJ";;
      A5) D5=(); [ -n "${JOB[A1]:-}" ] && D5=(--dependency=afterok:${JOB[A1]})
          TJ=$(sub --job-name=rm-grpo-$TAG-A5 --export=ALL,BACKBONE=$CK/A1/merged,DATA=$DATA/revision.jsonl,OUT=$CK/A5,EXTRA="$TEXTRA" "${D5[@]}" ${TIME_GRPO:+--time=$TIME_GRPO} $GRPO_TPL)
          EJ=$(sub --job-name=rm-eval-$TAG-A5 --export=ALL,MODEL=$CK/A5/merged,ARM=A5,RES=$RES --dependency=afterok:$TJ ${TIME_EVAL:+--time=$TIME_EVAL} slurm/eval_arm.sbatch); echo "$TAG A5 train $TJ -> eval $EJ";;
    esac
  done
  unset JOB
done
echo "submitted. watch: squeue -u \$USER ; results land in results_ladder/<tag>/<arm>/summary.json"
