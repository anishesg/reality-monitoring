#!/usr/bin/env bash
# Stage 3 (2026-09-23): when Fable contagion (live) and decomposition (batch) have finished, judge both with the Astra equivalence judge,
# regenerate the frontier figures/tables/summary and the contagion reports. Identification is judged by stage 2.
set -uo pipefail; cd "$(dirname "$0")/.."; source ~/.rm_keys; export RM_SPEND_CAP_ANTHROPIC=1000
until grep -q "DONE-CONTAGION" logs/ctg_fable.log 2>/dev/null; do sleep 20; done
RM_SPEND_TAG=judge_ctg_fable RM_BATCH_INFLIGHT=4 python3 analysis/judge_equiv.py --kind contagion results_contagion/fable51 > logs/judge_ctg_fable.log 2>&1 &
until grep -q '"tokens_in"' logs/v17_fable.log 2>/dev/null; do sleep 20; done
RM_SPEND_TAG=judge_v17_fable RM_BATCH_INFLIGHT=4 python3 analysis/judge_equiv.py --kind v17 results_api/fable51 > logs/judge_v17_fable.log 2>&1 &
until grep -q "STAGE2-IDENT-JUDGED" logs/fable_stage2.log 2>/dev/null; do sleep 20; done
wait
python3 analysis/figures_frontier.py > logs/figures_frontier.log 2>&1
python3 analysis/report_contagion.py results_contagion/fable51 > results_contagion/fable51/report.txt 2>&1
python3 harness/spend.py > logs/spend_final.txt 2>&1
echo "STAGE3-DONE"
