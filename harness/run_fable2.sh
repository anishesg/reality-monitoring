#!/usr/bin/env bash
# Stage 2 of the Fable 5.1 replication (V, 2026-09-23): after identification, run contagion on the LIVE API (12 sequential phases; Anthropic
# batches queue ~1 h each) and the v17 decomposition through batches (one parallel round), in parallel. Same ledger and $0 stop.
set -uo pipefail; cd "$(dirname "$0")/.."; source ~/.rm_keys; export RM_SPEND_CAP_ANTHROPIC=1000
until grep -qE "DONE-IDENT-API|STOP|Traceback" logs/fable_all.log; do sleep 15; done
grep -q "DONE-IDENT-API" logs/fable_all.log || { echo "identification did not finish cleanly; not launching stage 2"; exit 1; }
M=claude-fable-5-1; E=low
mkdir -p results_contagion/fable51; [ -s results_contagion/fable51/peer_msgs.jsonl ] || cp results_contagion/qwen7b/peer_msgs.jsonl results_contagion/fable51/peer_msgs.jsonl
# (contagion on the live API and the v17 decomposition through batches were launched directly on 2026-09-23 03:10; see logs/ctg_fable.log, logs/v17_fable.log)
# elicited fill-in with the judge-mapped own answers (swap in own.jsonl.judged now: the first pass must keep the string-matched file so its job hash reattaches), then the equivalence judge
cp results_ident_api/fable51/own.jsonl.judged results_ident_api/fable51/own.jsonl
RM_SPEND_TAG=ident_fable RM_BATCH_INFLIGHT=8 python3 experiments/run_identification_api.py --provider anthropic --model $M --effort $E --claims claims3.jsonl --n 500 --out results_ident_api/fable51 --batch > logs/ident_fable_elicited2.log 2>&1
RM_SPEND_TAG=judge_ident_fable RM_BATCH_INFLIGHT=4 python3 analysis/judge_equiv.py --kind ident results_ident_api/fable51 > logs/judge_ident_fable.log 2>&1
echo "STAGE2-IDENT-JUDGED"
wait
