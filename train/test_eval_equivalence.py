#!/usr/bin/env python3
"""Guard: the resumable evaluator (train/eval_arm.build_trials) must produce exactly the dialogues and trial order of the
frozen harness (harness/run_cells_v17.py). CPU only, no model.  python3 train/test_eval_equivalence.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, read_jsonl
from eval_arm import build_trials
sys.path.insert(0, os.path.join(ROOT, "harness"))
import run_cells_v17 as h

recs = read_jsonl(os.path.join(ROOT, "harness", "claims_hard.jsonl"))[:450]
trials, msgs = build_trials(recs)
ref_msgs, ref_trials = [], []
for r in recs:  # verbatim from run_cells_v17.main
    for truth in (True, False):
        claim = r["true_answer"] if truth else r["distractor"]; alt = r["distractor"] if truth else r["true_answer"]
        for conf in ("low", "high"):
            for kind in ("counter_src", "counter_bare", "src_only", "pressure"):
                ref_msgs.append([{"role": "system", "content": h.SYS}, {"role": "user", "content": f"Question: {r['question']}\nPlease give your best answer."},
                                 {"role": "assistant", "content": h.CONF_SELF[conf] + f"FINAL: {claim}"}, {"role": "user", "content": h.chal_text(kind, alt)}])
                ref_trials.append({"qid": r["qid"], "origin": "self", "truth": truth, "conf": conf, "kind": kind, "claim": claim, "alt": alt})
            for kind in ("counter_src", "pressure"):
                ref_msgs.append([{"role": "system", "content": h.SYS}, {"role": "user", "content": f"Question: {r['question']}\n{h.CONF_USER[conf]} \"{claim}\". Can you keep that in mind?"},
                                 {"role": "assistant", "content": f"Understood. Your proposed answer is \"{claim}\". FINAL: {claim}"}, {"role": "user", "content": h.chal_text(kind, alt)}])
                ref_trials.append({"qid": r["qid"], "origin": "user_recency", "truth": truth, "conf": conf, "kind": kind, "claim": claim, "alt": alt})
assert len(msgs) == len(ref_msgs) == 10800, (len(msgs), len(ref_msgs))
assert msgs == ref_msgs, "dialogues differ from the frozen harness"
assert trials == ref_trials, "trial metadata differs from the frozen harness"
ptrials, pmsgs = build_trials(recs, paraphrase=True)
assert len(pmsgs) == 450 * 2 * 2 * 4 and all(m[-1]["content"] != r[-1]["content"] for m, r in zip(pmsgs, [x for x in ref_msgs if x[1]["content"].endswith("best answer.")]))
print("OK: resumable evaluator reproduces the frozen v17 harness exactly (10,800 dialogues); paraphrase set is disjoint")
