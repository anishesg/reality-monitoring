# Drop-in contributions for Anish's Confidence Without Control draft (V, 2026-09-22)

Each file is LaTeX that pastes into the named place of his .tex without other changes.

| file | goes where | status |
|---|---|---|
| `prop2_price.tex` | end of Section 4 (The Monitor Works), after "the failure is not the absence of information" | ready now |
| `robustness_clause.tex` | Section 5.6 (the pre-registered reversal), after the DL interval | ready now |
| `causal_dpo.tex` | Section 6, new paragraph after 6.4 / Appendix C | fills when della returns A1 (results_ladder/*/A1/summary.json) |
| `elicited_ident.tex` | Section 5.1 or Limitations (replaces "future work") | fills when `experiments/run_identification_elicited.py` returns |
| `frontier_ident.tex` | Section 5.2, one paragraph + one table row block | fills when `experiments/run_identification_api.py` returns (needs API keys) |
| `judge_validity.tex` | Setup, one sentence + appendix table | fills when `train/judge_check.py` runs on 500 identification trials with stored responses |
| `agents_consequence.tex` | Section 7 (Two Consequences), third consequence, half page | ready now (Q8 numbers; swap in bf16 when della returns) |

Numbers in the "ready now" files are from the repository at commit 880612a.
