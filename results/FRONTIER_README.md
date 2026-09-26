# Frontier replication (2026-09-22): GPT-6 Astra, Claude Fable 5.1

Same three experiments as the open models, same questions, same prompts, same challenge texts; only the generation backend differs.
Runs are one deterministic pass (seed 0 where the API accepts it), reasoning effort **low**, through each provider's Batch API
(OpenAI Batch API for Astra, Anthropic Message Batches for Fable; half list price; the org's live limit for gpt-6-astra is 50 requests/day).
Spend is capped in code (`harness/spend.py`, hard stop at $250 per provider, ledger `~/.rm_spend/`, `--rebuild` recomputes from the
providers' own job records). Astra total: **$144.45** for 46k requests including the judge passes.

| experiment | script | Astra | Fable 5.1 |
|---|---|---|---|
| v17 decomposition, 450 q, all 12 cells | `harness/run_cells_v17_api.py --batch` | `results_api/astra` (10,800) | `results_api/fable51` |
| three-cell identification, injected + elicited, 500 q | `experiments/run_identification_api.py --batch` | `results_ident_api/astra` (11,280 + 500 signals) | `results_ident_api/fable51` |
| answer contagion, 300 q, pairwise/genuine/chain k=8, Qwen 1.5B/7B/14B peers + self as sender | `harness/contagion.py --backend batch` | `results_contagion/astra` (14,259) | `results_contagion/fable51` |

## Grading: exact match, then an equivalence judge
Frontier models restate a retained answer in new words ("Compounded quarterly" -> "Quarterly compounding is more profitable"), which the
exact-match grader used for the open models scores as `switch_other`. `analysis/judge_equiv.py` re-grades every `switch_other` / `ambiguous`
row with an LLM judge (gpt-6-astra, ~20 output tokens: does the final answer express the CLAIM, the ALTERNATIVE, the TRUE answer, or something
else?) and writes `<run>/judged.jsonl`; the original string outcome is kept as `outcome_string`. Rows re-graded for Astra: 3,686 / 10,800
(decomposition), 6,777 / 11,280 (identification), 5,110 / 14,259 (contagion). The judge also maps the model's own free-form answers onto the
bank candidates for the elicited cells (`--own`; 271 of 409 unmatched answers mapped, 130 match no candidate and are excluded).
`RM_GRADING=string` reproduces the raw numbers. Open-model runs did not store responses, so they remain string-graded; their FINAL lines copy
the injected string almost verbatim, so the correction there is small, but this asymmetry is stated wherever the two are compared.

## Astra results (judge-graded; 95% question-cluster bootstrap CIs)
**Three-cell, injected, counter-argument.** T->F abandon 0.22 [0.18,0.25] (open 7B-14B: 0.80-0.94). F->T accept 0.95 [0.94,0.97].
F->F leave 0.96 [0.95,0.98], of which to the cue 0.23 [0.20,0.27] and to the TRUE answer 0.57 [0.53,0.62] (open models: ~0 to truth).
Pressure abandon 0.21. Elicited (own reasoned reply as the challenged turn, 357 q): T->F 0.19 [0.15,0.22], pressure 0.13.
**Decomposition, correct self claims.** sourced counter 0.16, bare counter 0.15, source-only 0.21, pressure 0.18; confidence use 0.16 -> 0.16.
**Contagion.** fold to a peer naming the alternative: none 0.15, mention 0.11, reason 0.11, conf_lo 0.10, conf_hi 0.11, gen 1.5B/7B/14B/Astra
0.12/0.11/0.12/0.12 (spread 0.015; P1 holds). Genuine (own reasoned answer): fold 0.00-0.03, accept true alternative 0.5-1.0 (n small).
Chains: p_ww 0.96 [0.93,0.98], p_cw 0.01, lambda 0.95, pi_inf 0.21; wrong seed 0.19 flat over 8 hops, right seed 0.10 -> 0.14; from-k1
closed form 8/8 inside for both.

## Files
`analysis/figures_frontier.py` -> `figures/frontier_{ident,ident_elicited,decomp,confuse,contagion}.pdf`, `results/frontier_summary.json`,
`paper/contrib/frontier_table.tex`; paragraph draft `paper/contrib/frontier_ident.tex`. Spend: `python3 harness/spend.py`.
