# Essential experiments, ranked by yield per GPU-hour (V, 2026-09-24, deadline 2026-09-25 AoE)

The paper already stands as a measurement + mechanism + frontier paper. What lifts it from "accept" to "spotlight" is
(a) a second frontier family showing the same structure, (b) the causal origin (A1), (c) the fix at capability parity (A3).
(a) turned out to be free. (b) is the single cheapest GPU job with the largest payoff. (c) is the expensive one.

## Tier 0: zero GPU. Done or running today.
| # | Experiment | Status | What it does for the paper |
|---|---|---|---|
| E0 | Claude Fable 5.1 on the three-cell identification (12k trials) and the decomposition (10.8k trials), judge-graded | DONE. Data was collected 2026-09-22/23 and excluded only because of the billing incident; re-included 2026-09-24 | Second frontier family. Replicates every Astra signature: T->F 0.10 (Astra 0.22, open models 0.56-0.94); F->F 0.89 with 0.61 of it re-derived to the *true* answer (Astra 0.57; open models 0.00); F->T 0.95; confidence effect <= 0.03; source effect ~0. The "recompute-to-truth at the frontier" claim now rests on two model families. |
| E1 | Elicited three-cell identification on the six open models (model's own reasoned answer as the challenged turn), llama.cpp Q8 on the 64-core Azure CPU VM | RUNNING since 2026-09-24 ~06:10 UTC (`azure/cpu_run_ident_elicited.sh`, tmux `elic`, order qwen7b olmo llama8b qwen14b qwen3_8b r1_7b; ~2-3 h per 7B model, ~4 h for 14B) | Closes the limitation stated in Section 9 and fills the "elicited" columns of Table 2 for open models, so the open-vs-frontier comparison is like-for-like. |
| E2 | Contagion chain for Qwen2.5-14B on the CPU VM (GGUF cached), ~8 h | queue after E1 if time remains | One more point on the 1.5B -> 7B -> 14B -> frontier scale curve. Optional. |

## Tier 1: the first 30 GPU-hours (run these before anything else)
| # | Job | GPU-h | Prediction (pre-registered) | If it lands | If it fails |
|---|---|---|---|---|---|
| G1 | A1 = published DPO recipe on OLMo-2-7B-SFT and Tulu-3-8B-SFT, seed 0, + A0/A1 challenge eval | ~10 | H1: pressure-only abandonment rises >= 0.15 over A0 in both backbones | Section 7 goes from hypothesis to causal result: generic preference tuning *installs* the cue. This is the paper's origin claim. | The corpus account loses its causal leg; Section 7 becomes an honest null, the paper stays a measurement + frontier paper. Still submittable. |
| G2 | A1 seeds 1-2 on both backbones | ~20 | same | Seed-robust H1; the single-seed rule is satisfied for the headline. | Report the seed spread. |
| G3 | Capability evals (MMLU, GSM8K, IFEval) for A0 and A1 | ~4 | H5 | Shows the defect is installed at no capability change. | |

## Tier 2: the next 100 GPU-hours
| # | Job | GPU-h | If it lands |
|---|---|---|---|
| G4 | A3 (STAND, GRPO on post-challenge correctness) + A2 (revision-DPO control) on OLMo, 3 seeds, + evals | ~55 | H2/H3: retain >= 0.6 and accept >= 0.7 at capability within a point of A0. The fix at capability parity, with A2 as the same-signal control. Together with G1 this is the full arc: origin, fix, control. |
| G5 | Same on Tulu | ~55 | Two backbones for the fix. Skip if G4 is not clean. |
| G6 | Firewall chain: one A3 agent at position two of a contaminated Qwen chain | ~1 (needs G4) | Tests the Lean-proved reset-and-reconverge prediction (P4: downstream error cut >= 0.40). This is the result that makes the agents section predictive rather than descriptive. |

## Drop (not worth the hours before this deadline)
A4/A5 arms (H3/H4 confidence-coupling), the A2 hyperparameter sweep (10 h), 13B/32B scale rung (H7/H8, 40+ h), 72B contagion at bf16.
Contagion at bf16 for the 7B models (replacing the 8-bit CPU numbers) is a precision upgrade, not a new result.

## If AWS arrives (200 GPU-hours, ~24 h wall clock on 8 GPUs)
Run order: G1 -> G3 -> G2 -> G4 -> G6 -> G5. G1 finishes in ~4 h wall clock; the paper's Section 7 paragraphs and Table 4 fill from
`results_ladder/` via the existing report scripts. Everything through G4 fits in ~90 GPU-hours, which leaves margin for a failed seed.

## If no GPU arrives
The paper is submitted as measurement + mechanism + two frontier families + elicited open-model replication (E0 + E1), with Section 7
rewritten as a pre-registered design whose runs are pending. G1 alone (~10 GPU-h, one A100 for one afternoon) is the cheapest purchase
that changes the paper's category; that spend is your call.
