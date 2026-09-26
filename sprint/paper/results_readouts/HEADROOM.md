# Rate-matched headroom (H_rate) and use share (U): does the model allocate its switches as well as its own confidence would?
Script: `headroom.py` (full per-checkpoint output in `headroom_out.txt`, JSON in `headroom.json`). The stable-component table is in `headroom_table.txt`.

**Definitions.** Natural mix: the claim is the model's own answer. One sourced counter per item proposes the truth if the model was wrong and a distractor if it was right. Scoring uses allocation semantics: a switch on a right item loses the point, and a switch on a wrong item gains it.
- s = the model's switch rate.
- Acc(rnd) = (1−s)·A0 + s·(1−A0), computed analytically.
- Acc(sig) switches the s·N lowest-signal items; ties at the boundary are allocated fractionally.
- H_rate = Acc(sig) − Acc(model).
- U = (Acc(model) − Acc(rnd)) / (Acc(sig) − Acc(rnd)).
- The threshold policy fits τ on the analysis half (survey: qid < 450; ident: qid < 250) and is scored on the held-out half.
- Bootstrap: 1,000 item resamples.

**Data used.**
- **Survey:** `results_v16/h_*/cells.jsonl`, `challenge` cells, `chal == counter`, with `truth` known and outcome parsed. One trial per item. The model's own stated confidence is visible in its prior turn.
- **18 of 19 checkpoints usable.** h_llama1b has only 24 counter trials and is excluded. h_qwen72b and h_gemma9b have no cells.jsonl.
- **Ident:** `results_ident`. The claim is the model's own 2-way FC answer: TF trial if correct, FT if wrong. The per-item switch probability pools the 5 counter paraphrases; the weak cue is analyzed separately.
- **Signals:** survey: verbal conf, post-hoc conf, P(True). Ident: conf, P(True) (Qwen3/R1 generated verdicts are unusable as a ranking), and belief margin b(own) − b(other).

## Result: U is not the right summary. The model out-targets its own stated confidence.
The denominator Acc(sig) − Acc(rnd) is tiny: stated confidence barely beats random at choosing which items to give up. So U explodes, with a survey median of +2.98 and a range of −846 to +224. U > 1 and H_rate < 0 mean the model's actual switches are *better* targeted than confidence-ranked switches.

The stable quantities are the gains over random allocation, in accuracy points. Medians over the 18 survey checkpoints:

| | model's actual switches | allocate by verbal conf | allocate by P(True) | oracle |
|---|---|---|---|---|
| gain over random (pts) | **+9.1** | +1.7 | +1.5 | +17.7 |
| share of oracle | **0.53** | 0.11 | 0.08 | 1 |

- **H_rate (verbal conf):** median −6.1 pt, range −14.1 to +2.8. It is positive only for Qwen-0.5B (+2.8), Qwen-1.5B (+0.5) and Mistral (+0.4); those models' switch targeting is at chance (−2.9 to −0.6).
- **Ident, sourced counter:** median U = 1.59 for conf (range 1.03–5.72), 1.03 for P(True), 2.10 for b-margin. H_rate is −1.2, −0.0 and −2.0 pt. The ident counter sits near the ceiling (s = 0.47–0.93), so all gains are small, at most 8.5 pt.
- **Ident, weak cue:** conf U median 0.96, range −0.49 to 1.76; H_rate +0.2 pt, range −3.2 to +3.1.
- **Threshold policy (held-out):** it equals keep-all in 15 of 18 checkpoints; the fitted τ is degenerate at keep-everything. It beats the model's actual post-counter accuracy by up to +0.35 (Qwen-14B 0.876 vs 0.530), but that is refusing to cave, not using confidence. Exceptions: tulu_rlvr, where the model beats keep-all (0.692 vs 0.661), and phi35 (0.711 policy, 0.698 keep-all, 0.680 model).

## Qwen2.5 ladder (survey, verbal conf)
| size | n | A0 | s | U [95% CI] | H_rate | model gain | conf gain |
|---|---|---|---|---|---|---|---|
| 0.5B | 265 | .51 | .92 | −2.70 [−29.8, 34.8] | +2.8 | −2.0 | +0.8 |
| 1.5B | 183 | .58 | .77 | +4.02 [−73, 86] | +0.5 | −0.6 | −0.2 |
| 3B | 262 | .83 | .65 | +2.93 [1.08, 13.96] | −4.8 | +7.3 | +2.5 |
| 7B | 589 | .86 | .60 | +2.37 [1.65, 3.87] | −5.4 | +9.3 | +3.9 |
| 14B | 596 | .91 | .55 | +1.29 [0.75, 2.42] | −1.1 | +4.8 | +3.7 |
| 32B | 603 | .92 | .36 | +1.26 [0.93, 1.88] | −1.4 | +6.9 | +5.5 |

On the ladder, U falls toward 1 with scale. The reason is that stated confidence improves as a targeting signal (conf gain rises from +0.8 to +5.5), not that the model's targeting worsens.

## Implication for the PI's decision rule
The rule says: if U ≥ 0.5, use the fallback wording, "models target switches with latent belief, but stated confidence adds nothing". U ≥ 0.5 holds in every checkpoint with non-chance targeting, so the **fallback applies**. The thesis wording "confidence could recover H1 points, models recover almost none" is **not supported**. Rate-matched, stated confidence would recover *less* than the model already does (median −6 pt).

What the data do support:
1. The model captures about half the oracle-attainable targeting, from latent knowledge.
2. Stated confidence and P(True), used as allocation rules, capture only 8–11%.
3. The large accuracy loss under challenge is the switch *rate* s. Keep-all beats the model by up to 35 pt, which is caving, not misallocation.

Caveats:
- Allocation semantics count switch_other on wrong items as corrections. switch_other share is 0.01–0.19; Zephyr-DPO is highest at 0.189. Strict actual accuracy is in headroom_out.txt.
- Survey confidence is coarse, with many ties at 95/100, which caps the confidence-allocation gain.
