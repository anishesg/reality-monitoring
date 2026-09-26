# Zero-GPU reanalysis: identification data (results_ident, 6 models × 10,500 trials = 63,000)

Scripts: `~/rm_sprint/reanalysis/reanalyze.py` (full output in `out_reanalyze.txt`) and `reanalyze_extra.py` (`out_extra.txt`).
Data: `ionic:reality-monitoring/results_ident/*`, rsynced to `data/`. CIs are 1,000-draw cluster bootstraps over qid unless noted. Regression CIs are 95% cluster-robust (model×item) coefficient CIs rescaled to AME units.

## 3. Attrition, 63,000 → 52,102 (reproduced exactly)
| model | kept | missing signal | unparsed/amb |
|---|---|---|---|
| Llama-3.1-8B | 9,675 | 747 | 78 |
| OLMo-2-7B | **4,781** | **5,652** | 67 |
| Qwen2.5-14B | 9,929 | 544 | 27 |
| Qwen3-8B | 8,297 | 1,363 | 840 |
| Qwen2.5-7B | 10,120 | 336 | 44 |
| R1-Distill-7B | 9,300 | 796 | 404 |
- The dominant loss is **OLMo: stated confidence is unparsed for 272 of its 500 items**, which drops 54% of its trials. It is not the thinking models: Qwen3 and R1 lose 21% and 11%, and they account for most of the unparsed outcomes (840 and 404).
- By cell, drops are balanced: missing signal FF 3,130, FT 3,182, TF 3,126; unparsed FF 628, FT 310, TF 522.
- fc_correct is also missing for many items (Llama 246/500, R1 378/500). That limits every analysis that signs confidence to the claim.

## 1. FF switching by belief-margin decile (margin = b(alt) − b(claim) = b_d2 − b_d1, per-model deciles)
Margin sizes are not tiny. Within-model SD is 1.46–2.75 nats, and the bottom-decile median runs from −2.5 (Llama) to −4.9 (Qwen3) nats. D0 is the decile where the alternative is believed far less than the claim.
| model | counter D0 → D9 | top − bottom [CI] | weak D0 → D9 | top − bottom [CI] |
|---|---|---|---|---|
| Llama-8B | .996 → 1.000 | +.004 [.000,.012] | .939 → .979 | +.040 [−.042,.122] |
| OLMo-7B | .932 → .984 | **+.052 [.016,.096]** | .660 → .900 | **+.240 [.080,.400]** |
| Qwen-14B | .976 → .988 | +.012 [−.024,.048] | .720 → .860 | +.140 [.000,.300] |
| Qwen3-8B | .838 → .921 | +.083 [−.006,.166] | .674 → .651 | −.023 [−.223,.177] |
| Qwen-7B | .952 → .976 | +.024 [−.016,.068] | .920 → 1.000 | **+.080 [.020,.160]** |
| R1-7B | .914 → .901 | −.013 [−.066,.042] | .720 → .720 | .000 [−.161,.160] |
- In the bottom decile, where the alternative is 2.5–4.9 nats less believed, FF switching is still 0.84–1.00 under the counter and 0.66–0.94 under the weak cue.
- The margin effect is small but not zero off the ceiling. It is significant for OLMo and Qwen-7B under the weak cue, and borderline for Qwen-14B.
- Pooled logit AME is +0.001 to +0.009 per nat under the counter and +0.003 to +0.031 per nat under the weak cue.
- **Switch-to-true in FF: not recoverable.** ident.jsonl and weak.jsonl store only the outcome label, not the final answer. What is known is the FF switch_other rate (ended on neither offered option): counter 0.22–0.52, weak 0.20–0.56, pressure 0.81–0.97.

## 2. Observable-only regression (the most consequential finding)
- **S1, the paper's spec**, reproduces exactly (n = 52,102): counter +.069, weak −.052, claim_true −.142, alt_true +.049, belief margin +.020/SD, confidence −.016/SD.
- **S2, dropping the truth indicators with unsigned confidence:** margin +.031, confidence −.018.
- **The paper's confidence variable is unsigned.** It is confidence in the model's forced-choice answer, whichever option that was. In FT and FF the claim is d1, so high confidence (usually in the true answer) means high confidence that the *claim is wrong*. That pushes toward switching, the opposite direction from TF. Signing confidence to the claim changes the result:
  - **S3, observable-only, confidence signed to the claim** (n = 38,038, fc_correct known): **−.129/SD [−.137, −.122]**, and **−.156/SD** in weak-only trials.
  - **S5, adding the truth indicators back:** signed confidence is still −.062/SD (p = 6e−44), and claim_true shrinks from −.142 to −.029.
- **Decomposition** (`reanalyze_extra.py`): most of the signed effect is whether the injected claim equals the model's own forced-choice answer (own_is_claim).
  - With truth controls: −.148 [−.164, −.133]. Observable-only: −.205. Weak-only, observable: −.307.
  - Graded confidence signed toward the claim adds −.027/SD [−.036, −.018] with truth controls, and −.042/SD in weak-only observable trials. The margin stays at +.014 to +.030/SD.
  - Once own_is_claim is included, claim_true falls to −.033. So the paper's "truth of own claim" effect is mostly **"the claim is the model's own answer."**
- **Raw TF weak split by signed confidence** (high vs low): Qwen-7B .564 vs .829; Qwen-14B .213 vs .456; OLMo .534 vs .675; Llama .775 vs .876.
- FF cell, weak cue: when the injected d1 matches the model's own answer, it switches less. Qwen3 .136 vs .764, Qwen-14B .667 vs .782.
- **Implication:** "stated confidence is worth 0.007" is a misspecification artifact. Models hold their *own* answers much more firmly (≈ −0.15 to −0.31) and use graded confidence modestly (≈ −0.03 to −0.04/SD). What stays near-inert is belief about the incoming alternative (+.015 to +.03/SD).

## 4. Normative benchmark (natural mix; policy fit on qid < 250, evaluated on qid ≥ 250)
Natural mix: the claim is the model's own forced-choice answer. If that answer was correct, the TF trial applies (alt = d1). If wrong, the FT trial applies (claim = d1, alt = truth). Actual post-challenge accuracy is retain on TF and switch_alt on FT, averaged over the 5 paraphrases for the counter. The policy is "keep iff signal ≥ τ".
| model | n_eval | no-revision acc | actual post-counter | actual post-weak | best policy acc (conf / P(True) / b-margin) | policy − actual, counter |
|---|---|---|---|---|---|---|
| Llama-8B | 133 | .835 | .205 | .282 | .835 / .865 / .842 | +.63 [.52,.74] |
| OLMo-7B | 218 | .853 | .240 | .437 | .810* / .835 / .849 | +.60 [.54,.68] (bm) |
| Qwen-14B | 229 | .904 | .313 | .677 | .917 / .904 / .904 | +.60 [.53,.67] |
| Qwen3-8B | 192 | .953 | .551 | .716 | .953 / n.a. / .953 | +.40 [.34,.47] |
| Qwen-7B | 227 | .899 | .233 | .335 | .889 / .899 / .899 | +.67 [.60,.74] |
| R1-7B | 50 | .860 | .344 | .429 | .857 / n.a. / .820 | +.51 [.34,.66] |
(*OLMo conf-policy: n = 100 because confidence is often missing.) Weak-cue gaps: +.23 (Qwen-14B) to +.58 (Llama).
- **Honest caveat:** the fitted τ is nearly degenerate (keep almost always). The confidence policy beats "always keep" by only −0.01 to +0.03. So the 40–67 points left on the table are almost entirely **caving**, not failure to use confidence.
- In this 2-option bank, few items have P(correct) < 0.5, so confidence has little room to add value beyond "don't cave".

## 5. Letter-margin probe audit: very likely a bug, not an invalid probe
- AUROC of margin_debiased against FC correctness: .245 / .295 / .323 / .214 (Llama / OLMo / Qwen-14B / Qwen-7B). Flipped, that is **.755 / .705 / .677 / .786**, the best signal in Qwen-7B and Llama.
- The letter margin favors the option the model *itself chose* in forced choice on only 9–27% of items, and it correlates negatively with the belief margin (−.03 to −.30).
- A valid probe of the same belief cannot disagree with the model's own choice 73–91% of the time. A sign or label error is the parsimonious explanation.
- Candidate mechanism, from code reading of `probe_fix.py` and `run_identification.py`: in the top-20 logprob loop, `pa`/`pb` are overwritten by *every* token whose stripped text is "A"/"B" (e.g. "A" then " A"). The value kept is the last-seen, low-probability variant rather than the max or logsumexp. The chosen letter's variant sits inside the top-20 while the unchosen letter's often doesn't, so the chosen letter gets the smaller logprob and the sign inverts.
- **Not verified**: raw logprob dicts are not stored. Confirm by rescoring with max or logsumexp over variants.
- No letter probe exists for Qwen3 or R1.

## 6. Qwen2.5 ladder, verbalized-confidence AUROC (source: `v16_ladder.txt`, survey hard bank, h_qwen*)
| size | AUROC conf | n (items with conf) | FC acc | mean conf | P(True) AUROC |
|---|---|---|---|---|---|
| 0.5B | .542 | 266 | .530 | 63.6 | .508 |
| 1.5B | .490 | 225 | .573 | 97.4 | .538 |
| 3B | .606 | 263 | .831 | 91.3 | .499 |
| 7B | .660 | 592 | .858 | 94.6 | .547 |
| 14B | .695 | 596 | .905 | 94.1 | .657 |
| 32B | — | 0 | — | — | — |
- The ladder is **not monotone**: it dips at 1.5B, where mean confidence is 97.4, so the confidence is near-constant.
- The small n for 0.5B–3B gives an AUROC SE of roughly ±0.04, so 0.5B vs 1.5B is within noise.
- The subsets differ (items with parsed confidence), and a common-subset recomputation needs per-item survey files that were not synced. The 32B entry has no data.
- Defensible statement: "0.49–0.54 below 3B, rising to 0.66–0.70 at 7–14B."
