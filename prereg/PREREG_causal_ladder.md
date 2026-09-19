# Pre-registered predictions: causal post-training ladder (frozen BEFORE any arm is trained)
Date: 2026-09-18. Backbones: allenai/OLMo-2-1124-7B-SFT, allenai/Llama-3.1-Tulu-3-8B-SFT.
Eval: harness/run_cells_v17.py on the held-out first 450 hard-bank qids (never used in training); endpoints as in analysis/analyze_v17.py.
Arms: A0 SFT | A1 generic-DPO (10k Tulu-3 preference pairs) | A2 revision-DPO | A3 revision-GRPO (verifiable reward) |
      A4 confidence-conditioned GRPO | A5 A1 -> revision-GRPO.

H1 (DPO installs pressure capitulation, causal): A1 pressure_abandon(self, truth=T) > A0 by >= 0.15 in both backbones,
   with counter_bare abandonment unchanged (+/- 0.05). Falsified if A1 - A0 < 0.05 in both.
H2 (verifiable reward populates the frontier): A3 retain_correct >= 0.70 AND accept_valid_correction >= 0.70 in >= 1 backbone;
   A2 improves retain_correct but stays < 0.70 on at least one axis (preference data underfits the asymmetry).
H3 (repair): A5 pressure_abandon <= A0 + 0.05 (RLVR on top of DPO removes the installed defect).
H4 (confidence coupling): A4 conf_use_self (counter_src, low-high) >= 0.20 vs < 0.05 in A0..A3; verbalized-confidence AUROC not lower than A0.
H5 (no capability cost): mmlu/gsm8k/ifeval within 1.0 point of A0 for A2..A5; A1 may move either way.
H6 (generalization): the A3 gain in retain_correct is >= 60% as large on TruthfulQA items as on MMLU-Pro items (train bank is SciQ + MMLU-Pro 450-899 only).
Decisive negative: if A3 retain_correct < 0.5 in both backbones, the fix is not achievable with LoRA + 128-token rollouts at this budget; report as such.

Reward definitions (frozen with the predictions): A3/A5 reward = 1[final answer correct] + 0.1[explicit FINAL line]. A4 reward = A3 reward + 0.5*(1 - Brier(stated confidence in final answer, correctness)), -0.3 if no CONFIDENCE line. No term rewards conditioning on the prior stated confidence; H4 tests whether that emerges.

## Scale rung (added 2026-09-18, before any 13B/32B result is read)
Checkpoints: allenai/OLMo-2-1124-13B-{SFT,DPO,Instruct}, allenai/OLMo-2-0325-32B-{SFT,DPO,Instruct}; STAND arms trained on the 13B/32B SFT.
H7 (defect persists with scale): the DPO-stage rise in pressure_abandon seen at 7B (0.19->0.73) is >= 0.15 at both 13B and 32B, and the
   RLVR stage does not bring it back below SFT + 0.05. Falsified if the 32B DPO - SFT gap is < 0.05.
H8 (fix persists with scale): A2 (STAND-DPO) gain in retain_correct at 32B is >= 60% of the 7B gain; if A3 runs at 32B, same for A3.
   No claim is made about models larger than 32B; the scale trend within 7B-32B is the evidence offered.

## Frozen run list and hyperparameter provenance (2026-09-19, before any arm has trained)
Splits. TEST = hard-bank qids 0-449 (all paper numbers). DEV = hard-bank qids 800-899, held out of all training and used only
for the A2 sensitivity sweep below. TRAIN = SciQ (600) + hard-bank qids 450-799. No number reported in the paper is computed on DEV.

Runs this week, in priority order (C = conditional):
 R1  OLMo-2-7B-SFT: A0 (reused), A2, A1, A3, A4, A5 on TEST.
 R2  Tulu-3-8B-SFT: A0 (reused), A2, A1, A3 on TEST. (A4, A5 conditional on R1 A4/A5 being informative.)
 R3  A2 sensitivity sweep on DEV (train/sweep_a2.sh): beta in {0.05, 0.1, 0.2} at lr 5e-6; lr in {2e-6, 1e-5} at beta 0.1; OLMo only.
     Selection rule fixed in advance: max min(retain_correct, accept_valid_correction) on DEV. The selected setting is used for
     A2 on TEST; all five DEV rows are reported in the appendix. If the selected setting is not (0.1, 5e-6), R1 A2 is rerun with it.
 R4  LLM-judge agreement check on 500 stratified TEST trials from the published-checkpoint study (train/judge_check.py).
 R5  Stage measurement at 13B and 32B (slurm/measure_stages.sh), no training.
 R6C OLMo-2-32B-SFT: A0, A2 (needs 4x80GB). R7C OLMo-2-13B-SFT: A0, A2, A1, A3. R8C 32B A3 only if R1 A3 retain_correct >= 0.6.
 R9C Frontier-API measurement of the v17 decomposition (150 TEST questions x self-origin cells) on 4-5 API models, if budget allows.

Hyperparameters and where each number comes from (S = standard default in the method's paper or library; H = our heuristic, swept
or ablated; F = fixed by the existing harness and not changed):
 DPO beta 0.1 (S, Rafailov et al. 2023 default; swept R3) | lr 5e-6 (S, common LoRA-DPO setting; swept R3) | epochs 2 (H; not swept,
 loss curve reported) | effective batch 16 (H) | max_length 1024 (F: dialogues are < 600 tokens) | LoRA r 64, alpha 128, dropout 0.05,
 all linear layers (S, common; r not swept) | GRPO beta 0.04 (S, DeepSeekMath default) | G 8 (S) | completion 128 tokens (F: the
 harness format is two sentences + FINAL line; 288-token eval cap) | temperature 1.0 (S) | lr 5e-6 (H, same as DPO) | 1 epoch (H).
 Reward: +1 correct (definitional) | +0.1 FINAL-format bonus (H, standard practice; ablated at 0 in one A3 run if time allows) |
 A4: -0.3 missing CONFIDENCE, +0.5*(1-Brier) (H; the 0.5 weight is ablated at {0.25, 1.0} only if A4 is informative).
 Data: 600 pairs per cell x 12 cells = 7,200 revision pairs (H: chosen so one epoch is ~450 optimizer steps at batch 16); 10,000
 generic pairs (H: matches the revision set within 1.4x so A1 vs A2 differ in content, not volume).
 Analysis: frontier marker 0.70 on both axes (H: a pre-registered marker, not a claim; the paper reports the full (r, a) frontier and
 Pareto dominance over A0, with the 0.70 line drawn for reference) | confidence bins <=80 vs >=95 in v16 (F, from the existing
 published-checkpoint analysis; reported alongside AUROC, which is bin-free) | cluster bootstrap 500 resamples by question (S) |
 capability tolerance 1.0 point (H, conventional) | judge sample 500 (H: gives +/-4 pt agreement CI).
Predictions H1-H8 above are unchanged. Effect-size thresholds in H1 (0.15), H4 (0.20), H8 (60%) are heuristics chosen before data
and are reported as such; the primary inferential statements are CIs and Pareto comparisons, not threshold crossings.
