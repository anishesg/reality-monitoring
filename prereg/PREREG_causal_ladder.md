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
