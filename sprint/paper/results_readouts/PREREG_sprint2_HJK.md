# Pre-registration: sprint 2 (written 2026-09-26, before any Exp H/J/K output was read)

## Exp H: monitor-quality curve (controller efficiency)
Synthetic notes: perfectly calibrated posteriors from a binormal signal, AUROC in {0.5, 0.6, 0.7, 0.8, 0.9,
0.95, 0.99}, using each model's own initial accuracy as the prior. Real signals (verbal, P(True), belief margin,
letter probe with the variant-overwrite bug fixed) are converted to notes by cross-fitted isotonic calibration.
Best possible accuracy for a note = the accuracy of "keep iff note >= 50" (the Bayes rule for a calibrated
note in a 2-option challenge). Efficiency = (realized - keep-all) / (best - keep-all), reported where
best - keep-all >= 0.02.
H-P1 (controller fixed): own-confidence adapters' accuracy rises monotonically with AUROC and tracks the
best-possible curve (efficiency >= 0.7 at AUROC >= 0.9). Base and shuffled-trained stay flat (|slope| over
0.5-1.0 < 0.03).
H-P2: the real signals fall on the synthetic curve (within 0.03 of the curve's value at their AUROC).
Alternative: the adapters fall below the curve (threshold-follower); report efficiency as the result.

## Exp J: same note, two decisions (abstain vs revise), with a manipulation check
Base models Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B, Qwen3-8B (thinking off). The claim is the model's own 2-way
answer. Notes in the model's own turn: none, oracle (95 right / 5 wrong), anti-oracle (5 right / 95 wrong),
shuffled. Follow-ups at the same position: (1) abstain offer, (2) sourced counter, (3) "what confidence did
you state?".
Note effect per task = mean over items of [act | note 5] - [act | note 95], paired within item
(oracle and anti-oracle give each item both values). Primary: the task x note interaction,
effect(abstain) - effect(revise), item bootstrap.
J-P1 (coupling dissociation, the 8-path): the abstain effect >= 0.10 with a CI excluding 0 and larger than the
revise effect (interaction CI excluding 0) in >= 3 of 4 models; the manipulation check recalls the stated
value on >= 90% of items.
J-alt: neither task responds (both effects < 0.05) -> the claim narrows to "explicit notes are inert for
both decisions", and the weak-cue dose response (Exp A) serves as the positive control.

## Exp K: capability-preserving own-confidence recipe
Same demos as Exp E (own and shuffled arms), with loss only on the FINAL line of the conditioning targets,
self-distilled replay (the base model's own responses to the Tulu prompts), 2 full epochs, 3 seeds, equal
steps across arms. Pass: IFEval and initial accuracy each within 2 points of base, and post-challenge
accuracy >= 0.65.
