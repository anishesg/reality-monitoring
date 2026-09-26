# Pre-registration: sprint experiments B, C, E (written 2026-09-26 01:39 ET)
Written after job submission, before any trial output was inspected. The hash of this file is
recorded in git, and no results files exist yet under sprint/res_b, res_c, or res_e.

## Exp B: model-owned claims, manipulated belief margin (10-option MMLU-Pro)
Primary estimand: D = P(switch | FFfar) - P(switch | FFnear), per model and challenge
(counter, weak), with a cluster bootstrap over items. Near and far are the model's best- and
least-believed remaining wrong options, by b(x) in the same context the claim was made in.
Pipeline check: under pressure, which never names the alternative, near and far switch
rates must agree within their 95% CIs.
Validity: in a clean 2-way choice, the model prefers its claim over the far alternative
>= 90% of the time.
Prediction P1 (confidence-use gap holds): under the sourced counter, |D| <= 0.10 in at least
5 of 7 models, and FFfar switching >= 0.70.
Alternative P1' (belief is used when strong): D <= -0.15 (far switched to less) in the majority
of models under the counter or the weak cue. If P1' holds, the paper is reframed as "models act
on strong latent belief but ignore stated confidence and moderate belief".
Secondary: TFfar vs TFnear gives the same contrast for true claims; FT - FFnear gives the value
of truth of the alternative at a matched (near) margin.

## Exp C: own confidence fed through adapters trained on the reliability rule
Primary: post-challenge accuracy (sourced counter, natural challenge mix), with the calibrated
own-confidence note vs the shuffled note, averaged over 3 treatment seeds; item bootstrap.
Prediction P2: calibrated > shuffled with a 95% CI excluding 0, and calibrated > base (no note).
Kill: calibrated - shuffled <= 0.01 -> the trained rule does not transfer to own confidence.

## Exp E: training on own calibrated confidence (own vs shuffled-confidence arm, 3 seeds each)
Primary: post-challenge accuracy on held-out items under the calibrated own-confidence note,
own arm vs shuffled arm, and own arm vs base.
Prediction P3: own-arm - shuffled-arm >= 0.03, with a CI excluding 0.
Kill: difference <= 0 -> gains come from latent knowledge, not the stated-confidence channel.

## Addendum (2026-09-26 01:57 ET; before any Exp B/C/E/A/D output was read)
Result-slot decision rules adopted from the PI's rewrite plan (wording follows data):
- H/U (reanalysis): thesis wording if the rate-matched use share U < 0.3 in most checkpoints and U is
  flat or falling on the Qwen ladder; if U >= 0.5, narrow to "latent belief targets switches, stated
  confidence adds nothing".
- Exp B: thesis wording if |far - near| < 0.05 with the CI inside +/-0.08; if >= 0.10, reframe as
  "models act on strong belief but not stated confidence" (Yang & Jia central).
- Regression R1: thesis wording if the stated-confidence AME is < 0.02 in both specs; otherwise
  report the larger value and drop "under R1" from the abstract. (Note: the reanalysis before this
  addendum already found the signed-confidence AME = -0.129/SD observable-only, so the fallback applies.)
- Exp A: thesis wording if the largest effect is < 0.08 with no monotone trend; if a monotone effect is
  >= 0.10, cut "not an access failure".
- Exp C: own-calibrated beats shuffled with a CI excluding 0; otherwise report headroom only.
- Exp E: own beats shuffled (CI excluding 0) in every family run, and beats base; otherwise retitle 6.2
  "The rule does not yet transfer to self-report".
- Exp D: within 1 point of the control on every task; otherwise state the per-task cost.
