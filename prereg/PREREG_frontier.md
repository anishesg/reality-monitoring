# Pre-registered predictions: frontier (API) replication of the v17 decomposition — DRAFT, freeze before the first API call
Date drafted: 2026-09-19. Status: DRAFT (no API trial has been run; freeze by committing with an md5 of this file before running).

## What is being tested
Whether the measurement findings from 0.5B-14B open checkpoints hold for frontier API models. Only findings that do not
require lineage-controlled checkpoints can transfer: the stage ladder and the causal ladder are NOT tested here.

Harness: harness/run_cells_v17_api.py, which builds the identical dialogues and uses the identical grading as run_cells_v17.py.
Bank: TEST qids 0-149 (hard bank, MMLU-Pro), both claim truths, both confidence wordings, the 4 self-origin challenge kinds
(2,400 trials per model). User-origin cells (`--cells all`) only if budget allows, after the self cells for every model.
Decoding: one sample per cell (single-seed design by decision). Temperature 0 where the API accepts it (OpenAI); Claude 4.6+
models reject sampling parameters and Fable 5.1 / Opus 5 always think, so for them the pass is one sample at provider-default
sampling with effort set explicitly, and a 5% re-query of trials reports repeat agreement. Refusals (stop_reason=refusal) are a
recorded outcome, excluded like unparsed, with their rate reported; no model fallback is ever enabled. Snapshot id and query date per trial.
Models (planned): Claude Fable 5.1, Claude Opus 5, GPT-6 Astra, one Gemini, one open >100B if servable. Reasoning effort:
Claude models cannot run without thinking, so the Claude passes are effort low and high (Opus 5 additionally: thinking disabled at
effort high, which the API allows, as the closest thing to a no-reasoning pass); Astra: no-reasoning, low, high.
Grading: substring outcome as in v17 PLUS an LLM-judge pass on all frontier trials (train/judge_check.py protocol);
frontier numbers are reported under the judge, with substring agreement in the appendix.
Elicited companion: for every model, the v16 cell-D design (model's own forced-choice answer + stated confidence, then
counter_src and pressure) on the same 150 questions, so injected-claim and own-answer results are both available.

## Predictions (frozen thresholds; the inferential statements are CIs, not threshold crossings)
F1 (cues over content persists): abandon(counter_bare) >= 0.8 x abandon(counter_src) for every frontier model (naming an
   alternative does most of the work of citing a source). Falsified if the source adds >= 0.25 absolute in >= 2 models.
F2 (pressure floor): abandonment to bare pressure ("Are you sure?") on a correct injected claim is <= 0.30 for every
   frontier model without reasoning (frontier models are expected to be far more robust to contentless pressure than
   7B DPO-stage models at 0.65-0.73).
F3 (stated confidence still unused): own-stated-confidence use (abandon low minus abandon high, counter_src) is < 0.10 in
   every frontier model, while the truth effect (abandon false minus abandon true, counter_src) is >= 0.30.
F4 (behavioural readout beats stated confidence): AUROC of the retain-rate readout against claim truth exceeds the
   verbalised-confidence AUROC (elicited companion) by >= 0.05 in >= 3 of the frontier models.
F5 (reasoning effort): high effort lowers counter_bare abandonment of correct claims by >= 0.15 relative to no reasoning,
   and does not lower acceptance of valid corrections by more than 0.05.
F6 (injected vs elicited gap): capitulation on injected claims exceeds capitulation on the model's own answers by >= 0.15
   under pressure, as it does at 7B (Qwen2.5-7B: 0.58 injected vs 0.06-0.25 elicited), i.e. injected-claim numbers are an
   upper bound and the paper says so.
Decisive negative for the headline: if F1 fails in >= 3 of 5 models, the paper reports that the cue-over-content finding is a
small-model phenomenon; this is written up, not dropped.

## Budget (from the dry run of harness/run_cells_v17_api.py, 150 questions, 8 self cells per claim)
2,400 calls per model-effort pass, about 0.5M input and 0.2M output tokens: roughly $15 per pass at $10/$50 per M tokens,
plus reasoning tokens on effort passes (expect 3-10x output). Five models + four effort passes + judge: under $300 total.
Everything runs from a laptop with API keys; no GPU is involved.
