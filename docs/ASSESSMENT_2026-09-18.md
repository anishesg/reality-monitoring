# Reality Monitoring: state of the project, gaps, plan, and ICLR 2027 odds
*Written 2026-09-18 after a full read of anishesg/reality-monitoring (2 commits, ~1.7k lines of code, 17 checkpoints × 10,800 v17 trials).*

**Deadlines (ICLR 2027):** abstract **Sept 18, 11:59 PM AoE** (today), full paper **Sept 25, 11:59 PM AoE**. Reviews Nov 5, decisions Dec 16. Source: iclr.cc/Conferences/2027/CallForPapers.

---

## 1. What the repo actually contains

**Nothing has been trained.** Every result is a *measurement* of off-the-shelf checkpoints. The "DPO" and "RLVR" in the README refer to published post-training *stages* of open lineages (OLMo-2-7B SFT→DPO→RLVR, Tulu-3-8B SFT→DPO→RLVR, Zephyr-7B SFT→DPO), not to any fine-tuning done by us. There is no training script, no preference-data builder, no reward function, and no trained checkpoint anywhere in the repo.

Three experiment generations exist:

| Version | Bank | Design | Status |
|---|---|---|---|
| v1 `run_experiment.py` | SciQ (600) | open-ended answer, origin×conf×truth×challenge factorial | Abandoned. Open-ended baseline parsing broke (OLMo "accuracy" 0.06, Qwen-7B 0.012 = parse failure, not knowledge). |
| v15 `run_cells_v15.py` | SciQ | elicited own answer+confidence, P(True), memory of stated conf, prompted rescue | Results not in repo (only referenced). |
| v16 `run_cells_v16.py` | MMLU-Pro 600 + TruthfulQA 300, forced choice | model's *own* answer + inline/post-hoc confidence + P(True), then challenged; plus injected-confidence cells | 19 checkpoints, but **phi35 and qwen32b have n=0 (failed), llama1b has n=29**. Scale ladder has holes. |
| v17 `run_cells_v17.py` | same hard bank, 450 q | *injected* assistant claim with low/high conf × truth × 4 challenge kinds; user-origin at matched recency | 17 checkpoints, complete. This is the paper-deciding table. |

Plus `mech/probe_steer.py`: difference-in-means "confidence" probe + activation steering on 4 models. Probe accuracy = 1.000 at layer 5 in every model, which means it is trivially reading the literal "I'm really not sure" tokens, not an internal confidence state. No cross-validation, no shuffled-label null. Steering is null. **This should be dropped or redone**; a July 2026 arXiv paper (2609.17550) published exactly the validation protocol that kills naive probes like this one, so a reviewer will know.

## 2. What the data say (v17, self-origin, pooled)

Abandon rate by challenge kind. `counter_src` = "another source says X"; `counter_bare` = "Consider the answer X" (content only, no source); `src_only` = "another source disagrees" (no content); `pressure` = "Are you sure?".

| ckpt | counter_src | counter_bare | src_only | pressure | retain-correct | accept-fix |
|---|---|---|---|---|---|---|
| llama1b | .914 | .904 | .786 | .732 | .13 | .59 |
| llama3b | .936 | .942 | .875 | .875 | .12 | .80 |
| llama8b | .920 | .924 | .873 | .879 | .15 | .86 |
| qwen05 | .985 | .880 | .112 | .383 | .02 | .95 |
| qwen15 | .913 | .844 | .649 | .671 | .10 | .79 |
| qwen7b | .951 | .788 | .664 | .580 | .10 | .97 |
| qwen14b | .949 | .840 | .755 | .765 | .10 | .96 |
| mistral | .890 | .665 | .616 | .596 | .15 | .66 |
| phi35 | .821 | .767 | .764 | .749 | .36 | .80 |
| **olmo_sft** | .828 | .833 | .339 | **.185** | .21 | .77 |
| **olmo_dpo** | .857 | .844 | .735 | **.730** | .25 | .65 |
| olmo (rlvr) | .894 | .844 | .736 | .689 | .17 | .70 |
| **tulu_sft** | .931 | .848 | .496 | **.427** | .12 | .85 |
| **tulu_dpo** | .862 | .823 | .694 | **.655** | .24 | .78 |
| tulu_rlvr | .854 | .838 | .642 | .601 | .25 | .76 |
| zephyr_sft | .975 | .587 | .579 | .688 | .04 | .93 |
| zephyr_dpo | .927 | .781 | .889 | .698 | .12 | .77 |

Robust findings:
1. **Mere mention dominates.** `counter_bare` (no source, no argument) already triggers 60–94% abandonment. The "external source" adds 0–16 pts on top (Qwen, Mistral, Zephyr), ~0 in Llama (ceiling).
2. **DPO stage installs pressure-only capitulation.** OLMo .185→.730, Tulu .427→.655, not repaired by RLVR. Zephyr flat (.688→.698). Note DPO *slightly improves* retain-correct under content counters (OLMo .21→.25, Tulu .12→.24), so the story is "DPO makes the model fold to social pressure without evidence, while marginally better at holding under a stated alternative". That nuance should be in the paper.
3. **Scissors** (v16): verbalized-confidence AUROC rises with scale (Qwen .54→.70) but the behavioral gap between low- and high-confidence retention stays small.
4. **Self-confidence is used almost not at all**; the pre-registered self-discounting claim (P2) failed and reversed (family-level meta μ=−0.38 [−0.51,−0.26]). Effects are tiny in absolute terms.
5. **The value frontier is empty.** No checkpoint is above 0.7 on both retain-correct and accept-valid-correction. Best retain-correct is 0.36 (Phi-3.5).

## 3. Reconciling with "DPO reduces abandonment and increases confidence"

That sentence is **not supported by anything in this repo**. The repo's own headline is the opposite sign for pressure challenges. If your co-author has local training runs that show a *targeted* DPO reduces abandonment, they are not pushed. Before writing the abstract, confirm which of these is true:
- (a) Nothing has been trained yet and the intervention is the plan for this week. → Abstract must be written conditional on results, with the measurement findings as the anchor.
- (b) There are unpushed training results. → Push them; the plan below changes to "scale + control + ablate".

The rest of this doc assumes (a).

## 4. Threats a reviewer will raise (fix these, they are cheap)

1. **Parse/exclusion rates.** v17 unparsed+ambiguous: llama1b 43%, qwen05 21%, phi35 14%, mistral 11%, olmo_dpo 10%. Exclusions are not random (a model that hedges gets dropped). Report per-cell exclusion; drop llama1b/qwen05 from headline tables or fix the parser.
2. **Substring grading.** arXiv 2609.17550 shows substring grading underestimates capitulation by 18–24 pts vs an LLM judge. Run a Claude/GPT judge on a 500-trial stratified sample and report agreement; use the judge for the training-eval if disagreement >5%.
3. **Injected assistant turns.** In v17 the model never actually said "I am completely certain"; we wrote it into the transcript. v16 cell D (model's own answer + own confidence) is the defense — lean on it in the main text, use v17 for the decomposition.
4. **Ceiling effects.** Llama abandons >90% to a bare mention, so any Llama delta is compressed. Say so; report on the logit scale (the GEE already does).
5. **Lineage independence.** OLMo-2 and Tulu-3 DPO stages use the same Ai2 preference-data recipe. That is effectively k≈1.5, not 3. The *causal* replication in §5 is the real fix.
6. **Effect sizes vs claims.** "Self-confidence gets MORE weight than the user's" rests on differences of 0.02–0.20. Frame as "no evidence of self-discounting", not as a positive reversed effect.
7. **Scale.** Nothing above 14B evaluated successfully (32B failed). One 32B/70B run via vLLM TP=2 on della fixes this.
8. **Mech section.** Drop, or redo with question-level CV + shuffled null + positive control. As-is it will be cited as a methodological error.
9. **TruthfulQA "correct" answers** are contested; a source saying otherwise is sometimes right. Report MMLU-Pro and TruthfulQA separately.

## 5. The experiment that turns this into an ICLR paper: a causal post-training ladder

The correlational ladder says "the DPO stage installs capitulation". The paper-making move is to **make it happen and unmake it under our control**, starting from the same SFT checkpoint.

**Backbones:** `allenai/OLMo-2-1124-7B-SFT` and `allenai/Llama-3.1-Tulu-3-8B-SFT` (fully open recipes; the field's cleanest SFT-only starting points). Optionally Qwen2.5-7B-Instruct as an already-RL'd control.

**Arms (each ~8–12k training dialogues, LoRA r=64 or full FT if compute allows):**

| Arm | Data | Purpose |
|---|---|---|
| A0 SFT | none | baseline |
| A1 generic-DPO | 10k pairs subsampled from the actual Tulu-3 preference mixture (no challenge dialogues) | **Causal replication**: does ordinary preference tuning alone raise pressure capitulation? |
| A2 revision-DPO | challenge dialogues; chosen = retain when initial answer correct / switch when wrong; rejected = the opposite. Balanced across the 4 challenge kinds and both origins. | targeted fix, preference-based (PBT-style, but with source/content/pressure decomposition) |
| A3 revision-RLVR (GRPO) | same prompts, reward = post-challenge correctness (verifiable from gold), + small format reward | targeted fix, verifiable-reward RL — tests the "RLVR builds commitment" hypothesis causally |
| A4 confidence-conditioned RLVR | A3 + the model must emit CONFIDENCE; reward = correctness + Brier bonus + consistency (retain iff conf ≥ τ) | closes the "scissors": makes behavioral use track informativeness |
| A5 A1 then A3 | generic DPO followed by revision RLVR | does RLVR *repair* DPO-installed capitulation? (matches the real OLMo/Tulu pipeline order) |

**Training bank:** SciQ (600) + MMLU-Pro slice disjoint from the eval slice. **Eval banks:** held-out MMLU-Pro, TruthfulQA, plus one *out-of-distribution* bank (TriviaQA or GSM8K numeric) to show generalization of the revision policy rather than memorization.

**Primary endpoints (pre-register before training, as before):**
- retain-correct and accept-valid-correction under `counter_src` (the frontier plot). Success = an arm above 0.7 on both.
- pressure abandonment on correct answers (the DPO-installed defect).
- `counter_bare` abandonment (mere-mention priming).
- Capability retention: MMLU 5-shot, GSM8K, IFEval within 1 pt of A0.
- Confidence AUROC and behavioral confidence-use (A4 only).

**Controls:** prompt-only baseline (the v15 G10 "rescue" system prompt), and PBT's released Llama-3.1-8B-PBT checkpoint as an external comparator (huggingface: esteng).

**Compute (della A100-80GB):**
- Data generation: ~2 GPU-hours (vLLM, 7B).
- DPO LoRA, 10k pairs, 2 epochs: ~1–1.5 GPU-hours per arm.
- GRPO LoRA (TRL `GRPOTrainer` with vLLM rollouts, G=8, ~3k prompts × 2 epochs): ~6–10 GPU-hours per arm on one A100; 2–3 h on 4×A100.
- Eval: v17 harness ~15 min per checkpoint; capability suite ~30 min.
- Total: ~50–70 GPU-hours for 2 backbones × 5 arms + evals. Feasible in 3 days with 4 GPUs if the pipeline is written by Sept 20.

**Stack:** TRL (`DPOTrainer`, `GRPOTrainer`), PEFT, vLLM ≥0.6 for rollouts. OpenRLHF/verl are heavier and not worth setup time this week.

## 6. Novelty vs prior work (read these before writing related work)

| Paper | What they did | How we differ |
|---|---|---|
| Sharma et al. 2023 (Anthropic), *Towards understanding sycophancy* | PMs/RLHF favor sycophantic answers; "are you sure" flips | open checkpoints, stage ladder, causal replication via our own DPO |
| Laban et al. 2023, *FlipFlop* | challenge → performance drop | source/content/pressure decomposition |
| Wei et al. 2023 synthetic data; Chen et al. ICML'24 pinpoint tuning; Zhang et al. 2025 Pressure-Tune | SFT-style fixes | RLVR vs DPO comparison, frontier framing, stage-order (A5) |
| **Stengel-Eskin, Hase, Bansal, NAACL'25, PBT** (arXiv 2410.14596) | DPO on dialogue trees to accept good / resist bad persuasion. Llama-3.1-8B/70B, Mistral-7B. | **Closest.** They only have content-bearing persuasion; no pressure-only, no source-vs-content, no confidence, no stage ladder, no RLVR. Use their checkpoint as a baseline and beat/complement it. |
| Mytsyk et al. Aug 2026, BTS-GRPO (2608.25267) | label-free GRPO reward reduces flips 23→4% on 3B | we use verifiable labels (cheaper, stronger), 7–8B, decomposition |
| Saadat & Nemzer Feb 2026, *Certainty robustness* (2603.03330) | 2-turn benchmark on frontier APIs, numeric confidence | per-item confidence×flip linkage, open models, training |
| Aamir & Adil Jul 2026 (2609.17550) | steering null on 1–1.5B, validation protocol, substring-grading hazard | we should adopt their protocol or drop mech; adopt their grading check |
| SycEval, SYCON-Bench, Who Flips (2606.16011), DuET-PD | benchmarks | our benchmark is a means, not the contribution |

**Differentiated claims we can make that none of the above make:**
1. Most "sycophantic" revision is answer-priming by mere mention, not deference to authority or pressure (decomposition).
2. The DPO stage of a standard open recipe causally installs pressure-only capitulation (replicated by training it ourselves from the same SFT), and RLVR in the standard order does not repair it.
3. A verifiable-reward revision stage populates the previously empty retain/accept frontier without capability loss, and (A4) makes revision behavior track the model's own calibrated confidence.

Claim 1 is already in hand. Claims 2–3 require this week's runs. The "reality monitoring" self-vs-user framing should be **demoted to one section**; its pre-registered result failed and the effects are tiny.

## 7. Paper skeleton (9 pages)

1. Intro: capitulation under challenge; the missing decomposition; the missing causal test.
2. Setup: forced-choice hard bank; four challenge kinds × two origins × own vs injected confidence; metrics (retain-correct, accept-fix, pressure-abandon, frontier).
3. Measurement across 15 checkpoints: mere-mention priming (Fig 1), scissors (Fig 2), stage ladder (Fig 3), frontier (Fig 4). Self-vs-user confidence as a subsection with the honest null.
4. Causal post-training ladder: A0–A5 on two SFT backbones (Table 2, Fig 5 frontier movement).
5. Analysis: what RLVR changes (confidence use, OOD generalization, capability retention); prompt-only and PBT baselines.
6. Related work; limitations (template challenges, ≤8B trained, injected turns); pre-registration receipts in appendix.

## 8. ICLR odds, honestly

- **As-is (measurement only):** ~10–15%. Crowded topic, no intervention, sub-14B, template challenges, failed preregistration, weak mech section. Would likely be a workshop paper.
- **With A1–A3 done cleanly on two backbones and the reviewer-threat list in §4 addressed:** ~25–35%. The causal replication of the DPO effect plus a verifiable-reward fix that populates the frontier is a real, citeable result.
- **Risks that push it down:** RLVR fails to move retain-correct (then the paper is "DPO breaks it and we can't fix it", still publishable but weaker); capability regression; running out of time and submitting with one backbone.

## 9. Seven-day schedule

| Day | Deliverable |
|---|---|
| Sept 18 (today) | Submit title + abstract (draft below). Confirm (a)/(b) in §3. Freeze pre-registration for A0–A5. |
| Sept 19 | Write `train/build_pairs.py` (challenge dialogues → DPO pairs + GRPO prompts), `train/dpo.py`, `train/grpo.py`, `train/reward.py`. Smoke on Qwen2.5-0.5B locally/ionic. LLM-judge agreement check on 500 v17 trials. |
| Sept 20 | Launch A1, A2 on both backbones (della). Rerun qwen32b eval (TP=2). Fix llama1b/qwen05 parsing or drop. |
| Sept 21 | Launch A3, A4, A5. Evaluate A1/A2. Write §2–3 of paper with existing figures. |
| Sept 22 | Evaluate A3–A5; capability suite; OOD bank. Fig 5 frontier movement. |
| Sept 23 | Write §4–5, related work, limitations. Ablations if a run failed. |
| Sept 24 | Full draft, internal review, figures polished, appendix with prereg hashes. |
| Sept 25 | Final edits, submit by AoE. |

## 10. Draft abstract (submit today; revisable until Sept 25)

**Title:** Mere Mention, Not Authority: Where LLM Capitulation Comes From and How Post-Training Installs and Removes It

**Abstract.** Language models routinely abandon correct answers when a user pushes back. We ask *what* in a challenge drives revision and *which stage* of post-training installs the behavior. Decomposing challenges into source, content, and bare pressure across 15 open checkpoints (0.5B–14B; Qwen, Llama, Mistral, Phi, OLMo-2, Tulu-3, Zephyr), we find that merely mentioning an alternative answer, with no source or argument, already triggers 60–94% abandonment; attributing it to an external source adds at most 16 points, and the model's own stated confidence changes revision by under 5 points even as its informativeness grows with scale. Along published SFT→DPO→RLVR lineages, the DPO stage sharply raises capitulation to contentless pressure (OLMo-2: 19%→73%; Tulu-3: 43%→66%) and the subsequent RLVR stage does not repair it. No checkpoint retains more than 36% of its correct answers under counter-assertion while also accepting valid corrections. We then test these observations causally by post-training the same SFT backbones ourselves: [generic preference tuning alone reproduces the pressure-capitulation jump, whereas a verifiable-reward revision stage (retain if correct, accept if wrong) raises correct-answer retention from X% to Y% while keeping valid-correction acceptance above Z% and general capability within 1 point, and generalizes to out-of-distribution questions]. Our results locate capitulation in answer-priming and preference optimization rather than deference to authority, and show that it is a fixable artifact of the training recipe. Code, pre-registrations, and checkpoints are released.

*(Bracketed sentence is a placeholder; ICLR lets you revise the abstract until the full-paper deadline. If (b) in §3 holds, fill it now.)*

## Addendum (2026-09-18, later): figures + one correction
- `analysis/figures_paper.py` regenerates Fig 1–4 into `figures/` from the existing aggregates (decomposition, scissors, stage ladder, frontier).
- **Correction to the "scissors" claim.** The README says behavioral confidence use is flat with scale. That is true for *injected* confidence (same items, low vs high wording; Qwen 0.5B→14B: −0.00, −0.01, +0.01, +0.05, +0.04). With the model's *own elicited* confidence (v16 cell D) the low-minus-high abandonment gap grows 0.07→0.31, but low- and high-confidence bins there are different questions, so it is confounded by item difficulty. Fig 2 shows both series and labels the confound; the text must say "controlled confidence use stays near zero" and not overclaim.
- Azure status and guardrails: see `docs/AZURE.md`. Monthly budget removed at user request; lifetime $10k budget with kill switch at 85%/95% remains.
