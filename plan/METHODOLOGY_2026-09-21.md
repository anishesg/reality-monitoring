# Methodology brief for the co-author discussion (2026-09-21, evening)

Two designs now exist in the repo and they are not yet reconciled. This brief lays out what has been run, what each design
proposes, where co-author A's audit bites, and a merged GPU plan with the decisions that have to be made before jobs are submitted.

## 1. What has been run and what it showed

### 1a. Measurement study (co-author A; 19 checkpoints, ~400k trials; `results/`)
Thirteen experiments: fragility with a scale plateau; challenge decomposition; praise-as-pressure; the scissors (confidence
AUROC rises with scale, behavioral use stays ~0.05); the pre-registered matched-position reversal (family meta mu = -0.38);
surfacing null; survival-as-calibration (16/19); re-ask vs reconsider; the reliability-conditioned SFT repair (3 seeds,
-20 pts capability); stage localization (DPO-stage jump in 2/3 lineages); FIRM/POISON both-directions SFT (2 seeds);
weak-DPO null; steering null.

### 1b. co-author A's independent audit of that study (`docs/reviews/2026-09-21/`, today)
Three issues it calls blocking, verified against committed trial files:
1. The headline "scissors" figure puts an AUROC of *elicited numerical* confidence next to a behavioral effect of *inserted
   qualitative* confidence phrases. Different estimands; the 0.05 comes from cell E, the AUROC from cell A.
2. The decomposition table pools true and false inserted claims and both confidence phrases. Recomputed harmful-only
   abandonment is lower than reported (Qwen-7B: sourced .90, bare .62, source-only .50, pressure .40 vs the draft's
   .95/.79/.66/.58). Every "own answer" in those cells is inserted, so retention measures resistance to rejecting supplied
   text, not stability of something the model generated.
3. The surfaced/latent comparison inserts the "surfaced" number rather than having the model generate it, assigns one
   reliability value per item (curves compare different item sets), and the SFT control is not byte-identical.
The audit's recommendation: narrow the paper to "Cues over Confidence: reliability use in answer revision", run one matched
experiment with one primary interaction (Block 1) and one intervention that changes reliance on the signal (Block 2), and keep
scale, training history, speaker attribution, repeated-challenge calibration, and multi-agent chains OUT of the main paper.

### 1c. Statistics repair (author V, done; `analysis/meta_robustness.py`)
Hartung-Knapp CI for the matched-position reversal [-0.54, -0.22]; every leave-one-family-out interval excludes zero. This
closes the k=6 objection regardless of which framing wins.

### 1d. Answer contagion between agents (author V; Azure CPU, Q8 weights; `results_contagion/`, prereg P1-P5)
Three models complete (9,600 trials each), OLMo running, chains being re-run under a fixed protocol (see caveat).

| model | no message | scripted mention | + reason | generated peer (weak / same / strong) | identity spread (P1 <= .10) | accept true alt | contaminated chain k1 -> k8 | clean chain k1 -> k8 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | .29 | .50 | .61 | .80 / .80 / .80 | .001 | .81-.85 | .88 -> .80 | .10 -> .26 |
| Llama-3.1-8B | .42 | .73 | .84 | .80 / .89 / .84 | .09 | .80-.83 | .85 -> .82 | .29 -> .67 |
| Mistral-7B | .46 | .71 | .77 | .89 / .88 / .79 | .10 | .63-.70 | (rerun) | (rerun) |

Findings so far: who the peer is does not matter (P1 holds in 3/3 at the threshold); a reason adds <= .11 (P2 holds); a
contaminated chain never self-corrects (80% wrong at hop 8); a *clean* Llama chain drifts to 67% wrong by hop 8 with no
adversary. The pre-registered Markov prediction fails as specified (hop 1 receives a bare seed, hops 2+ receive full replies
that are near-absorbing at .98); the two-phase version from the measured hop-1 state fits (8/8 and 6/8 hops inside CI),
reported as post hoc. Caveats: pairwise cells use *injected* priors (the audit's issue 2 applies to them too); chain hops
>= 2 are genuine generations and do not have this problem; Mistral's chains failed on a template restriction (two consecutive
user turns) and all four models' chains are being re-run with a merged single-turn protocol for consistency.

## 2. The two designs

### Design V (author V, `plan/RESEARCH_PLAN.md`, `train/`, `harness/contagion.py`)
Thesis extension: surface cues drive revision -> the failure propagates between agents -> preference optimization installs
it -> a verifiable-reward fix removes it without capability cost.
- E2 real-recipe DPO (A1): Tulu-3 preference mixture, r64, 3 seeds x 2 SFT backbones -> "DPO causes" instead of "is associated".
- E3 STAND (A3): GRPO with reward = post-challenge correctness, 20% replay, 3 seeds x 2 backbones; A2 (DPO on the same
  dialogues) as the "is RL needed" control. Target: covers the mention attack, capability within 1 pt (FIRM: doubt-only, -12 to -21).
- E1 contagion (running) + firewall cell with STAND; E4 frontier via API; E6 13B/32B stage rung.
- Pipeline: `train/build_data.py` -> `train/dpo.py` | `train/grpo.py` (+`reward.py`) -> merge -> `train/eval_arm.py`
  (frozen v17 harness + MMLU/GSM8K/IFEval) -> `results_ladder/<tag>/<arm>/summary.json`; `slurm/submit_ladder.sh` chains
  train -> eval with afterok; per-size GPU defaults (7B: 1x80GB; 32B: 4x80GB + vLLM server for rollouts).

### Design A (co-author A, `della/DELLA_SPEC.md`, `experiments/run_identification.py`, `train_repair_v3.py`)
Thesis narrowing per the audit: "Confidence Without Control" / "Cues over Confidence".
- Identification wave (running on ionic): 3-cell design per item (claim T/alt d1; claim d1/alt T; claim d1/alt d2, the
  identification cell) x 5 counter paraphrases + pressure; 4 uncertainty signals in clean context (token logprobs, verbalized
  confidence + FC margin, P(True), sample consistency k=8). On Qwen2.5-7B/14B, Llama-8B, OLMo-7B, Qwen3-8B, R1-Distill-7B.
- D1 unquantized scale ladder to 70B on that one protocol (~10 A100-h).
- D2 repair-v3 (the "oral-maker" per the spec): reliability-conditioned SFT with real Tulu-3 replay, lr x replay-frac x
  3 seeds + byte-identical controls, external MMLU/GSM8K capability eval; deliverable = conditioning-strength vs capability-cost
  frontier (~36 A100-h). Already started on ionic with low-priority A100 access.
- D3 open-ended (non-MCQ) replication (~6 A100-h); D4 legitimate re-ask with paraphrases and sampling (~4 A100-h).

## 3. Where they agree, where they conflict

Agree: the paper needs (i) a measurement repair (harmful vs beneficial split, elicited vs inserted kept apart), (ii) one
causal intervention with a capability-cost frontier, (iii) evidence beyond 14B or beyond open weights. Both designs use the
same harness, banks, and grading; both are pre-registered or can be.

Conflict 1, the thesis. Design A narrows to reliability use in revision and explicitly drops multi-agent, training history,
and scale from the main paper. Design V broadens to propagation and a training cause + cure. These are two papers if both
are pursued fully. The audit is right that the current draft's headline needs repair first; it is also true that the
contagion results are the only new result in the repo this week that a reviewer would retell, and they came in cheaply.

Conflict 2, the intervention. D2 (SFT that teaches obedience to a supplied reliability value, with replay to protect
capability) vs A3 (RL with a verifiable correctness reward, no reliability metadata needed). D2 answers "can dependence on the
signal be installed at zero cost"; A3 answers "can correct-answer retention be installed at zero cost". They are not
substitutes; D2 matches the narrowed thesis, A3 matches the broadened one.

Conflict 3, injected priors. The audit's issue 2 applies to my pairwise contagion cells and to the v17 cells alike. A
genuine-answer version of the pairwise cell (model generates its own answer first; only initially-correct items enter the
fold cell; only initially-wrong items enter the accept cell) is a 20-minute change to `harness/contagion.py` and should be
run before the paper cites the pairwise numbers. The chain cells from hop 2 already are genuine.

## 4. Proposed merged plan (to decide with co-author A before submitting GPU jobs)

Priority for the 09-25 deadline, in order:
1. Measurement repair (zero GPU, co-author A): harmful/beneficial split of every table; elicited and inserted confidence never on
   the same axis; item-crossed reliability values. This is required whichever thesis wins.
2. Block 1 / identification wave (ionic, running) — the matched experiment the audit asks for.
3. ONE intervention with the capability frontier. Decision: D2 or A3 first. Recommendation: D2 first because it is already
   running and matches the narrowed thesis; A3 second on della (24 jobs, ~70 GPU-h) because it is the only candidate for a
   fix that needs no supplied metadata, and it feeds the contagion firewall cell.
4. Contagion, genuine-answer pairwise version + merged-turn chains (CPU, running) — placed in the paper as Section 7
   "consequence for agent systems" (one figure, half a page) if the thesis stays broad, or as an appendix + follow-up paper
   if it narrows. This is the decision co-author A and author V need to make explicitly.
5. E2 real-recipe DPO (12 jobs, ~18 GPU-h) — only if the training-history claim stays in the paper.
6. D1 unquantized ladder / E6 stage rung — rebuttal.
7. Frontier API runs (E4) — cheap, laptop, run regardless; strengthens either framing.

GPU jobs to submit on della once the intervention decision is made: D2 (24 x 1.5 h) or A3 (24 x 3 h) first, then E1 bf16
contagion replication (5 x 20 min), then E2 (12 x 1.5 h). Everything needed is in `slurm/` (author V) and `della/` (co-author A);
the two directories use different conventions and should be unified by whoever submits.

## 5. Where this stands against the bar (plan/VENUE_BAR.md)
Accepted alignment papers 2025-26 that reach spotlight carry a mechanism, a causal intervention with cost accounting, and
models people use. After the measurement repair + one intervention frontier + identification wave, the paper meets the
"careful, pre-registered, causal" bar the ICML 2026 oral explicitly asks for. The contagion result is what would make it
retellable. Estimate: accept ~40-50% after items 1-3; spotlight plausible only if the intervention frontier has a point at
(full conditioning, ~0 cost) or the contagion firewall works on a genuine-answer protocol.
