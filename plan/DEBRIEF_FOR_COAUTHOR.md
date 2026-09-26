# Debrief: what I (author V) am running, why, and what I need from you — 2026-09-21

**One-paragraph version.** Your measurement study establishes that revision follows surface cues, and your audit says the
headline figures need an estimand repair before they can carry the claim. I am adding three things that turn the paper from
a careful measurement paper into a causal one with a mechanism: (1) a formal model (§3) that defines "use of a signal" as a
matched contrast, gives the Bayesian benchmark, states the corpus-conditional mechanism as one assumption, and derives every
section's prediction from it; (2) the training half of §7: the public DPO recipe applied to SFT backbones to show the
behavior is *caused*, and STAND (GRPO with a post-challenge-correctness reward) as a fix that covers the named-alternative
attack with the capability cost measured, alongside your D2; (3) §8, propagation between agents, with a two-state dynamics
model whose parameters are measured pairwise and tested on chains. Everything is pre-registered (`prereg/`), 3 seeds for
trained arms, one deterministic pass with question-level bootstrap for measured checkpoints, same harness and banks as yours.

## What is already done (mine)
- `analysis/meta_robustness.py`: Hartung–Knapp CI for the matched-position reversal [-0.54, -0.22]; every leave-one-family-out
  interval excludes 0. Closes the k=6 objection.
- Contagion (§8), Azure CPU, Q8 weights, 300 questions x 9,600 trials per model: Qwen-7B, Llama-8B, Mistral-7B complete,
  OLMo running. P1 (sender identity irrelevant) holds 3/3; P2 (evidence adds <= .15) holds; chains never recover (80% wrong
  at hop 8); clean Llama chains drift to 67% wrong. Prereg Markov prediction failed as specified (hop-1 seed differs from
  forwarded replies); two-phase version fits; reported as a correction. Your audit's injected-prior point applies to the
  pairwise cells; a genuine-answer version is queued. Chain hops >= 2 are genuine generations.
- `paper/sections/theory.md` (§3), `paper/sections/agents.md` (§8 draft), `paper/OUTLINE_v2.md` (how it all flows).
- Your v2 della package imported at `della/v2_epistemic_ladder/` with two fixes: matched-position origins (your own v17
  control shows the unmatched self-vs-user contrast is recency) and the harmful/beneficial split from your audit. Token not
  imported; please rotate it.

## What is queued on della (one command: `bash slurm/submit_everything.sh`), ~175 GPU-h
1. Contagion bf16 replication, 5 models (2 GPU-h). 2. STAND (A3) + revision-DPO control (A2), 3 seeds x {OLMo-2-7B-SFT,
Tulu-3-8B-SFT}, frozen v17 eval + MMLU/GSM8K/IFEval (55–70 GPU-h). 3. Firewall: STAND agent in the chain. 4. Real-recipe DPO
(A1), 3 seeds x 2 (18). 5. 13B/32B stage checkpoints (12). 6. Contagion at 32B/72B unquantized (6). 7. STAND at 32B (25).
8. A2 sweep on DEV, selection rule pre-registered (8). 9. Your v2 ladder on anchors + 32B/72B (25).
Steps 1–4 are the deadline set; 5–9 are rebuttal-strength.

## Where our designs meet, and the three decisions
- D2 and STAND are not substitutes: D2 installs dependence on a supplied reliability value; STAND installs retention of
  correct answers under any cue with no metadata. Both go in §7 with their capability-cost curves. Decision 1: agree.
- Your audit recommends keeping multi-agent out of the main text. My case for one half-page in §8: it is the only result this
  week a reviewer will retell, it is derived from the same model as §5–6, and it is what the 2026 multi-agent papers
  (plan/VENUE_BAR.md) lack. Decision 2: main text (half page + Fig. 6) vs appendix. I propose main text if the genuine-answer
  cells and the bf16 replication agree with the Q8 numbers.
- Decision 3: the audit's measurement repair (harmful/beneficial split, elicited vs inserted on separate axes) is yours; it
  gates §5–6 and must be done before the 25th. I can take the figure code if you send the recomputed tables.

## Authorship
I am asking for shared first authorship on the basis of §3 (theory), §7's causal/fix half (A1, A2, A3), §8 in full, the
statistics repair, and the frontier replication. You own §1–2, §4–6, §9, D2, FIRM, and the audit-driven repair.
