# Research plan: remaining experiments for ICLR 2027 (temporary folder; delete before camera-ready)

Last updated 2026-09-21. Owners: V = author V, A = co-author A. Deadline: full paper 2026-09-25 23:59 AoE; rebuttal window later
carries anything marked "rebuttal". Everything here is frozen in `prereg/` before it runs; this file is the map, not the contract.

## 0. Thesis and what the remaining work must prove

Paper thesis (from `paper/PAPER.md`): whether a model keeps or drops a challenged answer is governed by conversational
surface cues (an alternative is mentioned, doubt is expressed, the cue is recent), not by epistemic content (own calibrated
confidence, evidence in the challenge). The measurement study (13 experiments, 19 checkpoints, ~400k trials) establishes
this. Three things are still asserted rather than shown, and they are the difference between a poster and a spotlight:

| Gap | Current status | Closing experiment |
|---|---|---|
| Preference optimization *causes* the pressure-capitulation jump | observational across released checkpoints; synthetic DPO test was too weak (null) | E2 real-recipe DPO |
| A fix exists that covers the mention attack *without* capability loss | FIRM removes doubt-capitulation but still folds 93-100% to "another source says X" and costs 12-21 accuracy points | E3 STAND (GRPO, verifiable reward) |
| The failure matters beyond one user talking to one model | asserted in the intro | E1 answer contagion between agents, E4 at the frontier |

## 1. Experiment list (priority order)

### E1. Answer contagion between agents  [V] — RUNNING (Azure CPU), then della GPU replication
Code `harness/contagion.py`; prereg `prereg/PREREG_contagion.md` (P1-P5). Cells: pairwise (who/what the peer is), chain
(k=1..8, Markov prediction vs measured), firewall (trained agent at position 2). Models: Qwen2.5-7B/14B, Llama-3.1-8B,
Mistral-7B, OLMo-2-7B; N=300 held-out questions. Status 2026-09-21: Qwen-7B pairwise 75% done on Azure CPU (Q8 weights);
early read: generated peer messages fold 0.74-0.76 regardless of peer strength (P1 holds); scripted mention 0.47 vs
no-message 0.31 vs reasoned 0.62 (P2 borderline). Della bf16 replication: `bash slurm/submit_contagion.sh` (5 jobs x 20 min).
Deliverable: Fig. "contagion" (pairwise bars + chain curve with Markov overlay), Table of P1-P5 outcomes per model.

### E2. Real-recipe DPO installs the defect  [V on della / A on ionic] — DEADLINE
`train/dpo.py` arm A1: allenai/llama-3.1-tulu-3-8b-preference-mixture (10k pairs), LoRA r64, lr 5e-6, beta 0.1, 2 epochs,
on OLMo-2-7B-SFT and Tulu-3-8B-SFT, 3 seeds. Eval with the frozen harness + capability suite. Prediction (prereg H1):
pressure_abandon rises >= 0.15 over A0 on both backbones. This turns Section 6's "preference-optimized descendants show..."
into "preference optimization with the public recipe causes...". 12 jobs, ~18 GPU-h. Submit: `bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"`.

### E3. STAND: a fix that covers the mention attack at zero capability cost  [V] — DEADLINE if GPU by 09-23, else rebuttal
`train/grpo.py` arm A3: GRPO, G=8, reward = post-challenge correctness (+0.1 format), 20% plain-QA replay, LoRA r64, on the
same two backbones, 3 seeds; A2 (DPO on the same revision dialogues) as the "is RL necessary" control. Prereg H2/H3:
retain_correct >= 0.6 with accept_valid_correction >= 0.7 and MMLU/GSM8K/IFEval within 1 point. Contrast with FIRM
(`experiments/train_firmness.py`): FIRM = doubt-only coverage, -12 to -21 accuracy; STAND must beat both. 24 jobs, ~70 GPU-h.
Also feeds E1's firewall cell (`FIREWALL=checkpoints/olmo/A3/merged bash slurm/submit_contagion.sh olmo_fw`).

### E4. Frontier contagion and decomposition via API  [V, laptop] — DEADLINE
`harness/contagion.py --backend api` and `harness/run_cells_v17_api.py` on claude-fable-5-1, gpt-6-astra, gemini-3-pro,
deepseek-v4; N=150; peer messages reused from `results_contagion/_peers`. Prereg P5: fold(mention) >= 0.30 in >= 2 of 3.
Needs API keys; ~$60-80 per model. This is the single largest lever on "frontier relevance".

### E5. FIRM repair  [A] — DEADLINE, 2 GPU-h
Re-parse `results_firm/firm_s0` with the fallback parser (21% unparsed; the 0.15 pressure number rests on 20 trials) and add
seed 2 so the causal both-directions claim rests on 3 seeds with n >= 200 each.

### E6. Scale rung  [A/V] — rebuttal
`bash slurm/measure_stages.sh olmo13 olmo32` (published 13B/32B SFT/DPO/Instruct, eval only, ~12 GPU-h) then A1/A2 at 32B
(`--backbones olmo32 --arms "A0 A2 A1"`, 4x80GB). Prereg H7/H8.

### E7. Statistics and grading validity  [V] — DONE / DEADLINE
DONE: `analysis/meta_robustness.py` (Hartung-Knapp CI [-0.54,-0.22]; all leave-one-family-out exclude 0). DEADLINE: LLM-judge
agreement on 500 stratified trials (`train/judge_check.py`, ~$10) so substring grading has a reported error bar.

### E8. Re-ask reframing  [A, no compute]
Report the re-ask result under sampling (temperature 0.7, 5 samples) or reframe as "knowledge intact, dialogue state removes
it"; under greedy decoding the 1.00 reads as tautological.

## 2. Compute plan

| Where | What | Wall clock | Cost |
|---|---|---|---|
| Azure CPU D64s_v7 (credit) | E1 open models, Q8 | ~3.5 h/model, 4 models | ~$50 |
| della (A's allocation) | E1 bf16 replication, E2, E3, E5, E6 | E1 2 h; E2 6 h on 3 GPUs; E3 24 h on 3 GPUs | 0 |
| Laptop + API keys | E4, E7 judge | 4-6 h | ~$300 |
| Azure GPU | none: sponsored subscription is denied every GPU family (docs/AZURE.md) | | |

Aggregate GPU-hours if everything runs: ~135 (E1 3, E2 18, E3 70, E5 2, E6 40). On 4 parallel A100s: ~1.5 days wall clock.

## 3. Timeline to the deadline

- 09-21 (Mon): E1 Azure run completes overnight; E7 done; plan + dashboard pushed. A: E5, start della E1 + E2.
- 09-22 (Tue): E4 frontier (needs keys). E2 numbers. Write Section 7 (contagion) and the causal paragraph of Section 6.
- 09-23 (Wed): E3 if della has the GPUs; otherwise mark rebuttal. Figures final: contagion, DPO-causal, STAND-vs-FIRM frontier.
- 09-24 (Thu): full draft; abstract rewritten to what landed; reviewer-threat pass (`docs/ASSESSMENT_2026-09-18.md` list).
- 09-25 (Fri): buffer, submit.

## 4. What a spotlight requires (see plan/VENUE_BAR.md)

One retellable result plus airtight method. Candidate sentence: "A single mentioned alternative propagates through a chain of
agents exactly as the pairwise rates predict, frontier models are as susceptible as 7B models, and a small agent trained with
a verifiable reward stops the cascade at zero capability cost." E1 + E4 + E3 are that sentence; E2 makes the cause claim
causal; E5/E7/E8 remove the three objections a careful reviewer would raise first.
