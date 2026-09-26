# Paper outline v2 (2026-09-21): how the pieces flow

Working title: **Cues over Content: Conversational Surface, Not Evidence, Governs Answer Revision in Language Models**
(the co-author's title kept; the theory section and the agent section are added; the measurement repair from the audit is applied.)

| § | Section | Claim it makes | Evidence | Status | Owner |
|---|---|---|---|---|---|
| 1 | Introduction | One sentence: revision follows surface cues, not evidence or own reliability; four contributions | — | draft exists (A); rewrite intro paragraph to name the model (§3) and the agent consequence (§7) | A + V |
| 2 | Related work | Sycophancy, conformity, calibration, multi-agent failure; what each leaves open | — | draft exists (A); add the 2026 multi-agent papers (plan/VENUE_BAR.md) | A |
| 3 | **A model of answer revision** | Revision policy π(a,c); behavioral use U_z as a matched contrast; Bayesian benchmark; Prop. 1; corpus-conditional mechanism; six derived predictions; propagation dynamics | paper/sections/theory.md | drafted (V) | V |
| 4 | Setup | Banks, protocol, decomposition, metrics (now defined via §3), statistics, pre-registrations, seed rule | — | draft exists (A); add seed rule + harmful/beneficial reporting convention | A |
| 5 | **Surface, not evidence** (P-surface, P-recency) | Naming an alternative does the work; praise acts like pressure; source vs position reversal | v17 decomposition (harmful-only recompute), annotation cells, matched-position meta (DL + HK + LOO) | measured; needs the audit's recompute | A (+V stats) |
| 6 | **The calibration–use gap** (P-flat, P-surfacing) | AUROC(r→t) rises with scale; U_r ≈ 0 under matched conditions; surfacing is null | v16 ladder (elicited association and inserted effect on SEPARATE axes), forced_conf2, identification wave (3-cell, 4 signals) | measured; figure must be redrawn per audit; identification wave running (ionic) | A |
| 7 | **Training installs it and can remove it** (P-training) | Stage jump at DPO; real-recipe DPO causes it; FIRM/POISON both directions; STAND / D2 fix with capability cost curve | stage ladder; A1 (della); FIRM (3 seeds after repair); A3/A2 (della, 3 seeds); D2 (ionic) | stage ladder measured; A1/A3/A2 queued; D2 running; FIRM seed 2 pending | V (A1, A3, A2) + A (FIRM, D2) |
| 8 | **Propagation between agents** (P-propagation) | Receiver folds regardless of sender competence; chains follow the two-state dynamics; stationary error set by drift; a trained agent resets the chain | contagion pairwise (injected + genuine), chains (merged protocol), firewall, 32B/72B, frontier APIs | 3 models done (Q8), OLMo + reruns running, bf16 replication + firewall queued on della, frontier pending keys | V |
| 9 | Two uses of the pathology | Survival-as-calibration; re-ask under sampling (D4) | existing + D4 | measured; re-ask needs D4 or reframing | A |
| 10 | Limitations, negative results, pre-registration receipts | Steering null; prereg failures (self/other reversal; chain Markov hop-1); Q8 vs bf16; topologies not measured; 7–72B open + 2–4 API models | — | write last | V + A |

## The spine in three sentences
Revision is governed by visible surface cues rather than by the evidence a challenge carries or the model's own reliability
(§5–6), because the trained policy is the corpus conditional on visible context (§3). Preference optimization sharpens that
policy toward capitulation and a correctness-rewarded stage reverses it without capability loss (§7). Because a forwarded
answer is itself a surface cue, the same policy makes answer-passing pipelines converge to an error rate set by drift alone,
and a single retrained agent resets them (§8).

## Figures (target: 7 in main text)
1. Decomposition bars, harmful-only, 5 anchor models (§5). 2. Matched-position reversal forest plot with DL and HK intervals (§5).
3. The gap: AUROC(r→t) vs U_r across the Qwen ladder, two panels, never one axis (§6). 4. Identification 3-cell result (§6).
5. Retain/accept frontier with arrows from SFT base to A1, A2, A3, FIRM, D2; capability deltas inset (§7).
6. Contagion: pairwise bars by peer kind + chain curves with the closed-form fit and the firewall reset (§8).
7. Frontier panel: the same cells on Fable 5.1 / GPT-6 Astra next to the 7B–72B open models (§8).

## Section 8 draft (numbers as of 2026-09-21; replace Q8 with bf16 when della returns)
See paper/sections/agents.md.
