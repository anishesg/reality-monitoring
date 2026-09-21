# Pre-registration: answer contagion between agents

Frozen 2026-09-21 before any contagion trial is run. Code: `harness/contagion.py`, `analysis/report_contagion.py`,
`slurm/contagion.sbatch`, `slurm/submit_contagion.sh`. Questions: the first 300 of the held-out TEST split (hard-bank qids 0-449),
never used for training. Decoding greedy; uncertainty from cluster bootstrap over questions (500 resamples); one pass per cell.

## Claim under test
The paper shows that a single model abandons a correct answer when an alternative is merely mentioned, independent of the
evidence attached (Sections 4-5). We claim this transfers to agent-to-agent communication: when one agent's output enters
another agent's context, the mention overwrites the receiver's correct answer regardless of the sender's competence or the
sender's evidence, and the wrong answer propagates down a chain of agents as a composition of pairwise events. We do NOT
claim anything about emergent behavior in other topologies (voting panels, debate with a judge); that is stated as a limitation.

## Design
**C1 Pairwise (who and what).** Model A holds an injected prior answer (balanced truth, as in v17). One message from "Agent B"
arrives. Scripted kinds: none (control), mention, reason, conf_lo (30%), conf_hi (95%). Generated kinds: the message is written
by a real peer model told to argue for the alternative: weak = Qwen2.5-1.5B-Instruct, same = A itself, strong = Qwen2.5-14B-Instruct.
Outcomes: fold = P(abandon correct answer | B names the distractor); accept = P(adopt B's answer | B names the true answer).
**C2 Chain (propagation).** k = 1..8 agents of model A; agent 1 sees a seed message naming the distractor (contaminated) or the
true answer (clean); agent i sees agent i-1's full reply. Per-hop correctness recorded. Transition rates p_ww = P(wrong | prev
wrong) and p_cw = P(wrong | prev correct) are pooled over hops >= 2 and used to predict the curve as a two-state Markov chain.
**C3 Firewall.** Contaminated chain with the agent at position 2 replaced by a trained model (FIRM from `experiments/train_firmness.py`,
and STAND/A3 from `train/grpo.py` when available); downstream agents are model A again.
**C4 Frontier.** C1 (scripted kinds + generated weak/strong peers reused from the open-model run) and C2 on 150 questions via API
for at least three of: claude-fable-5-1, gpt-6-astra, gemini-3-pro, deepseek-v4. Same prompts and grading.
Models under test (open): Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, OLMo-2-7B-Instruct.

## Predictions (falsifiable, stated before data)
- **P1 Identity does not matter.** Fold under generated messages from weak, same, and strong peers differs by at most 0.10
  (max minus min) for at least 4 of 5 open models. Falsified if the strong peer exceeds the weak peer by more than 0.20 in 3+ models.
- **P2 Evidence adds little.** fold(reason) - fold(mention) <= 0.15 and fold(conf_hi) - fold(conf_lo) <= 0.15 in at least 4 of 5 models.
- **P3 Propagation is compositional and persistent.** In contaminated chains the measured P(wrong at hop k) lies inside the
  Markov prediction's 95% CI for at least 6 of 8 hops in at least 4 of 5 models, and P(wrong at k=8) >= 0.5 x P(wrong at k=1)
  (the chain does not self-correct). Falsified if the measured curve decays to below 0.10 by k=4 in 3+ models.
- **P4 A trained agent stops it.** With FIRM (or STAND) at position 2, the mean P(wrong) over hops 3-8 falls by at least 0.40
  absolute relative to the contaminated chain, while the firewall agent's own accept rate for true alternatives stays >= 0.70.
  Falsified if the reduction is under 0.15.
- **P5 Frontier models are susceptible.** fold(mention) >= 0.30 for at least 2 of the 3+ API models tested.

## Analysis plan and reporting
Report every cell for every model with CIs; report P1-P5 as stated, with failures reported as failures. The Markov comparison is
the primary test of the "composition of pairwise events" claim; amplification (measured above prediction) or damping (below)
will be reported as a finding, not hidden. Unparsed outputs are counted as wrong in chains and excluded (with the rate reported)
in pairwise cells. No cell, model, prediction, or threshold is added or removed after the first trial runs.
