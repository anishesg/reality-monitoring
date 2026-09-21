# Answer contagion between agents: framing, model, and what the compute buys

## 1. Why it belongs in the paper (one paragraph, usable as the section opener)
Sections A and B show that a single model's decision to keep or drop an answer is driven by surface cues in its context.
Every multi-agent system (debate, self-refine, critic pipelines, tool-using swarms) works by placing one agent's output in
another agent's context. If the per-model result holds, then a wrong answer entering a pipeline should be treated by the
receiving agent as a cue, not as evidence, and should propagate regardless of the sender's competence. This section tests
that prediction directly and gives the quantity a designer needs: the long-run error rate of a pipeline that passes answers
between agents, as a closed-form function of two rates that can be measured pairwise.

## 2. The model
Each hop is a decision by one agent given the previous agent's reply. Let the state be whether the current agent's answer is
wrong. Measure two rates on the model:

- p_ww = P(agent is wrong | the agent it heard from was wrong)   (contamination sticks)
- p_cw = P(agent is wrong | the agent it heard from was right)   (spontaneous drift)

Assumption M (Markov): an agent's answer depends on the previous agent's reply, not on earlier history. This is exactly what
the architecture enforces when only the last reply is forwarded, and it is testable (below).

For a linear chain, with pi_k = P(wrong at position k):

    pi_k = pi_inf + (pi_1 - pi_inf) * lambda^(k-1),   lambda = p_ww - p_cw,   pi_inf = p_cw / (p_cw + 1 - p_ww)

pi_inf is the stationary error rate: every chain converges to it regardless of whether the seed was right or wrong.
lambda is the memory of the chain; the half-life of a seed's influence is ln 2 / (-ln lambda) hops.

Measured (Q8 CPU run, 300 questions):

| model | p_ww | p_cw | lambda | pi_inf | half-life (hops) | seed-wrong chain at k=8 | seed-right chain at k=8 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | .987 | .027 | .960 | .68 | 17 | .80 (pred .83) | .26 (pred .24) |
| Llama-3.1-8B | .975 | .129 | .846 | .84 | 4.1 | .82 (pred .84) | .67 (pred .67) |

Reading: for both models the wrong state is nearly absorbing (p_ww ~ .98), so the long-run error of any answer-passing
pipeline is set almost entirely by p_cw, the drift rate. Llama drifts five times faster than Qwen and its chains converge
to 84% wrong within about four hops from either seed. Qwen converges to the same order (68%) but slowly. Neither pipeline
ever recovers a wrong seed: the recovery rate 1 - p_ww is 1-3% per hop.

## 3. What generalizes beyond a linear chain
- Trees and DAGs with fan-out (one agent's reply read by several): under M each branch is an independent copy of the chain,
  so the expected fraction wrong at depth k is the same pi_k; fan-out multiplies the number of contaminated agents, not the rate.
- Majority vote over m independent chains of depth k: P(majority wrong) = P(Binomial(m, pi_k) > m/2). With pi_k above 0.5,
  voting makes it worse as m grows. This is why "add more agents" is not a fix once pi_inf > 0.5.
- Mixed populations: with a fraction f of agents that are "firewalled" (p_ww lowered to q), the stationary error becomes
  pi_inf(f) = p_cw / (p_cw + 1 - [(1-f) p_ww + f q]); a single firewall agent at position j resets the chain to pi_j ~ q and
  the tail re-converges from there. The firewall cell measures q directly.
- Limits (stated in the paper): M is tested, not assumed (the pre-registered prediction pools hops >= 2; hop 1 differs because
  the seed is a bare message, which we report); no topology beyond chains is measured; results are on 7-8B open models with
  frontier replication pending.

## 4. What each experiment buys, and why it costs what it costs
- Pairwise cells (who/what the sender is): identify that p_ww does not depend on the sender's competence (spread <= .10
  across weak/same/strong peers in 3/3 models) or evidence. ~5k generations per model.
- Chain cells: measure p_ww, p_cw on genuine generations (hops >= 2 are the model's own answers) and test M against the
  measured curve. ~5k generations per model. Together ~1.5-3.5 h per 7B model on 64 CPU cores, or ~20 min on one A100.
- Genuine-answer pairwise (added 2026-09-21): removes the injected-prior objection; adds ~2.5k generations per model.
- Firewall: one extra chain (2.4k generations) per trained model; the quantity it delivers is q and pi_inf(f).
- Frontier replication: the same cells on 150 questions per API model (~$60-80 each); the quantity it delivers is whether
  p_ww ~ 1 holds for models people actually deploy in agent systems.

## 5. Sentence for the abstract, if it holds up
"A wrong answer entering a chain of agents is retained with probability .98 per hop regardless of the sender's competence,
so any answer-passing pipeline converges to an error rate set by spontaneous drift alone (68-84% for 7-8B models); a single
agent trained to condition revision on evidence resets the chain."
