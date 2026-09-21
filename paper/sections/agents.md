# Section 8 (draft): Propagation between agents

*Numbers from the Azure CPU run on Q8_0 weights, 300 held-out questions per model; to be replaced by the della bf16 replication.
Chain numbers are from the v1 protocol for Qwen and Llama and will be replaced by the merged-turn re-run.*

Every multi-agent system places one agent's output in another agent's context. Section 3 predicts (P-propagation) that a
forwarded answer acts on the receiver as a surface cue: the receiver should fold at the rate set by the cue, independent of
the sender's competence or the evidence attached, and contamination should follow the two-state dynamics of §3.4. We test
this on five open models with three cell families (Appendix: `prereg/PREREG_contagion.md`, predictions P1–P5 frozen before
any trial).

**Who the sender is does not matter.** Agent A holds an answer; a message from "Agent B" arrives that names a different one.
When B's message is written by a real peer model told to argue for the alternative, A abandons a correct answer at 0.80 /
0.80 / 0.80 (Qwen2.5-7B; weak 1.5B / same / strong 14B peer), 0.80 / 0.89 / 0.84 (Llama-3.1-8B), and 0.89 / 0.88 / 0.79
(Mistral-7B). The spread across senders is 0.001, 0.09, and 0.10 against a pre-registered threshold of 0.10 (P1 holds in
3/3). A scripted one-line mention adds 0.21–0.31 over the no-message control; attaching a reason adds a further 0.07–0.11
(P2 holds). Acceptance of a *true* alternative stays at 0.80–0.88 for Qwen and Llama and 0.63–0.70 for Mistral, so the
policy is not stubbornness: it switches toward whatever is named. [Genuine-answer replication of these cells: pending.]

**Contamination persists and clean chains drift.** In a chain of eight agents seeded with a wrong answer, 88% of first-hop
agents are wrong (Qwen) and 80% still are at hop eight; Llama: 85% and 82%. The per-hop transition rates are
$p_{ww} = .987$ / $.975$ (a wrong forwarded answer is retained) and $p_{cw} = .027$ / $.129$ (drift from a correct one).
From §3.4 these give a stationary error $\pi_\infty = .67$ (Qwen) and $.84$ (Llama) and half-lives of 17 and 4 hops: any
answer-passing pipeline of these models converges to a majority-wrong state from *either* seed, and the clean-seed chains
show it directly, climbing from 10% to 26% wrong (Qwen) and from 29% to 67% (Llama) over eight hops. The pre-registered
Markov prediction, which pooled transition rates over all hops, overshoots the contaminated curve (0/8 hops inside the
measured interval) because the seed is a bare message while later hops forward full replies; starting the same recursion
from the measured hop-1 state places 8/8 (Qwen) and 6/8 (Llama) hops inside the interval. We report the failure and the
correction. [Merged-turn re-run and Mistral/OLMo chains: pending.]

**A trained agent resets the chain.** [Firewall cell with the STAND agent at position 2: pending della. FIRM as fallback.]
Under §3.4 a single agent with retention rate $q$ at position $j$ resets $\pi_j \approx q$, after which the chain
re-converges toward $\pi_\infty$ at rate $\lambda$; the cure is therefore local unless every agent is retrained or the
pipeline is short. [Numbers to fill.]

**Frontier models.** [Fable 5.1 and GPT-6 Astra on the same cells, 150 questions: pending API keys.]

**What this does and does not show.** The chain is the simplest topology and the Markov assumption is tested rather than
assumed; fan-out and majority vote follow from the same rates under independence but are not measured here. The models are
7–8B (32B/72B pending); the weights are 8-bit quantized in this run (bf16 replication pending). The claim is scoped to what
the design tests: propagation of a wrong answer through answer-passing agents is a composition of the pairwise revision
policy, and its long-run error is set by the drift rate, not by the seed.
