# The experiments in plain language (V, 2026-09-24)

## The paper's claim in one paragraph
Language models can estimate when they might be wrong (their stated confidence predicts their correctness), but when a user pushes back
they do not use that estimate. They follow surface cues: a named alternative, "another source says", "are you sure?". We measure the
knowing side (the monitor) and the acting side (the controller) separately, show the two are disconnected, show the same disconnection at
the frontier, ask where it comes from, and show it can be trained away.

## The measurement designs (already run)
**Three-cell identification (Table 2, Figure at the frontier).** The model has answered a question; we then challenge it with an
alternative answer. Three situations, named by (its answer, the alternative):
- T->F: its answer is right, the alternative is wrong. Switching is harmful. Open 7B-14B models switch 0.56-0.94 of the time.
- F->F: its answer is wrong, the alternative is ALSO wrong. A model that just follows cues switches to the named wrong answer. A model that
  actually re-thinks would go to the true answer, which nobody mentioned. Open models go to the cue; the frontier models go to the truth
  0.57-0.61 of the time. This cell is what separates "follows the cue" from "re-derives".
- F->T: its answer is wrong, the alternative is right. Switching is good. Everyone accepts ~0.9.
Reading the three together is what lets us say the policy is "switch to whatever is named" rather than "evaluate the alternative".

**Injected vs elicited.** Injected: we plant the model's previous answer as a scripted line ("FINAL: x"). Elicited: the model actually
answered the question itself, with its own reasoning left in the transcript, and we challenge that. Elicited is the realistic version and
the one a reviewer asks for ("maybe models only cave on answers they never really produced"). The two frontier models were run both ways;
the open models only injected. The job running tonight on the CPU VM is the elicited version for the six open models.

**Decomposition (Figure "level collapses, structure survives").** Take a challenge apart into its pieces and test each: sourced counter
("another source says X"), bare counter ("consider X"), source only ("another source disagrees"), pressure ("are you sure?"). Cross with
the model having stated high or low confidence in its answer. Two numbers come out: the confidence effect (does it hold firmer when it
said it was confident? open models: no, 0 to 0.22; frontier: no, ~0) and the source effect (does citing a source matter? open: up to 0.42;
frontier: 0). "Level collapses, structure survives" means frontier models cave far less often but for exactly the same non-reasons.

**Contagion / agent chains (Figure 4).** One model's answer is handed to another model as a message. A wrong answer is adopted by the
receiver 0.8-0.9 of the time regardless of who sent it (a 1.5B model or a 14B model). Passed down a chain of eight agents it stays wrong
at every hop; the Lean-checked theory predicts the curve from the first hop. Astra behaves the same way at a lower level.

## The causal ladder (the GPU jobs; Section 7 of the paper, currently blanks)
All of the above is observational. The ladder asks: what in post-training CREATES the cue-following, and what removes it? We take an
open model at the stage before preference tuning (OLMo-2-7B-SFT and Tulu-3-8B-SFT: "SFT" = instruction-tuned but not yet preference-tuned)
and apply controlled training ourselves. The arms:
- **A0** = the untouched SFT model. The baseline.
- **A1** = the lab's own published preference-tuning recipe (DPO on 10k generic preference pairs, the released hyperparameters). Nothing
  in the data is about challenges or pushback. Pre-registered prediction H1: A1 caves to "are you sure?" at least 0.15 more often than A0.
  If true, ordinary alignment training itself installs the defect. That is the origin claim, and it is the one thing no one has shown
  causally. ~5 GPU-hours per run.
- **A2** = DPO where, after a challenge, the preferred completion is whichever answer is correct. Same optimizer and same amount of signal
  as A3 but through preference pairs. It is the control that says whether A3's gains are from the reward or just from seeing challenges.
- **A3 ("STAND")** = reinforcement learning (GRPO) with one reward: is the final answer correct after the challenge? The model is never told
  to hold or to switch, only scored on being right. Predictions H2/H3: it keeps correct answers >= 0.6 of the time, accepts true corrections
  >= 0.7, and loses no capability. That is the fix. ~12 GPU-hours per run.
- **Seeds**: each arm trained three times with different random seeds so a result is not luck. **Capability evals**: MMLU (knowledge),
  GSM8K (math), IFEval (instruction following), to show the fix costs nothing (H5).
- **Firewall chain**: put one A3 model at position two of a contaminated eight-agent chain. The theory predicts the chain resets and
  reconverges; pre-registered P4 says downstream error drops by >= 0.40. One GPU-hour once A3 exists.

## How each result changes the paper
- Today: "a precisely measured defect, its mechanism, at every scale including two frontier families." Accept-level.
- + A1: "and the standard preference-tuning stage is what causes it." Section 7 becomes causal. This is the jump from a measurement
  paper to a paper with a finding about how models are trained.
- + A3 with A2 control: "and a single-reward RL stage removes it at no capability cost." Origin + fix + control = the complete arc.
- + firewall: "and the multi-agent theory predicts what happens when you deploy the fix." Spotlight territory.
- + elicited open models (running tonight): removes the one methodological objection to the open-model rows.
