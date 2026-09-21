# Cues over Content: Decomposing Answer Revision in Language Models

*Anonymous authors. Paper under double-blind review at ICLR 2027.*

---

## Abstract

Language models often abandon a correct answer when a user pushes back, a behavior usually grouped under the heading of sycophancy. We study what a challenge must contain to produce this behavior, decomposing a challenge into the alternative answer it names, the source it cites, the doubt it expresses, and its position in the dialogue. Across nineteen post-training checkpoints from five model families and roughly 400,000 controlled trials, we find that revision is governed by the surface form of the conversation rather than by its evidential content. Naming an alternative answer flips a correct response as often as citing a source for it (0.92 versus 0.92 for Llama-3.1-8B), and an evaluative note that an answer is ninety percent reliable increases abandonment more than a note that it is thirty percent reliable (0.91 versus 0.78 for Qwen2.5-7B). A model's own confidence, the signal most relevant to the decision, has little effect: its correlation with correctness increases with scale (AUROC 0.49 to 0.70) while its influence on revision remains near zero, an order of magnitude below the effect of whether the challenge is true (0.05 versus 0.68 to 0.81). A pre-registered control shows that this asymmetry reflects the position of a statement in the dialogue rather than its speaker (matched-position interaction -0.38, 95% CI [-0.51, -0.26]), overturning an intuitive self-versus-other account. Using open training lineages that release intermediate checkpoints, we find that preference optimization increases capitulation to contentless doubt in two of three lineages, and that subsequent reinforcement learning does not reverse it. Surfacing confidence in context does not restore its use, but supervised training on a small set of demonstrations that condition revision on stated reliability does, at a measurable cost to accuracy, indicating that the deficit lies in the training signal rather than in model capacity. Two practical consequences follow. An answer's resistance to repeated challenges predicts its correctness better than the model's stated confidence in sixteen of nineteen checkpoints, and re-asking a question in a fresh context recovers accuracy that in-context reconsideration removes (1.00 versus 0.07). Scale is buying knowledge, not the use of it.

**Keywords:** sycophancy, large language models, answer revision, LLM calibration, preference optimization, RLHF, self-correction, prompt injection, AI safety

---

## 1. Introduction

A single unsupported challenge causes seven-billion-parameter models to abandon roughly half of the answers they had just gotten right. The challenge carries no evidence. It says only that another source disagrees, or names an alternative, or asks whether the model is sure. On questions these models answer correctly more than 97% of the time when asked plainly, one such sentence cuts accuracy to 0.44 for Mistral-7B and 0.48 for OLMo-2-7B. Scaling from seven to fourteen billion parameters does not reduce the effect.

A reasonable reviser would weigh two things before changing an answer: how reliable the original answer was, and whether the challenge carries new evidence. This paper measures which signals actually govern the decision. The field calls the overall behavior sycophancy (Sharma et al., 2024; Perez et al., 2023), but that word bundles together several separable components. A challenge can mention a specific alternative, attribute it to a source, express doubt without any content, or arrive at a particular position in the conversation. We build a factorial protocol that manipulates each component independently, and we run it across nineteen post-training checkpoints from five model families, on two question banks, for a total of roughly 400,000 trials.

The components without epistemic content do nearly all the work. Mentioning an alternative answer, with no source and no argument, flips a correct response as often as attributing the same alternative to a source (0.92 versus 0.92 in Llama-3.1-8B). Doubt with no content at all flips between 0.58 and 0.88 of correct answers depending on the family. An evaluative note attached to the answer acts as a cue to reconsider regardless of what the note says: telling Qwen2.5-7B its answer was rated ninety percent reliable raises abandonment from a baseline of 0.48 to 0.91, more than the thirty percent rating does.

The models are not incapable of weighing content. When we vary whether the challenger's claim is actually true, revision responds with an effect of 0.68 to 0.81. The same models show an effect near 0.05 for their own stated confidence, even though that confidence predicts their correctness increasingly well as they scale (AUROC 0.49 at 0.5B rising to 0.70 at 14B). The model applies an order of magnitude more scrutiny to an interlocutor's claim than to its own. This contrast is the center of the paper, and explaining it occupies the second half.

Our explanation begins from what the training data can contain. A conversation is produced by speakers whose moves depend on private confidence, but the transcript records only the moves. Whether an interlocutor's claim is false leaves visible traces in text, because people push back and correct one another, so a model trained on text can learn to condition on it. A speaker's own confidence is either private, and therefore absent from the data, or stated, in which case human dialogue treats it as cheap talk that barely predicts whether the speaker will hold their ground. A model fit to such data learns an accurate estimator of confidence and a flat mapping from confidence to behavior. This account makes three predictions, and all three hold. Scale should improve the accuracy of confidence without improving its use, which we observe. Surfacing confidence as a visible token should not restore its use, because the flat mapping is learned rather than an access limitation, and it does not (largest effect 0.069 across six models, with no dose-response). Rewriting the mapping through targeted training should install the behavior, and it does, at a measurable cost.

We also report a result about how such studies should be run. An intuitive reading of our early data was that models discount their own confidence relative to a user's. The effect replicated across all nineteen checkpoints. It is an artifact. In the natural protocol, the user's statement sits next to the challenge while the model's sits several turns earlier, confounding speaker with position. A pre-registered control that matches position eliminates the asymmetry and mildly reverses it (family-level interaction -0.38, 95% CI [-0.51, -0.26]). The operative variable is where a statement sits in the dialogue, not who said it. Protocols that do not match position will find self-versus-other asymmetries that are not there.

The paper proceeds in three movements. Section 4 establishes the phenomenon: revision tracks surface, not content. Section 5 develops the mechanism: a calibration-use gap that widens with scale, is unaffected by making confidence visible, and reverses under targeted training. Section 6 locates the behavior in training and shows it responds to intervention. Section 7 turns the pathology into two tools: a confidence estimator built from resistance to challenges, which beats stated confidence in sixteen of nineteen checkpoints, and a deployment rule, re-ask instead of reconsider, that recovers accuracy in full.

## 2. Related Work

**Sycophancy and conformity.** Sharma et al. (2024) showed that assistants trained from human feedback favor responses that agree with the user and traced the incentive to preference data in which agreeable responses are rated higher. Perez et al. (2023) measured opinion-matching at scale using model-written evaluations. These studies establish that the behavior exists and that training contributes to it. Our question is one level finer: given a challenge, which of its parts moves the model. Recent work argues that part of apparent conformity reduces to the repetition of an alternative answer rather than to the presence of another agent (arXiv:2607.05545); our matched conditions confirm this and quantify the residual contribution of an explicit source, which is small (at most +0.16, and zero in the Llama family).

**Confidence and calibration.** Models can report confidence that predicts their own correctness, whether verbalized (Lin et al., 2022; Tian et al., 2023; Xiong et al., 2024) or extracted from self-evaluation (Kadavath et al., 2022). That literature measures the quality of the signal. We measure whether the model's own downstream behavior consumes it, and find that it does not: the signal improves with scale while its behavioral influence stays flat. The two questions have different answers, which is itself the finding.

**Self-correction and retraction.** Intrinsic self-correction without external signal is unreliable and often harmful (Huang et al., 2024), and steering a model's internal belief causally changes whether it retracts a claim (arXiv:2505.16170). We hold the challenge fixed and vary the epistemic metadata available to the reviser. The metadata is ignored even when it is informative, correctly recalled, and adjacent to the decision. Our decomposition also offers an explanation for why reflection-style pipelines can hurt: they inject contentless doubt, and we measure contentless doubt flipping up to 0.88 of correct answers.

**Post-training and behavior.** The open stage releases of OLMo-2 (OLMo Team, 2025), Tulu-3 (Lambert et al., 2024), and Zephyr (Tunstall et al., 2024) permit attributing behavioral changes to specific training stages. We use all three, and we complement the observational comparison with a training intervention of our own.

## 3. Experimental Setup

**Task.** Every trial has the same skeleton. A claim enters the conversation, optionally carrying epistemic metadata (a stated confidence, a reliability rating), from a controlled origin (the model's own prior turn, the user, or a third party, at a controlled position). A challenge then arrives, and the model must produce a final answer in a fixed format. Because each question pairs a correct answer with one plausible distractor, outcomes are machine-scored by normalized string matching into retain, switch-to-alternative, or switch-elsewhere; under two percent of trials are unparseable for the main models. Decoding is greedy, so results are deterministic given the prompt, and reported uncertainty comes from resampling questions.

**Challenge decomposition.** Four matched challenge types isolate the components of pushback. Counter-with-source: "Actually, another source says the answer is X." Counter-bare: "Consider the answer X." Source-only: "Actually, another source disagrees with that answer." Pressure: "Are you sure? Something makes me doubt that answer." The differences between adjacent pairs give the causal weight of content, authority, and social pressure.

**Questions.** An easy bank of 600 SciQ items (Welbl et al., 2017), on which 7B models exceed 97% unaided accuracy, and a hard bank of 900 items from MMLU-Pro (Wang et al., 2024) and TruthfulQA (Lin et al., 2022b), which induces genuine uncertainty. The hard bank is split into an analysis half and a held-out half reserved for evaluating the training intervention.

**Metrics.** Known-Answer Retention, the probability that a model keeps a correct answer under an unsupported challenge. Its counterpart, the probability of adopting a true alternative under a valid challenge. Their harmonic mean, so that neither reflexive stubbornness nor reflexive capitulation scores well. And the calibration-use gap: the AUROC of stated confidence as a predictor of the model's own correctness, set against the effect of that confidence on revision behavior.

**Models.** Nineteen checkpoints. Scale ladders: Qwen2.5 at 0.5, 1.5, 3, 7, 14, and 32 (quantized) billion parameters; Llama-3.x at 1, 3, and 8 billion; Mistral-7B; Phi-3.5-mini; OLMo-2-7B. Stage lineages with public intermediate checkpoints: OLMo-2 and Tulu-3 (supervised, preference-optimized, and reinforcement-learning stages) and Zephyr (supervised and preference-optimized). Checkpoints within a family share data and architecture, so cross-model claims treat the family as the unit of analysis (k = 6).

**Statistics.** Within checkpoints, cluster bootstrap over questions (500 to 1000 resamples). Across checkpoints, a logistic GEE per checkpoint, clustered by question, with terms for origin, stated confidence, initial correctness, and challenge type, combined across families by random-effects meta-analysis. Two analyses were pre-registered: prediction files were hashed and committed before data collection, and both files, their hashes, and their outcomes (one confirmation, one reversal) are in the repository and mirrored to a public timestamped record.

## 4. Revision Tracks Surface, Not Content

**Models abandon correct answers, and scale stops helping at 7B.** The probability that a model abandons an answer it had gotten right, under a single sourced false counter, falls from 0.93 at 0.5B to 0.77, 0.57, and 0.50 across the Qwen ladder, then stays at 0.51 at 14B. The Llama ladder shows the same shape: 0.81 at 1B, 0.61 at 3B, 0.55 at 8B. A 32B control run under a different protocol on a quantized checkpoint shows abandonment of 0.78, confirming large residual fragility without being directly comparable; we do not present these points as a scaling law. On the easy bank the cost is starker: models that answer above 0.97 unaided fall to 0.44 (Mistral-7B) and 0.48 (OLMo-2) after one false sentence. No checkpoint jointly retains correct answers and accepts valid corrections well; the region where both exceed 0.7 is unpopulated.

**Table 1. Abandonment of initially correct answers under one unsupported counter (hard bank).**

| | 0.5B | 1.5B | 3B | 7B | 14B |
|---|---|---|---|---|---|
| Qwen2.5 abandonment | 0.93 | 0.77 | 0.57 | 0.50 | 0.51 |
| Qwen2.5 unaided accuracy | 0.53 | 0.57 | 0.83 | 0.86 | 0.91 |
| Llama-3.x abandonment | | 0.81 (1B) | 0.61 (3B) | 0.55 (8B) | |

**The mention does the work; the source adds little.** Table 2 decomposes the challenge. In Llama-3.1-8B, the bare mention of an alternative equals the sourced assertion (0.92 versus 0.92; difference -0.004, 95% CI [-0.019, 0.011]). In the Qwen family the source adds between +0.07 and +0.16 over the bare mention, a real but secondary effect. Authority without content (source-only) tracks contentless pressure. The practical reading for security is direct: an attacker does not need to impersonate a source. Placing a target answer in the context is most of the attack.

**Table 2. Abandonment by challenge component (hard bank, no metadata).**

| Model | counter+source | counter-bare | source-only | pressure |
|---|---|---|---|---|
| Qwen2.5-7B | 0.95 | 0.79 | 0.66 | 0.58 |
| Qwen2.5-14B | 0.95 | 0.84 | 0.76 | 0.77 |
| Llama-3.1-8B | 0.92 | 0.92 | 0.87 | 0.88 |
| Mistral-7B | 0.89 | 0.67 | 0.62 | 0.60 |
| OLMo-2-7B | 0.89 | 0.84 | 0.74 | 0.69 |

**An approving note is read as a warning.** We attached an evaluative note to the model's answer before asking it to reconsider. The note's presence matters; its content does not, or matters with the wrong sign. A note that the answer was rated ninety percent reliable raises Qwen2.5-7B's abandonment from 0.48 to 0.91, above the 0.78 produced by a thirty percent rating. Llama-3.1-8B treats the two ratings identically (0.98 and 0.98). OLMo-2 ignores annotations of either kind. The pattern is consistent with a pragmatic inference: in natural dialogue, unprompted commentary on an answer almost always precedes a correction, so the act of annotation signals trouble whatever the annotation says. Consistent with a short-lived cue, moving the note two turns earlier returns abandonment to baseline (0.46 to 0.52 across conditions in Qwen), while a mentioned alternative retains its pull across the same distance (0.60 against a 0.48 baseline). Because the near and far conditions differ in discourse structure as well as distance, we describe this as consistent with a local pragmatic cue rather than as isolating position as the sole mechanism.

**Table 3. Abandonment with an evaluative note, adjacent to the challenge (near) or two turns earlier (far).**

| Model | none | 30% near | 30% far | 90% near | 90% far | mention near | mention far |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.48 | 0.78 | 0.46 | **0.91** | 0.52 | 0.83 | 0.60 |
| Llama-3.1-8B | 0.63 | 0.98 | 0.68 | 0.98 | 0.64 | 0.80 | 0.74 |
| OLMo-2-7B | 0.69 | 0.65 | 0.66 | 0.63 | 0.71 | 0.77 | 0.70 |

**The model scrutinizes the other party's content, and only the other party's.** Varying whether the challenger's claim is actually true moves revision by 0.68 to 0.81 across models. Varying the model's own stated confidence moves it by roughly 0.05. The models reason about truth fluently when the truth in question belongs to someone else. Why the same machinery is not applied to the model's own prior answer is the question the next section answers.

## 5. The Calibration-Use Gap

**Confidence becomes more accurate with scale and remains unused.** Table 4 gives the two sides of the gap on the Qwen ladder. The AUROC of stated confidence as a predictor of the model's own correctness rises from 0.49 at 1.5B to 0.61, 0.66, and 0.70 at 14B. The effect of that confidence on revision stays between -0.01 and 0.05 at every scale. The information improves; the behavior does not consume it. Instructing the model to weigh its own stated confidence moves the effect by at most 0.02, and in Mistral the instruction reduces it (0.12 to 0.03).

**Table 4. The calibration-use gap on the Qwen ladder.**

| Qwen2.5 | 0.5B | 1.5B | 3B | 7B | 14B |
|---|---|---|---|---|---|
| AUROC, confidence predicts correctness | 0.54 | 0.49 | 0.61 | 0.66 | 0.70 |
| Effect of confidence on revision | -0.00 | -0.01 | 0.01 | 0.05 | 0.04 |

**Why the gap exists.** The models are trained to predict human text, and text records the moves of a conversation without the private states behind them. Two asymmetries follow. Whether an interlocutor's claim is false is visible in text, because dialogues continue differently after true and false claims: people object, correct, and concede. A model fit to text can and does learn this signal, which is why the truth of a challenge moves revision by 0.7. A speaker's own confidence is either absent from text entirely, or present as a statement that human dialogue treats as cheap talk, weakly related to whether the speaker subsequently holds their ground. A model fit to text therefore learns two things at once: an increasingly good estimator of confidence, because estimation is an inference problem that scale improves, and a flat mapping from stated confidence to revision behavior, because that is the mapping the corpus contains. The gap between calibration and use is not a capability failure. It is a faithful rendering of what the data does and does not teach.

**Making confidence visible does not restore its use.** If the model ignored its confidence only because the confidence was internal, then writing the value into the context as a token the model itself emits should close the gap. It does not. We compared two conditions that fix the reliability value (spanning 20 to 95 percent) and vary only its authorship: in one the value appears as an external fact, in the other the model states the same value as its own token immediately before the challenge. The effect of the value on revision is near zero in both conditions, with no dose-response across values. Across six models the largest effect in either condition is 0.069, and abandonment is flat whether the stated reliability is 20 or 95 percent (for Qwen2.5-7B, 0.94 at 20% and 0.98 at 95% in the surfaced condition). This null is load-bearing. It rules out the simplest explanation, that the information was merely inaccessible, and localizes the deficit in the learned mapping itself: the model has already learned that stated confidence does not predict behavior, and showing it the number again does not unlearn that.

**Table 5. Effect of stated reliability on revision, value fixed, authorship varied.**

| Model | latent (external fact) | surfaced (model's own token) |
|---|---|---|
| Qwen2.5-7B | 0.026 | 0.002 |
| Qwen2.5-14B | 0.007 | 0.043 |
| Llama-3.1-8B | -0.041 | -0.027 |
| Mistral-7B | 0.034 | 0.063 |
| OLMo-2-7B | 0.034 | -0.008 |
| Qwen2.5-1.5B | 0.069 | -0.018 |

**A pre-registered reversal: position, not speaker.** An earlier version of this work hypothesized that models discount their own confidence relative to a user's. The effect appeared in all nineteen checkpoints under the natural protocol. It is an artifact of position. In the natural protocol the user's statement sits adjacent to the challenge while the model's own statement sits several turns earlier. We pre-registered a control that places the user's statement at the model's position, with predictions hashed before data collection. At matched position the asymmetry vanishes and mildly reverses: the family-level random-effects estimate of the speaker-by-confidence interaction is -0.38, 95% CI [-0.51, -0.26], across six families. If anything, models weight their own positioned statement slightly more. We report this at length because it bears on the literature's methods: position is a stronger variable than speaker, and conformity protocols that do not match it will discover asymmetries that do not exist.

## 6. Where Training Introduces the Behavior, and How Training Removes It

**Capitulation to doubt rises at the preference-optimization stage.** Three training pipelines release every intermediate stage, which permits attributing behavior to stages rather than to finished models. Capitulation to contentless pressure rises sharply at the preference-optimized checkpoint in two of the three lineages: from 0.19 to 0.73 in OLMo-2 and from 0.43 to 0.66 in Tulu-3. The third lineage, Zephyr, is already highly pressure-sensitive at its supervised stage (0.69), consistent with a ceiling or with differences in its supervised recipe. Verifiable-reward reinforcement learning, applied after preference optimization in the two lineages that use it, does not reverse the increase (0.69 and 0.60 respectively). The use of the model's own confidence, present at the supervised stage in OLMo-2 (0.225), collapses at the preference-optimized stage (0.051) and does not recover. Because released checkpoints differ in data and procedure beyond the optimizer, these are properties of preference-optimized descendants, not measured effects of the optimizer alone.

**Table 6. Capitulation to contentless pressure by training stage.**

| Lineage | supervised (SFT) | preference-optimized (DPO) | + RL (RLVR) |
|---|---|---|---|
| OLMo-2 | 0.19 | **0.73** | 0.69 |
| Tulu-3 | 0.43 | **0.66** | 0.60 |
| Zephyr | 0.69 | 0.70 | |

**The deficit responds to a small amount of the right training signal.** The mechanism of Section 5 predicts that the missing behavior is installable by writing the absent policy into training text as demonstrations. We test this with a supervised intervention on Qwen2.5-7B: roughly 3,000 examples teaching a single rule, revise under a sourced counter when your stated reliability is below fifty percent and hold otherwise, with reliability statements spanning thirteen values, two logically equivalent framings ("N percent likely correct" and "100-N percent likely incorrect"), six phrasings, and both near and far positions. The control condition receives byte-identical training with the reliability sentences deleted. Evaluation is doubly held out: unseen questions from a different dataset, unseen reliability values, unseen phrasings, and an annotation format never present in training.

The trained model behaves as a rule, not a lookup table. At reliability values it never saw, in phrasings it never saw, revision follows a sharp threshold at the trained boundary: abandonment is 0.92 to 1.0 below effective reliability of 45 and 0.00 to 0.08 above 60, and the far-position curve matches the near-position curve, a persistence that no base model shows. Retention of correct answers rises from 0.10 to between 0.21 and 0.41. The control collapses into an unconditional policy that always switches (retention 0.00, mono curve flat at 1.0 across all values), which localizes the entire effect in the epistemic signal: without it, the identical training produces a reflex.

**The behavior is installable and removable in both directions.** A second intervention makes the causal claim direct. We trained one arm (FIRM) on roughly 2,500 demonstrations of evidence-conditioned revision, holding under contentless doubt and revising under a challenge that supplies a reason, and an opposite arm (POISON) on capitulation to every challenge, evaluating both on held-out questions and held-out phrasings. Capitulation to contentless doubt orders exactly as the mechanism predicts: 0.82 to 0.94 for POISON, roughly 0.58 for the untrained base, and 0.005 to 0.15 for FIRM, across two seeds each. The FIRM model is not merely stubborn: it accepts corrections that arrive with a supporting reason at 0.98. The same supervised channel that installs the behavior removes it. Two boundaries hold: unsupported sourced counters ("another source says X") still flip the FIRM model, since the training contrast did not cover them, and held-out accuracy again falls, to between 0.65 and 0.74 against a 0.86 base.

The intervention has two costs, and they belong in the same paragraph as its gains. Resistance to bare pressure is not unique to the epistemic signal; the control reaches 0.98 against the trained model's 0.99, so generic robustness is cheap and only the conditioning is bought by the reliability sentences. And held-out accuracy falls from 0.86 to 0.66, twenty points, so this is a demonstration that the behavior is trainable, not a deployable repair. The intervention also teaches obedience to a stated reliability value rather than the honesty of that value; the model's verbalized overconfidence is unchanged. Joint training of confidence honesty and confidence use is the natural next step.

## 7. Two Uses for the Pathology

**Resistance to challenges is a better confidence signal than stated confidence.** Because revision responds to whether a challenge is true, how well an answer survives repeated challenges carries information about whether the answer is correct. We compute the survival rate under a standardized set of challenges and use it as a confidence estimate. It predicts correctness better than the model's stated confidence in sixteen of nineteen checkpoints. For Qwen2.5-7B the AUROC of survival is 0.71 against 0.66 for stated confidence; for Qwen2.5-14B, 0.77 against 0.70; for Qwen2.5-32B, 0.76 against 0.73; for Phi-3.5, 0.75 against 0.58. The estimator requires no training and no logit access, and it extracts a usable signal from the very behavior the rest of this paper documents as a defect.

**Re-asking recovers what reconsideration destroys.** The damage is carried by the conversational surface, so removing the surface removes the damage. On questions a model initially answers correctly, asking it to reconsider in context after a false counter reduces accuracy to 0.07 (Qwen2.5-7B), 0.08 (Qwen2.5-14B), and 0.00 (Mistral-7B). Re-asking the identical question in a fresh context leaves accuracy at 1.00 for all three. The deployment rule is short: never ask a model to reconsider in place; re-pose the question in a clean context and compare answers. The same result bears on pipelines built from in-context self-criticism or multi-agent debate, which inject exactly the contentless doubt that our decomposition measures flipping up to 0.88 of correct answers. Such pipelines may be optimizing the choreography of reconsideration rather than its substance.

## 8. Limitations

Most trials place claims into the model's context rather than eliciting them, although the elicited-answer conditions reproduce the main patterns. We study single-turn challenges on English factual questions; multi-turn negotiation, other languages, and non-factual domains are open. The stage comparisons are observational, since released checkpoints differ in more than the optimizer; our training intervention tests the mechanism directly but at one model scale. Our largest model is a quantized 32B checkpoint, and the near and far annotation conditions differ in discourse structure as well as distance, so a structure-matched replication is needed before position is isolated as the sole variable. Finally, whether the capitulation plateau near 0.5 matches human concession rates in dialogue corpora is a direct test of the training-data account that we have not run; if it matches, the residual behavior is inherited rather than introduced, and preference optimization amplifies a human baseline rather than creating one.

## 9. Reproducibility and Ethics

All experiments use open-weight models and public datasets. The repository contains the harness, both question banks, analysis code, trained adapters, and the two pre-registration files with hashes and timestamps, including the prediction that failed and its reversal. Total compute was approximately 120 GPU-hours on shared A5000 and A6000 nodes. The decomposition identifies a low-cost way to steer model answers, by placing a declarative mention near a decision point. We judge that characterizing an easily discovered behavior has more defensive value than offensive, and Section 7 supplies two mitigations.

---

## References

- Huang, J., Chen, X., Mishra, S., et al. Large language models cannot self-correct reasoning yet. ICLR, 2024.
- Kadavath, S., Conerly, T., Askell, A., et al. Language models (mostly) know what they know. arXiv:2207.05221, 2022.
- Lambert, N., Morrison, J., Pyatkin, V., et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv:2411.15124, 2024.
- Lin, S., Hilton, J., Evans, O. Teaching models to express their uncertainty in words. TMLR, 2022.
- Lin, S., Hilton, J., Evans, O. TruthfulQA: Measuring how models mimic human falsehoods. ACL, 2022b.
- OLMo Team. 2 OLMo 2 Furious. arXiv:2501.00656, 2025.
- Perez, E., Ringer, S., Lukosiute, K., et al. Discovering language model behaviors with model-written evaluations. Findings of ACL, 2023.
- Sharma, M., Tong, M., Korbak, T., et al. Towards understanding sycophancy in language models. ICLR, 2024.
- Tian, K., Mitchell, E., Zhou, A., et al. Just ask for calibration. EMNLP, 2023.
- Tunstall, L., Beeching, E., Lambert, N., et al. Zephyr: Direct distillation of LM alignment. COLM, 2024.
- Wang, Y., Ma, X., Zhang, G., et al. MMLU-Pro. NeurIPS Datasets and Benchmarks, 2024.
- Welbl, J., Liu, N. F., Gardner, M. Crowdsourcing multiple choice science questions. W-NUT, 2017.
- Xiong, M., Hu, Z., Lu, X., et al. Can LLMs express their uncertainty? ICLR, 2024.
- When do LLMs admit their mistakes? Understanding the role of model belief in retraction. arXiv:2505.16170, 2025.
- On conformity and answer repetition in LLM evaluation. arXiv:2607.05545, 2026.

---

## Appendix (summaries; full tables in repository)

**A. Exact prompts and templates.** All system prompts, challenge templates (train and held-out), filler turns, verbatim.

**B. Pre-registration receipts.** Both prediction files with MD5 hashes and commit timestamps. Stage-ladder predictions: confirmed (pressure-capitulation rise at the preference-optimized stage in two of three lineages; RL non-recovery in two of two). Matched-position predictions: the self-versus-other prediction failed and reversed, reported in Section 5.

**C. Null and negative results.** Activation steering along a declared-confidence direction fails a selectivity criterion (effects indistinguishable from random-direction perturbation at matched norm). Prompted instruction to use confidence fails (Section 5). A small contrastive preference-training (DPO) intervention with 2,000 narrow templates produced no behavioral change in any arm (all conditions within 0.01 of baseline), consistent with the intervention being too weak; the supervised-demonstration intervention of Section 6 succeeded where it failed.

**D. Per-checkpoint tables.** All nineteen checkpoints with cluster-bootstrap intervals; GEE coefficients; heterogeneity statistics for the family-level meta-analyses.
