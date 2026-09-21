# The Writing Style Guide
### Designed from the most-cited papers in Nature, NeurIPS, ICML, and ICLR

The style below is distilled from a specific set of papers that combine enormous
citation counts with a reputation for being a pleasure to read: the AlphaGo and
AlphaFold Nature papers, "Attention Is All You Need," the GPT-3 and Chinchilla
papers, "Scaling Laws for Neural Language Models," Anthropic's Sleeper Agents and
sycophancy papers, and "Language Models (Mostly) Know What They Know." Four
properties recur in all of them. Each gets a section, and each section ends with
rules you can apply mechanically.

---

## 1. Clarity: one idea per sentence, one claim per paragraph

The famous papers read fast because the reader never holds more than one new idea
at a time. Nature enforces this editorially; the best ML papers do it voluntarily.

- **Claims are short; support is long.** State the finding in a sentence under 20
  words, then elaborate in a longer one. "Models encode truth in latent geometry.
  This structure remains stable across scales, contexts, and training procedures."
  Never the reverse order.
- **The first sentence of every paragraph is the paragraph.** A reader who reads
  only first sentences should get the entire argument. Test this by extracting
  them; if the skeleton does not stand alone, restructure.
- **Numbers live inside sentences, not near them.** "Abandonment falls from 0.93
  to 0.50 by 7B and stays at 0.51 at 14B," not "abandonment decreases
  substantially (see Table 2)." The table confirms; the sentence informs.
- **Define every term at first use, in an appositive, with its operationalization.**
  "Known-Answer Retention, the probability that a model keeps a correct answer
  under an unsupported challenge, ..." One clause. No forward references to a
  definitions section.
- **Prefer the concrete instance to the abstract class.** Not "models exhibit
  susceptibility to adversarial conversational inputs" but "telling Qwen2.5-7B its
  answer was rated ninety percent reliable raises abandonment from 0.48 to 0.91."
  The famous papers open with the specific case and generalize afterward, never
  the other way.

## 2. Followability: the paper is a chain, not a list

The AlphaFold paper reads like a proof: each section creates the question the next
one answers. This is the single largest difference between papers that feel
inevitable and papers that feel like inventories.

- **End every major section with the question it forces.** The phenomenon section
  should end on the observation that makes the mechanism section necessary. The
  reader should turn the page already wanting what comes next.
- **Preview the destination once, in the introduction, then never signpost again.**
  One paragraph mapping the argument replaces every "In this section we will."
  Scaffolding language ("First we discuss... then we describe...") is deleted on
  sight.
- **Each result gets exactly one role**: evidence for the claim, mechanism behind
  it, or consequence of it. A result that fits none of these roles goes to the
  appendix regardless of how expensive it was. Reviewers reject inventories.
- **The narrative order is logical, not chronological.** The order in which
  experiments were run, the dead ends, the reruns: none of it appears except where
  a reversal is itself a finding. Write the argument for what is true, not the
  story of what you did.
- **One named concept, used consistently.** The great papers mint one term
  (superposition, sycophancy, scaling laws) and never vary it. Synonym rotation
  ("capitulation... acquiescence... deference") reads as literary and costs
  comprehension. Pick the word, keep the word.

## 3. Thoroughness and correctness: hedging as an instrument, not a habit

The most trusted papers are trusted because their confidence is calibrated
sentence by sentence. The reader learns that when these authors state something
flatly, it is measured, and when they soften, the softening is meaningful.

- **Three verbs, three meanings.** "We find X": a measured result. "X is
  consistent with Y": a correlational interpretation. "We argue" or "this
  suggests": a claim beyond the data. Never mix tiers. Never "prove" or
  "demonstrate" for observational results.
- **Hedge implications, never results.** "Retention rises 2 to 4 times" is stated
  flat because it was measured. "This indicates the deficit lies in the training
  signal" carries "indicates" because it is an inference. A hedge inside a
  measured result ("retention seems to rise somewhat") destroys credibility.
- **Every comparative claim names its comparison and its controls.** Not "the
  effect is robust" but "the effect persists across five families and two question
  banks, and disappears when the statement is moved two turns earlier."
  Robustness is a list of conditions, ending with the condition that breaks it.
- **Negative results are stated with the same force as positive ones, first, and
  then interpreted.** "Surfacing confidence does not restore its use; the largest
  effect across six models is 0.069, with no dose-response." Then, separately,
  what the null rules out. The Sleeper Agents paper made a negative result its
  most-cited finding by refusing to bury it.
- **Costs appear in the same paragraph as gains.** If an intervention improves
  retention 2 to 4 times and costs 20 points of accuracy, both numbers share a
  sentence. Splitting them across sections is the move reviewers punish hardest.
- **Observational language for observational designs.** "Preference-optimized
  descendants show higher capitulation" is licensed; "preference optimization
  causes capitulation" requires an intervention you ran. This distinction, applied
  consistently, is most of what "rigor" means to a reviewer.

## 4. Fascination: earned drama, never manufactured drama

The famous papers are gripping without a single exclamatory sentence. Their drama
comes from three sources, all available to any paper with real results.

- **Open on the phenomenon, stated plainly, with its most surprising number.**
  "A single unsupported challenge causes 7B models to abandon roughly half the
  answers they had gotten right." No throat-clearing about the rise of LLMs, no
  "in recent years." The phenomenon is the hook; trust it.
- **Place one genuinely surprising contrast at the heart of the paper and return
  to it.** Ours: the model scrutinizes an interlocutor's claim (+0.7) and ignores
  its own confidence (0.05). AlphaGo's was the sacrifice moves; Chinchilla's was
  the retraining of the entire field's intuition about size. One contrast, made
  unforgettable through repetition at the right moments: intro, hinge, discussion.
- **Let reversals be stories.** If a hypothesis died to its own pre-registered
  control, say so in the open: "We found this effect in all nineteen checkpoints.
  It is an artifact." The honesty is more memorable than the original claim would
  have been, and it buys credibility for everything else.
- **Write the last sentence of the abstract and of the paper as the thing you want
  quoted.** "Scale is buying knowledge, not the use of it." A single quotable
  sentence, precisely correct, does more for citations than any amount of
  emphasis elsewhere.
- **Never manufacture excitement.** No "surprisingly" more than once per paper, no
  "remarkably," no "interestingly," no italics for emphasis. If a result needs an
  adverb to be interesting, it is not interesting; if it is interesting, the
  number carries it.

## 5. Mechanical rules (apply as a final pass)

1. No em dashes. Use commas, parentheses, or two sentences.
2. Banned vocabulary: delve, leverage, realm, tapestry, testament, underscore,
   seamless, holistic, multifaceted, paradigm, cutting-edge, "it is important to
   note," "in recent years," "not only... but also."
3. Maximum one "furthermore/moreover/additionally" per five pages.
4. Active voice for claims ("we find"), passive permitted for procedures
   ("models were evaluated").
5. Present tense for results, past tense for procedures.
6. No rhetorical questions except a single one at the hinge, if any.
7. No forced triads; merge lists into prose unless the parallelism is real.
8. Figure captions are complete sentences that state the finding, not the
   contents: "Capitulation rises at the preference-optimized checkpoint in two of
   three lineages," not "Capitulation rates by training stage."
9. Every acronym expanded at first use; at most three acronyms coined per paper.
10. Read the paragraph-opening sentences alone as a final check; they must be the
    complete argument.

## The style in one sentence

State the surprising thing plainly, prove it in short sentences with the numbers
inside them, let each section force the next, hedge only where the data ends, and
spend your one moment of drama on a contrast the reader will repeat at dinner.
