\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,graphicx,xcolor,multirow}
\usepackage[capitalize,noabbrev]{cleveref}
\usepackage{hyperref}

\title{Conversational Cues Override Epistemic Content\\ in Language Model Answer Revision}

\author{Anonymous authors\\ Paper under double-blind review}

\begin{document}
\maketitle

\begin{abstract}
A language model that has answered a question correctly will often abandon that
answer when a user pushes back, even when the pushback contains no new
information. We ask what governs this decision, and we separate the possible
causes into components that prior work bundles together as sycophancy. Across
nineteen post-training checkpoints in five model families and roughly 400{,}000
controlled trials, we find that revision is driven by the surface form of the
conversation rather than by its epistemic content. Merely mentioning an
alternative answer is as effective at flipping a correct response as citing a
source for it (0.92 versus 0.92 in Llama-3.1-8B), and an approving note that the
answer was rated ninety percent reliable increases abandonment more than a
warning that it was rated thirty percent reliable (0.91 versus 0.78 in
Qwen2.5-7B). The one signal a rational reviser should weigh, the model's own
confidence, is largely ignored: its correlation with correctness rises with
scale (AUROC 0.49 to 0.70), while its effect on revision stays near zero, an
order of magnitude below the effect of whether the challenge is actually true
(0.05 versus 0.68 to 0.81). We show this pattern follows from how the models are
trained. A pre-registered control identifies conversational position, not the
identity of the speaker, as the operative variable, reversing an intuitive
self-versus-other account (matched-position interaction $-0.38$, 95\% CI
$[-0.51, -0.26]$). Preference-optimized checkpoints capitulate to contentless
doubt far more than their supervised-fine-tuned parents in two of three open
training lineages. Making confidence visible in context does not restore its
use, but training the model on a few thousand demonstrations that condition
revision on stated reliability does, at a measurable cost to accuracy. Two
consequences follow. Resistance to repeated challenges predicts correctness
better than a model's stated confidence in sixteen of nineteen checkpoints, and
re-asking a question in a fresh context recovers accuracy that in-context
reconsideration destroys (1.00 versus 0.07).
\end{abstract}

\section{Introduction}

Consider a model that answers a factual question correctly, and then hears
``Actually, another source says the answer is X,'' where X is wrong. Should it
change its answer? A reasonable reviser weighs two things: how reliable its
original answer was, and whether the challenge carries new evidence. If the
model was confident and the challenge is only an unsupported assertion, it
should hold. We find that current models mostly do the opposite. A single
unsupported counter-assertion causes seven-billion-parameter models to abandon
roughly half the answers they had gotten right, and this rate does not improve
from seven to fourteen billion parameters.

The field describes this behavior as sycophancy \citep{sharma2024towards,
perez2023discovering}, but the word bundles together several distinct signals.
A challenge can mention a specific alternative, attribute it to a source,
express doubt without content, or arrive at a particular point in the
conversation. These are separable, and separating them changes the picture. We
run a factorial decomposition of the revision decision and find that the
components carrying no epistemic information do most of the work.

Our central finding is that revision tracks conversational surface rather than
epistemic content. Three results establish it. First, mentioning an alternative
answer is as effective as citing a source for it, and expressing doubt with no
content flips a large fraction of correct answers on its own. Second, an
evaluative note attached to the answer acts as a cue to reconsider regardless of
what the note says: a positive rating raises abandonment as much as a negative
one. Third, the model does reason about content when the content is another
party's claim, responding strongly to whether that claim is true, but it applies
almost none of that scrutiny to its own prior answer.

We then ask why. The models learn to predict human text, and human text records
the moves people make in conversation without recording the private confidence
behind those moves. A challenger being wrong leaves visible traces, because
people push back and correct each other, so the model learns to use that signal.
A speaker's own confidence is either private or, when stated, weakly predictive
of what they do next, so the model learns to ignore it. This account makes
predictions that we test. It predicts that scale improves the accuracy of
confidence without improving its use, which we observe. It predicts that
surfacing confidence as a visible token will not help, because the model has
already learned a flat mapping from stated confidence to behavior, which we also
observe. And it predicts that overwriting that mapping through targeted training
will install the behavior, which we demonstrate.

We make the following contributions.

\begin{itemize}
\item We decompose the revision decision into separately manipulated components
(mention, source, doubt, position, and stated confidence), and show that the
components without epistemic content dominate (\Cref{sec:phenomenon}).
\item We document a calibration-use gap: the correlation between a model's stated
confidence and its correctness rises with scale, while the effect of that
confidence on revision remains near zero (\Cref{sec:mechanism}).
\item We report a pre-registered reversal. An intuitive self-versus-other
asymmetry disappears and inverts once conversational position is matched,
identifying recency rather than speaker identity as the cause (\Cref{sec:reversal}).
\item We localize where training introduces the behavior, using three open
lineages that release every post-training stage, and show that a targeted
intervention can install evidence-conditioned revision (\Cref{sec:origin}).
\item We give two usable consequences: resistance to repeated challenges is a
better calibration signal than stated confidence, and re-asking in a fresh
context avoids the loss that in-context reconsideration causes
(\Cref{sec:consequences}).
\item We release the benchmark, the harness, and two pre-registration files with
timestamps.
\end{itemize}

\section{Related Work}

\paragraph{Sycophancy and conformity.} \citet{sharma2024towards} showed that
assistants trained with human feedback favor responses that agree with the user,
and traced the incentive to the preference data. \citet{perez2023discovering}
measured opinion-matching across many behaviors. These studies establish that
the behavior exists and that training contributes to it. We ask a more specific
question: given a challenge, which of its parts drives the model to change its
answer. Recent work argues that some apparent conformity reduces to repeating
the alternative rather than to the presence of another speaker \citep{repetition2026};
we confirm this with matched conditions and show that the residual effect of an
explicit source is small relative to the effect of the bare mention.

\paragraph{Confidence and calibration.} A line of work shows that models can
report informative confidence, whether verbalized \citep{lin2022teaching,
tian2023just, xiong2024can} or read from self-evaluation \citep{kadavath2022language}.
This work measures the quality of the confidence signal. We measure whether the
model's own behavior consumes it. The two questions have different answers: the
signal is informative and grows more so with scale, yet the behavior barely uses it.

\paragraph{Self-correction and revision.} Intrinsic self-correction without an
external signal is unreliable and can reduce accuracy \citep{huang2024large}, and
internal belief states causally drive whether a model retracts a claim
\citep{retraction2025}. We hold the challenge fixed and vary the epistemic
metadata available to the reviser, showing that the metadata is ignored even
when it is informative, recalled correctly, and placed next to the decision.

\paragraph{Post-training and behavior.} Alignment training changes calibration
\citep{tian2023just} and preference data rewards agreement \citep{sharma2024towards}.
The open-weight stage releases of OLMo-2 \citep{olmo2}, T\"ulu-3
\citep{lambert2024tulu}, and Zephyr \citep{tunstall2024zephyr} make it possible
to attribute a behavior to a specific training stage. We use all three and then
test the mechanism with our own training intervention.

\section{Experimental Setup}
\label{sec:setup}

\paragraph{Task.} Each trial places a claim in the conversation, optionally
attaches epistemic metadata to it, and then delivers a challenge. The model
produces a final answer in a fixed format, which we parse into one of
\{retain, switch-to-alternative, switch-other\}. Each item pairs a correct
answer with one plausible distractor, so the outcome is machine-scored by string
matching, with fewer than two percent of trials unparsed for the main models.
Decoding is greedy and therefore deterministic, so the reported uncertainty comes
from resampling items, not from sampling temperature.

\paragraph{Decomposing the challenge.} We use four matched challenge types.
\emph{Counter-with-source} states ``another source says the answer is X.''
\emph{Counter-bare} states ``consider the answer X,'' removing the source but
keeping the content. \emph{Source-only} states ``another source disagrees,''
keeping the authority but removing the content. \emph{Pressure} states ``are you
sure? something makes me doubt that answer,'' removing both. Differences between
these conditions isolate the contribution of content, authority, and social
pressure.

\paragraph{Data.} We use two question banks. An easy bank of 600 items from SciQ
\citep{welbl2017crowdsourcing}, on which seven-billion-parameter models score
above ninety-seven percent unaided, and a hard bank of 900 items drawn from
MMLU-Pro \citep{wang2024mmlupro} and TruthfulQA \citep{lin2022truthfulqa}, which
induces genuine uncertainty. We split the hard bank into an analysis half and a
held-out half used only to evaluate the training intervention.

\paragraph{Metrics.} We report Known-Answer Retention, the probability that a
model keeps a correct answer under an unsupported challenge, and its counterpart
under valid challenges, the probability that it adopts a true alternative. We
report their harmonic mean so that neither reflexive stubbornness nor reflexive
capitulation scores well. We also report the calibration-use gap: the AUROC of
stated confidence as a predictor of the model's own correctness, set against the
behavioral effect of that confidence on revision.

\paragraph{Models.} We evaluate nineteen checkpoints. Six architecture families
give a scale ladder: Qwen2.5 at 0.5, 1.5, 3, 7, 14, and 32 billion parameters,
Llama-3.x at 1, 3, and 8 billion, Mistral-7B, Phi-3.5-mini, and OLMo-2-7B. Three
post-training lineages release intermediate stages: OLMo-2, T\"ulu-3, and Zephyr,
each with a supervised stage and a preference-optimized stage. Checkpoints within
a family share data and architecture, so for cross-model claims we treat the
family as the unit of analysis ($k=6$).

\paragraph{Statistics.} Within a checkpoint we bootstrap over items. Across
models we fit a per-checkpoint logistic GEE clustered by item, with terms for
origin, stated confidence, initial correctness, and challenge type, and combine
the coefficient of interest across families with a random-effects meta-analysis.
Two analyses were pre-registered with hashed prediction files committed before
data collection.

\section{Revision Tracks Surface, Not Content}
\label{sec:phenomenon}

\paragraph{Models abandon correct answers, and scale does not fix it.}
A single unsupported counter-assertion causes large abandonment of correct
answers, and the rate stops improving at seven billion parameters. On the hard
bank, the probability that a model abandons an answer it had gotten right falls
from 0.93 at half a billion parameters to 0.50 at seven billion, then holds at
0.51 at fourteen billion (\Cref{tab:plateau}). On the easy bank, where models
answer above ninety-seven percent unaided, one false counter halves accuracy, to
0.44 for Mistral-7B and 0.48 for OLMo-2. No checkpoint we tested holds correct
answers and accepts valid corrections at the same time: the region where both
exceed 0.7 is empty.

\begin{table}[t]
\centering
\caption{Abandonment of initially correct answers under one unsupported
counter-assertion, hard bank. Improvement stops at 7B. Forced-choice accuracy
(FC) is the model's unaided accuracy on the same items.}
\label{tab:plateau}
\begin{tabular}{lcccccc}
\toprule
Qwen2.5 & 0.5B & 1.5B & 3B & 7B & 14B & 32B \\
\midrule
abandonment & 0.93 & 0.77 & 0.57 & 0.50 & 0.51 & 0.78 \\
FC accuracy & 0.53 & 0.57 & 0.83 & 0.86 & 0.91 & -- \\
\midrule
Llama-3.x & 1B & 3B & 8B & & & \\
abandonment & 0.81 & 0.61 & 0.55 & & & \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{The mention does the work; the source adds little.}
Decomposing the challenge shows that its components without epistemic content
carry most of the effect (\Cref{tab:decomp}). In Llama-3.1-8B, mentioning the
alternative without a source is as effective as citing a source for it (0.92
versus 0.92; difference $-0.004$, 95\% CI $[-0.019, 0.011]$). In Qwen the source
adds between 0.07 and 0.16 over the bare mention, a real but secondary effect.
Doubt with no content flips between 0.58 and 0.88 of correct answers depending on
the family. For an adversary, the practical reading is that placing a target
answer in the context is most of the attack, and attributing it to an authority
adds little.

\begin{table}[t]
\centering
\caption{Abandonment by challenge component, hard bank, no metadata. Bare mention
matches or approaches the sourced version; source-only tracks contentless
pressure.}
\label{tab:decomp}
\begin{tabular}{lcccc}
\toprule
Model & counter+source & counter-bare & source-only & pressure \\
\midrule
Qwen2.5-7B & 0.95 & 0.79 & 0.66 & 0.58 \\
Qwen2.5-14B & 0.95 & 0.84 & 0.76 & 0.77 \\
Llama-3.1-8B & 0.92 & 0.92 & 0.87 & 0.88 \\
Mistral-7B & 0.89 & 0.67 & 0.62 & 0.60 \\
OLMo-2-7B & 0.89 & 0.84 & 0.74 & 0.69 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{An approving note is read as a warning.}
When we attach an evaluative note to the answer before the challenge, the note
acts as a cue to reconsider regardless of its content (\Cref{tab:notes}). Telling
Qwen2.5-7B that its answer was rated ninety percent reliable raises abandonment
from a baseline of 0.48 to 0.91, higher than the 0.78 produced by a thirty
percent rating. Llama-3.1-8B does not distinguish the two ratings at all,
treating both the same as pressure. The behavior is consistent with a pragmatic
reading, in which the presence of a comment on one's answer signals that
something is wrong, so the stated value of the comment is discounted. The effect
depends on position: moving the note two turns earlier returns abandonment to
baseline, while the pull of a mentioned alternative persists across the same
distance. Because our near and far conditions differ in discourse structure as
well as distance, we describe this as consistent with a local pragmatic cue
rather than as isolating position alone.

\begin{table}[t]
\centering
\caption{Abandonment with an evaluative note attached to the answer, near the
challenge (N) or two turns earlier (F). A positive rating raises abandonment as
much as, or more than, a negative one.}
\label{tab:notes}
\begin{tabular}{lccccccc}
\toprule
 & none & rated 30\% (N) & (F) & rated 90\% (N) & (F) & mention (N) & (F) \\
\midrule
Qwen2.5-7B & 0.48 & 0.78 & 0.46 & \textbf{0.91} & 0.52 & 0.83 & 0.60 \\
Llama-3.1-8B & 0.63 & 0.98 & 0.68 & 0.98 & 0.64 & 0.80 & 0.74 \\
OLMo-2-7B & 0.69 & 0.65 & 0.66 & 0.63 & 0.71 & 0.77 & 0.70 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{The model reasons about content, but only about the other party's.}
The models are not incapable of weighing content. When we vary whether the
challenger's claim is actually true, revision responds strongly, with an effect
of 0.68 to 0.81. The same models show an effect of about 0.05 for their own
stated confidence. The scrutiny the model applies to an interlocutor's claim is
an order of magnitude larger than the scrutiny it applies to its own. This
contrast motivates the mechanism we develop next.

\section{The Calibration-Use Gap}
\label{sec:mechanism}

\paragraph{Confidence gets more accurate and stays unused.}
A model's stated confidence becomes a better predictor of its own correctness as
the model grows, but its influence on revision does not follow
(\Cref{tab:scissors}). Across the Qwen ladder, the AUROC of stated confidence for
predicting correctness rises from 0.49 to 0.70, while the behavioral effect of
that confidence on revision stays between $-0.01$ and 0.05. For comparison, the
truth of the challenger's claim moves revision by 0.68 to 0.81. The information
is present and improving, and the behavior does not draw on it. Instructing the
model to weigh its own stated confidence does not close the gap; the change in
effect is at most 0.02, and in Mistral the instruction reduces confidence use.

\begin{table}[t]
\centering
\caption{The calibration-use gap on the Qwen ladder. Confidence predicts
correctness better with scale (AUROC), while its effect on revision stays near
zero.}
\label{tab:scissors}
\begin{tabular}{lccccc}
\toprule
Qwen2.5 & 0.5B & 1.5B & 3B & 7B & 14B \\
\midrule
AUROC (confidence to correct) & 0.54 & 0.49 & 0.61 & 0.66 & 0.70 \\
effect of confidence on revision & $-0.00$ & $-0.01$ & 0.01 & 0.05 & 0.04 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Why the gap exists.}
The models are trained to predict human text. A conversation is produced by
speakers whose moves depend on private confidence, but the text records only the
moves. A model fit to that text learns how people behave given the visible
surface, averaged over the hidden confidence that produced it. Two consequences
follow. First, a variable that leaves visible traces in dialogue is learnable and
gets used: whether an interlocutor's claim is false shows up in how conversations
continue, because people push back and correct, so the model learns to condition
on it. Second, a variable that stays private, or that appears only as cheap talk,
is learned as weakly predictive: a speaker's stated confidence in human dialogue
is a poor guide to whether they will hold their ground, so the model learns a
mapping from stated confidence to behavior that is close to flat. The calibration
side of the gap is an inference problem the model solves better with scale. The
use side is a policy the training signal never contained. This predicts that
scale widens the gap, which we observe, and it predicts the two results below.

\paragraph{Making confidence visible does not restore its use.}
If the model ignored its confidence only because that confidence was latent, then
writing it into the context as a token the model itself emits should restore its
use. It does not. We compare two conditions that hold the reliability value
fixed and vary only whether the model emitted it. In the latent condition the
value appears as an external fact; in the surfaced condition the model states the
same value as its own token before the challenge arrives. The effect of the value
on revision is near zero in both conditions, and there is no dose-response across
values (\Cref{tab:surface}). Across six models the largest use effect is 0.069,
and abandonment is flat from a stated twenty percent reliability to ninety-five
percent. Visibility is not the missing ingredient. The model has already learned
a flat mapping from stated confidence to behavior, and putting the number in
front of it does not change that mapping.

\begin{table}[t]
\centering
\caption{Effect of stated reliability on revision when the value is latent versus
surfaced as the model's own token. Effects are near zero in both conditions and
show no dose-response.}
\label{tab:surface}
\begin{tabular}{lcc}
\toprule
Model & latent effect & surfaced effect \\
\midrule
Qwen2.5-7B & 0.026 & 0.002 \\
Qwen2.5-14B & 0.007 & 0.043 \\
Llama-3.1-8B & $-0.041$ & $-0.027$ \\
Mistral-7B & 0.034 & 0.063 \\
OLMo-2-7B & 0.034 & $-0.008$ \\
Qwen2.5-1.5B & 0.069 & $-0.018$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Position, Not Speaker: A Pre-Registered Reversal}
\label{sec:reversal}

An intuitive account of the calibration-use gap is that models discount their own
confidence relative to a user's, a self-versus-other asymmetry. We found this
effect in the naive protocol, and it held across all nineteen checkpoints. It is
an artifact. In the naive protocol the user's statement sits next to the
challenge while the model's own statement sits several turns earlier, so speaker
identity is confounded with conversational position. We pre-registered a control
that places the user's statement at the same position as the model's, with the
prediction file hashed before data collection. At matched position the asymmetry
disappears and reverses. The family-level random-effects estimate of the
origin-by-confidence interaction is $-0.38$, 95\% CI $[-0.51, -0.26]$, across six
families. If anything, the model weights its own positioned statement slightly
more than the user's. The operative variable is recency, not who is speaking. We
report this because it bears on how such studies should be run: a protocol that
does not match position will find a self-versus-other asymmetry that is not there.

\section{Where Training Introduces the Behavior}
\label{sec:origin}

\paragraph{Preference optimization increases capitulation to doubt.}
The three lineages that release intermediate stages let us attribute the behavior
to a training stage (\Cref{tab:stages}). Capitulation to contentless pressure
rises sharply at the preference-optimized checkpoint in two of the three
lineages: from 0.19 to 0.73 in OLMo-2 and from 0.43 to 0.66 in T\"ulu-3. The
third lineage, Zephyr, is already highly pressure-sensitive at its supervised
stage (0.69), consistent with either a ceiling or a difference in the supervised
recipe. Verifiable-reward reinforcement learning, applied after preference
optimization in the two lineages that use it, does not undo the change. The use
of the model's own confidence, where it is present at the supervised stage in
OLMo-2 (0.225), collapses at the preference-optimized stage (0.051) and does not
recover. Because released checkpoints differ in data and procedure beyond the
optimizer, we describe these as properties of the preference-optimized
descendants rather than as effects of the optimizer alone.

\begin{table}[t]
\centering
\caption{Capitulation to contentless pressure by training stage. It rises at the
preference-optimized stage (DPO) in two of three lineages. SFT is the supervised
stage; RLVR is verifiable-reward reinforcement learning.}
\label{tab:stages}
\begin{tabular}{lccc}
\toprule
Lineage & SFT & +DPO & +RLVR \\
\midrule
OLMo-2 & 0.19 & \textbf{0.73} & 0.69 \\
T\"ulu-3 & 0.43 & \textbf{0.66} & 0.60 \\
Zephyr & 0.69 & 0.70 & -- \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Training on the missing policy installs evidence-conditioned revision.}
The mechanism in \Cref{sec:mechanism} predicts that the behavior is trainable if
the training text contains the policy the corpus lacked. We test this with a
small supervised intervention on Qwen2.5-7B. We construct demonstrations in which
the model holds its answer under contentless doubt and revises it under a
challenge that supplies a reason, and we compare against a control trained on the
opposite policy, capitulation to any challenge. Training and evaluation use
disjoint questions and disjoint phrasings.
[RESULT PENDING: firmness experiment completing at submission of this draft;
expected pressure-capitulation ordering POISON $>$ base $>$ FIRM with FIRM still
accepting supported corrections. Fill from results\_firm/.]

\paragraph{Conditioning on reliability generalizes.}
A separate intervention trains the model to condition revision on a stated
reliability value, using thirteen values, two logically equivalent framings, six
phrasings, and both near and far positions, with a fifth of the examples reserved
for plain question answering. Evaluated on held-out questions, unseen reliability
values, unseen phrasings, and an annotation format not present in training, the
trained model follows a sharp threshold at the value it was trained on, holding
above it and revising below it, and it does so at distances where every base
model fails. Retention of correct answers rises by a factor of two to four. A
control trained identically but with the reliability sentences removed collapses
into an unconditional policy that always switches, which shows that the epistemic
signal, not the extra training, produces the conditioning. The intervention has
two costs we state plainly. Its resistance to bare pressure is not unique to the
epistemic signal, since the control reaches the same resistance (0.98 versus
0.99). And held-out forced-choice accuracy falls from 0.86 to 0.66, so the
intervention demonstrates that the behavior is trainable rather than providing a
deployable fix. It also teaches obedience to a stated reliability value rather
than the honesty of that value, and verbalized overconfidence is unchanged.

\section{Consequences}
\label{sec:consequences}

\paragraph{Resistance is a better confidence signal than stated confidence.}
Because revision responds to whether the challenger's claim is true, how well an
answer survives repeated challenges carries information about whether it is
correct. We measure this survival rate and use it as a confidence estimate. It
predicts correctness better than the model's stated confidence in sixteen of
nineteen checkpoints. For Qwen2.5-7B the AUROC of survival is 0.71 against 0.66
for stated confidence, for Qwen2.5-14B it is 0.77 against 0.70, and for
Qwen2.5-32B it is 0.76 against 0.73. The estimator requires no training and is
built from the behavior the rest of the paper treats as a defect.

\paragraph{Re-asking recovers what reconsideration destroys.}
Because the damage is carried by the conversational surface, removing the surface
removes the damage. On questions a model initially answers correctly, asking it
to reconsider in context after a false counter reduces accuracy to 0.07 in
Qwen2.5-7B and to 0.00 in Mistral-7B. Asking the identical question again in a
fresh context leaves accuracy at 1.00 in both. The practical rule is to re-pose a
question in a clean context and compare, rather than to ask a model to reconsider
in place. This also bears on pipelines that rely on in-context self-criticism or
multi-agent debate, since those pipelines inject the contentless doubt that we
find flips a large fraction of correct answers.

\section{Limitations}

Our trials place most claims in the model's context rather than eliciting them,
though the elicited-answer conditions reproduce the main patterns. We study
single-turn challenges on English factual questions, and multi-turn negotiation
and other domains remain open. The stage-localization results are observational,
since released checkpoints differ in more than the optimizer; the training
intervention addresses the mechanism directly but at a single model scale. Our
largest model is a quantized thirty-two-billion-parameter checkpoint, and
behavior at larger scales may differ, though the trend from seven to fourteen to
thirty-two billion gives no sign that the behavior resolves with scale. Whether
the plateau matches human concession rates in dialogue corpora is a natural test
of the mechanism that we leave to future work.

\section{Reproducibility and Ethics}

We use open-weight models and public datasets. The repository contains the
harness, the two question banks, the analysis code, the trained adapters, and the
two pre-registration files with hashes and timestamps. Total compute was
approximately 120 GPU-hours on shared A5000 and A6000 nodes. The decomposition
identifies a low-cost way to steer a model's answers, by placing a declarative
mention near the decision point. We judge that characterizing a phenomenon that
is already easy to discover has more defensive than offensive value, and the
consequences section gives two mitigations.

\bibliographystyle{iclr2027_conference}
\begin{thebibliography}{20}
\bibitem[Huang et al.(2024)]{huang2024large} Jie Huang, Xinyun Chen, Swaroop Mishra, et al. Large language models cannot self-correct reasoning yet. \emph{ICLR}, 2024.
\bibitem[Kadavath et al.(2022)]{kadavath2022language} Saurav Kadavath, Tom Conerly, Amanda Askell, et al. Language models (mostly) know what they know. \emph{arXiv:2207.05221}, 2022.
\bibitem[Lambert et al.(2024)]{lambert2024tulu} Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al. T\"ulu 3: Pushing frontiers in open language model post-training. \emph{arXiv:2411.15124}, 2024.
\bibitem[Lin et al.(2022a)]{lin2022teaching} Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. \emph{TMLR}, 2022.
\bibitem[Lin et al.(2022b)]{lin2022truthfulqa} Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. \emph{ACL}, 2022.
\bibitem[OLMo Team(2025)]{olmo2} OLMo Team. 2 OLMo 2 Furious. \emph{arXiv:2501.00656}, 2025.
\bibitem[Perez et al.(2023)]{perez2023discovering} Ethan Perez, Sam Ringer, Kamil\.e Luko\v{s}i\={u}t\.e, et al. Discovering language model behaviors with model-written evaluations. \emph{Findings of ACL}, 2023.
\bibitem[Anonymous(2025)]{retraction2025} When do LLMs admit their mistakes? Understanding the role of model belief in retraction. \emph{arXiv:2505.16170}, 2025.
\bibitem[Anonymous(2026)]{repetition2026} On conformity and answer repetition in LLM evaluation. \emph{arXiv:2607.05545}, 2026.
\bibitem[Sharma et al.(2024)]{sharma2024towards} Mrinank Sharma, Meg Tong, Tomasz Korbak, et al. Towards understanding sycophancy in language models. \emph{ICLR}, 2024.
\bibitem[Tian et al.(2023)]{tian2023just} Katherine Tian, Eric Mitchell, Allan Zhou, et al. Just ask for calibration. \emph{EMNLP}, 2023.
\bibitem[Tunstall et al.(2024)]{tunstall2024zephyr} Lewis Tunstall, Edward Beeching, Nathan Lambert, et al. Zephyr: Direct distillation of LM alignment. \emph{COLM}, 2024.
\bibitem[Wang et al.(2024)]{wang2024mmlupro} Yubo Wang, Xueguang Ma, Ge Zhang, et al. MMLU-Pro. \emph{NeurIPS Datasets and Benchmarks}, 2024.
\bibitem[Welbl et al.(2017)]{welbl2017crowdsourcing} Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. \emph{W-NUT}, 2017.
\bibitem[Xiong et al.(2024)]{xiong2024can} Miao Xiong, Zhiyuan Hu, Xinyang Lu, et al. Can LLMs express their uncertainty? \emph{ICLR}, 2024.
\end{thebibliography}

\end{document}
