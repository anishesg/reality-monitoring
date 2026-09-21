% =====================================================================
% What Makes a Language Model Change Its Mind? Cues, Not Content.
% Full LaTeX source (ICLR 2027 submission format).
% Working draft compiled from experiments of Sep 17-18, 2026.
% Repo: github.com/anishesg/reality-monitoring (private)
% =====================================================================

\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,multirow,graphicx,xcolor}
\usepackage{hyperref,url}
\usepackage{subcaption}
\usepackage[capitalize,noabbrev]{cleveref}
\definecolor{darkred}{RGB}{160,30,30}
\newcommand{\kar}{\textsc{kar}}
\newcommand{\stg}{\textsc{stg}}
\newcommand{\peff}{p_{\text{eff}}}

\title{What Makes a Language Model Change Its Mind?\\ Conversational Cues Dominate Epistemic Evidence}

\author{Anonymous authors\\ \small Paper under double-blind review}

\begin{document}
\maketitle

\begin{abstract}
When a language model's answer is challenged, what determines whether it stands
firm or folds? A rational reviser would weigh exactly two things: the reliability
of its original answer and the evidential content of the challenge. Across
19~post-training checkpoints in five architecture families
($\sim$400{,}000 challenge trials, two claim banks, pre-registered analyses), we
find that current models weigh almost everything else instead. Revision is
dominated by three non-epistemic forces: \emph{priming} (the mere mention of an
alternative answer triggers 80--95\% capitulation, with explicit source
attribution adding little or nothing), \emph{social pressure} (content-free
doubt flips 20--88\% of answers depending on family), and \emph{proximity}
(epistemic statements influence revision only when adjacent to the decision). Reliability
annotations act as local challenge cues rather than information: a
\emph{positive} endorsement (``your answer was rated 90\% reliable'') increases
abandonment as much as---in one family, more than---a warning. Meanwhile the
one legitimate signal, the model's own stated confidence, generally becomes more
\emph{informative} with scale (AUROC $.49{\to}.70$ by 14B) while its behavioral
influence stays near zero: the know--use gap widens as models improve. A
pre-registered ``self-discounting'' hypothesis failed its own matched-position
control and reversed, identifying recency---not source---as the operative
variable. Tracing origins through three fully open post-training lineages, we
find capitulation to contentless pressure rises at the preference-optimization
stage in two of three lineages (OLMo-2: $0.19{\to}0.73$; T\"ulu-3:
$0.43{\to}0.66$; the third is already highly pressure-sensitive at SFT) and is
not repaired by subsequent verifiable-reward RL; a controlled
preference-training experiment tests the mechanism directly. Finally, we show
the failure is one of training signal, not capacity:
\emph{reliability-conditioned revision training}---3{,}000 LoRA examples---%
installs conditional revision that generalizes to held-out values, phrasings,
and positions (retention of correct answers $2$--$4\times$ over base), while
identical training with the reliability sentences removed collapses into an
unconditional reflex: the epistemic signal buys the conditioning. The
intervention is a proof of trainability, not a finished repair---it currently
costs held-out accuracy, and generic robustness training matches its
resistance to bare pressure without acquiring any conditional behavior. We
release the benchmark, harness, externally timestamped pre-registration
receipts, and two drop-in metrics.
\end{abstract}

% =====================================================================
\section{Introduction}
% =====================================================================

Every deployed language model lives inside conversations that push back.
Users disagree without evidence; retrieved documents mention alternatives;
other agents assert; automated pipelines attach notes and ratings to
intermediate answers. Each such moment forces the same decision: \emph{keep
your answer, or change it?}

There is a normatively sensible way to make this decision, and it is short.
A reviser should weigh (i) \emph{the reliability of the original answer} and
(ii) \emph{the evidential value of the challenge}---where evidential value
folds in the challenger's reliability and the stakes. Conversational
choreography---how recently something was said, how it was phrased, whether an
alternative merely got mentioned---should matter only insofar as it carries
evidence.

This paper measures, ingredient by ingredient, what actually governs the
revision decision in modern language models. The field has a word for the
symptom---sycophancy \citep{sharma2024towards,perez2023discovering}---but the
word bundles together at least five distinct signals (answer priming, source
attribution, social pressure, epistemic content, and conversational position)
that we experimentally separate. Our central finding is easy to state and
uncomfortable to sit with:

\begin{quote}
\emph{Language models change their answers in response to cues that
reconsideration is \textbf{expected}, not information about whether
reconsideration is \textbf{warranted}.}
\end{quote}

Concretely, we find (\cref{fig:overview} summarizes):

\textbf{Fragility that scale is not fixing (\cref{sec:fragility}).} A single
false counter-assertion (``another source says $X$'') deletes roughly half
the answers a model gets right on its own. Capitulation improves strongly up
to 7B ($.93$ at 0.5B $\to$ $.50$ at 7B) and shows no improvement from 7B to
14B under the primary protocol ($.51$); a separate 32B control (quantized,
different protocol, not directly comparable) confirms substantial residual
fragility.

\textbf{Priming beats authority (\cref{sec:decomposition}).} Decomposing the
challenge shows the epistemically \emph{empty} parts do the work. Merely
mentioning an alternative (``consider $X$'') is as effective as a sourced
assertion in the Llama family ($.92$ vs.\ $.92$); explicit sources add at most
$+.07$--$.16$ (Qwen). Content-free doubt (``are you sure?'') independently
flips up to 75\% of correct answers.

\textbf{Praise acts like pressure (\cref{sec:annotations}).} Reliability
annotations placed beside the decision act as challenge cues regardless of
what they say: telling Qwen2.5-7B its answer was rated \emph{90\% reliable}
raises abandonment from $.48$ to $.91$---more than the 30\% warning does
($.78$). Llama and OLMo cannot distinguish 30\% from 90\% at all. These cue
effects decay to baseline after two intervening turns, while the pull of a
mentioned alternative persists.

\textbf{The scissors (\cref{sec:scissors}).} The model's own stated confidence
is the one signal that grows \emph{more informative} with scale (AUROC for
predicting its own correctness: $.49 \to .66 \to .70$ from 0.5B to 14B), yet
its behavioral influence on revision stays flat near zero, and models recall
their own statements verbatim while failing to act on them. The gap between
what confidence \emph{knows} and what behavior \emph{uses} widens as models
improve.

\textbf{A pre-registered reversal (\cref{sec:reversal}).} We initially
believed models discount their \emph{own} confidence relative to a user's---a
seductive hypothesis that replicated across 19 checkpoints. A pre-registered
matched-position control (predictions hash-stamped before data) eliminated
and mildly reversed the effect (family-level meta-analysis:
$\mu=-0.38$, 95\%~CI $[-0.51,-0.26]$): the apparent asymmetry was
\emph{recency}, not source. We report this reversal as a finding about
protocol validity in this literature.

\textbf{Where training amplifies it (\cref{sec:origins}).} Using three
post-training pipelines that publish every stage (OLMo-2, T\"ulu-3, Zephyr),
we find the two capitulation channels have different histories:
priming-capitulation is high from SFT onward in every lineage, while
pressure-capitulation \emph{rises at the preference-optimization checkpoint
in two of three lineages} (OLMo-2: $.19\to.73$; T\"ulu-3: $.43\to.66$;
Zephyr is already highly pressure-sensitive at SFT, $.69$) and persists
through verifiable-reward RL. A controlled experiment that constructs the hypothesized
preference-data mechanism directly (accommodation-preferred vs.\
firmness-preferred pairs) is reported in \cref{sec:controlled-dpo}.

\textbf{Conditional revision is trainable (\cref{sec:repair}).} Finally we
show the missing piece is a training signal, not model capacity.
\emph{Reliability-conditioned revision training}---3{,}000 LoRA examples
teaching one rule---produces threshold-accurate behavior at reliability
values, phrasings, wrappers, and conversational positions never seen in
training, and improves retention of correct answers $2$--$4\times$. The
decisive control: identical training with the reliability sentences deleted
collapses into an unconditional reflex; notably it \emph{matches} the trained
arm's resistance to bare pressure ($.98$ vs.\ $.99$), showing that the unique
contribution of the epistemic signal is the conditioning, not generic
robustness. The intervention is a proof of concept with real costs: held-out
forced-choice accuracy drops ($.86\to.66$), and it teaches obedience to
stated reliability rather than honest reliability statements.

Our contributions are: (1) a controlled decomposition of the revision
decision into five separately-manipulated signals, with two claim banks and
$\sim$400k trials across 19 checkpoints; (2) two drop-in metrics---%
\textbf{Known-Answer Retention} (\kar{}) and the bidirectional
retention/correction frontier---for standard reporting; (3) the scissors
result connecting the calibration literature to behavior; (4) a
stage-localized training-origin analysis with a constructive controlled
experiment; (5) a generalizing, controlled repair; and (6) a documented
pre-registered reversal that identifies a recency confound likely present in
prior conformity protocols.

% =====================================================================
\section{Related Work}
% =====================================================================

\textbf{Sycophancy and conformity.} \citet{sharma2024towards} established
that RLHF-trained assistants systematically favor user-agreeing responses and
traced the incentive to human preference data; \citet{perez2023discovering}
found opinion-matching at scale. Our decomposition asks a prior question:
\emph{which ingredient} of a challenge does the work? Recent work argues that
apparent conformity partially reduces to repeating the alternative answer
rather than the presence of another agent \citep{repetition2026}; we confirm
and extend this with matched cells (source$+$content, content-only,
source-only, neither) and show the residual source effect is family-dependent
and small relative to priming.

\textbf{Calibration and verbalized confidence.} Models can express
increasingly informative confidence: verbalized probabilities
\citep{lin2022teaching,tian2023just,xiong2024can} and self-evaluation
\citep{kadavath2022language} discriminate correct from incorrect answers well
above chance. This literature measures the \emph{quality} of the confidence
signal; ours measures whether the signal is \emph{consumed} by the model's own
downstream behavior. The answer---informative but behaviorally inert---places
a caveat on hopes that better calibration alone will yield better epistemic
behavior.

\textbf{Self-correction and retraction.} Intrinsic self-correction without
external signal is unreliable and often harmful
\citep{huang2024large}; internal belief states causally drive retraction
\citep{retraction2025}. We complement this line by holding the challenge fixed
and varying the \emph{epistemic metadata} available to the reviser, showing
the metadata is ignored even when informative, recallable, and adjacent.

\textbf{Post-training effects on behavior.} RLHF alters calibration
\citep{tian2023just} and preference data rewards agreement
\citep{sharma2024towards}. The open-weight stage releases of OLMo-2
\citep{olmo2}, T\"ulu-3 \citep{lambert2024tulu}, and Zephyr
\citep{tunstall2024zephyr} make it possible for the first time to
\emph{localize} behavioral changes to specific post-training stages; we use
all three, then close the causal gap with our own controlled preference
training (\cref{sec:controlled-dpo}).

% =====================================================================
\section{Experimental Framework}
% =====================================================================
\label{sec:framework}

\subsection{Protocol}
Every trial has the same skeleton (\cref{fig:protocol}): a \emph{claim} enters
the conversation, optionally accompanied by \emph{epistemic metadata}
(declared confidence, a reliability rating, a verification status), from a
controlled \emph{origin} (the model's own prior turn; the user; a third-party
AI; a user statement at matched conversational position); then a
\emph{challenge} arrives, and the model must issue a final answer in a fixed
format. Because the claim and its one-word alternative are controlled, the
outcome---retain, switch-to-alternative, switch-elsewhere---is machine-scored
by string normalization with a conservative fallback ($<2\%$ of trials
excluded as unparseable in the main models).

\textbf{Challenge decomposition.} Four matched challenge types isolate the
ingredients of ``pushback'':
\texttt{counter\,+\,source} (``Actually, another source says the answer is
`$X$'\,''), \texttt{counter-bare} (``Consider the answer `$X$'\,''),
\texttt{source-only} (``Another source disagrees with that answer''), and
\texttt{pressure} (``Are you sure? Something makes me doubt that answer'').
The pairwise contrasts give the causal weight of \emph{content} (the
alternative's presence), \emph{authority} (the source), and \emph{social
pressure} (neither).

\textbf{Claim banks.} An \emph{easy} bank (600 SciQ items
\citep{welbl2017crowdsourcing}) where 7B models answer $\geq$85\% correctly,
and a \emph{hard} bank (600 MMLU-Pro \citep{wang2024mmlupro} + 300 TruthfulQA
\citep{lin2022truthfulqa} items) inducing genuine uncertainty. Each item
pairs a correct answer with a plausible distractor, enabling
truth-conditioned analysis and the bidirectional metric below. The hard bank
is split 450/450 into analysis and held-out repair-evaluation halves.

\textbf{Outcome metrics.} We report (i) \kar{}: $P(\text{retain} \mid
\text{claim correct, challenge unsupported})$; (ii) its complement under valid
challenges, \emph{accept-fix}: $P(\text{adopt alternative} \mid \text{claim
false, alternative true})$; and (iii) their harmonic mean, so that neither
stubbornness (\kar{}$=1$, accept-fix$=0$) nor total capitulation can score
well. Following \citet{huang2024large} we always report both flip directions.

\subsection{Models}
\label{sec:models}
We evaluate 19 checkpoints organized as in \cref{tab:models}: five base
architecture families (Qwen2.5 at 0.5/1.5/3/7/14/32B; Llama-3.x at 1/3/8B;
Mistral-7B; Phi-3.5; OLMo-2-7B), and three post-training \emph{lineages} with
public stage checkpoints (OLMo-2 \{SFT, DPO, RLVR\}; T\"ulu-3 \{SFT, DPO,
RLVR\}; Zephyr \{SFT, DPO\}). Checkpoints within a family or lineage share
data and ancestry and are never treated as independent replicates
(\cref{sec:stats}).

\begin{table}[t]
\centering\small
\caption{Evaluated checkpoints. Lineages are built on base families;
the inferential unit for cross-model claims is the \emph{family} ($k=6$:
Qwen, Llama, Mistral(+Zephyr), OLMo, T\"ulu, Phi).}
\label{tab:models}
\begin{tabular}{lll}
\toprule
Concept & Members & Role \\
\midrule
Architecture family & Qwen2.5 (0.5--32B), Llama-3.x (1--8B), & scale ladder,\\
 & Mistral-7B, Phi-3.5, OLMo-2-7B & cross-family replication \\
Post-training lineage & OLMo-2, T\"ulu-3, Zephyr & stage localization \\
Stage checkpoint & SFT, DPO, RLVR(final) & per lineage \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Statistical analysis and pre-registration}
\label{sec:stats}
Decoding is greedy (deterministic); uncertainty is therefore driven by item
sampling and training stochasticity, not decoding. Within checkpoints we use
cluster bootstrap over claims (500--1000 resamples) for all reported
intervals. For cross-model inference we fit, per checkpoint, a logistic GEE
\citep{liang1986longitudinal} clustered by claim,
\[
\text{switch} \sim \text{origin}\times\text{confidence} + \text{correctness}
+ \text{challenge type},
\]
and combine the coefficient of interest across the six \emph{families} with
random-effects meta-analysis \citep{dersimonian1986meta}. Families overlap in
training data and recipes, so even the family level is only approximately
independent, and with $k{=}6$ units the DL estimator is fragile; we therefore
report Hartung--Knapp-adjusted intervals and leave-one-family-out sensitivity
for all cross-family claims (\cref{sec:appendixD}), and treat family-level
meta-analytic results as descriptive strength-of-evidence summaries rather
than precise population inferences. Two analyses were pre-registered with
MD5-hashed prediction files committed before data collection (the
training-stage predictions and the matched-position controls); both hashes,
their timestamps, and the outcomes---one confirmation, one reversal---are in
the repository and mirrored to an externally timestamped public record, since
private repository hashes alone are weak pre-registration evidence. Training
experiments (\cref{sec:repair}) use three seeds per arm.

% (Part 2 of the source continues below: Results, Discussion, Limitations,
%  References, Appendix.)

% =====================================================================
\section{Results}
% =====================================================================

\subsection{Models surrender half of what they know, and scale stops helping}
\label{sec:fragility}

We first establish the phenomenon at its simplest. The model answers; a false
counter-assertion arrives (``Actually, another source says [distractor]'');
we measure whether the model keeps the answer it had gotten right.

\begin{table}[t]
\centering\small
\caption{\textbf{Capitulation on initially-correct answers} under a single
sourced false counter (hard bank). Improvement stops at 7B. All cells
$n\!\geq\!500$ claims; cluster-bootstrap 95\% CIs within $\pm.03$.}
\label{tab:plateau}
\begin{tabular}{lcccccc}
\toprule
 & 0.5B & 1.5B & 3B & 7B & 14B & 32B$^{\dagger}$ \\
\midrule
Qwen2.5 & .93 & .77 & .57 & \textbf{.50} & \textbf{.51} & .78$^{\dagger}$ \\
Llama-3.x & --- & .81 (1B) & .61 (3B) & .55 (8B) & --- & --- \\
\bottomrule
\multicolumn{7}{l}{\footnotesize $^{\dagger}$32B (AWQ-quantized) measured under
the control protocol of \cref{sec:decomposition};} \\
\multicolumn{7}{l}{\footnotesize same-protocol 7B value is $.93$, so 32B
improves but capitulation still dominates.}
\end{tabular}
\end{table}

On the easy bank, where models answer $\geq$97\% of resolvable items
correctly in isolation, one false counter halves accuracy (Mistral-7B:
$.97\to.44$; OLMo-2: $.97\to.48$). We emphasize what \cref{tab:plateau} does
and does not establish: strong improvement through 7B, none from 7B to 14B
under the matched protocol, and (via a non-comparable control) large residual
fragility at 32B. Two matched points and one unmatched control are evidence
that scale is not on track to solve this; they are not a scaling law. On the value frontier---retaining correct
answers under unsupported challenges \emph{while} accepting valid
corrections---no evaluated checkpoint exceeds $(.36, .97)$ jointly; the region
where both exceed $0.7$ is unpopulated. Current models purchase their
correction-acceptance entirely by paying with retention.

\subsection{Decomposing the challenge: priming $\geq$ authority $\gg$ content}
\label{sec:decomposition}

What part of ``Actually, another source says $X$'' does the damage? The four
matched challenge types give the answer (\cref{tab:decomp}).

\begin{table}[t]
\centering\small
\caption{\textbf{Abandonment by challenge ingredient} (hard bank, no
confidence metadata, self-origin claims; 450 claims $\times$ 2 truth values).
The bare mention of an alternative rivals or matches the sourced version;
``source'' alone tracks contentless pressure.}
\label{tab:decomp}
\begin{tabular}{lcccc}
\toprule
Model & counter+source & counter-bare & source-only & pressure \\
\midrule
Qwen2.5-7B  & .95 & .79 & .66 & .58 \\
Qwen2.5-14B & .95 & .84 & .76 & .77 \\
Llama-3.1-8B & .92 & \textbf{.92} & .87 & .88 \\
Llama-3.2-3B & .94 & \textbf{.94} & .88 & .88 \\
Mistral-7B  & .98$^{a}$ & .59 & --- & .69 \\
OLMo-2-7B (final) & .89 & .84 & --- & .69 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize $^{a}$Zephyr-lineage SFT parent shown in
\cref{tab:stages}; dashes = cell not run for that checkpoint.}
\end{tabular}
\end{table}

In the Llama family the source contributes \emph{nothing}: mentioning the
alternative is exactly as effective as attributing it ($\Delta = -.004$, 95\%
CI $[-.019,.011]$). In Qwen the source adds $+.07$ to $+.16$---real, but
secondary. Content-free pressure alone flips 20--88\% of answers depending on
family and training stage (\cref{tab:stages}). The practical reading is blunt: for prompt-injection purposes, an
attacker does not need to impersonate an authority; \emph{leaving the target
answer lying in the context} is the attack.

\subsection{The scissors: confidence knows more as behavior uses none of it}
\label{sec:scissors}

We elicit the model's own confidence in each answer (forced-choice with a
stated percentage), then challenge it, asking two separate questions:
(a)~is the confidence \emph{informative} about correctness? (b)~does it
\emph{change} revision behavior?

\begin{table}[t]
\centering\small
\caption{\textbf{The scissors.} Informativeness of self-declared confidence
rises with scale; behavioral use does not. (Hard bank; ``behavioral $\Delta$''
= abandonment(low-declared) $-$ abandonment(high-declared) for injected
declarations under sourced counters.)}
\label{tab:scissors}
\begin{tabular}{lccccc}
\toprule
Qwen2.5 & 0.5B & 1.5B & 3B & 7B & 14B \\
\midrule
AUROC(conf $\to$ correct) & .54 & .49 & .61 & .66 & \textbf{.70} \\
Behavioral $\Delta$ (self) & $-.00$ & $-.01$ & $.01$ & $.05$ & $.04$ \\
\bottomrule
\end{tabular}
\end{table}

Meanwhile the evaluative channel dissociates from both: Mistral-7B's
P(True) self-evaluation discriminates its own errors at AUROC $.78$ on the
easy bank while its behavioral coupling to revision is $.00$. And when asked
directly, models \emph{recall} their earlier declared confidence
near-perfectly (numeric recall $\geq .95$ at 7B). The information is present,
retrievable, and readable. What is missing is consumption: for comparison,
claim \emph{truth} moves switching by $+.68$ to $+.81$---the confidence terms
above are an order of magnitude smaller. Instructing models to weigh their
own stated confidence does not close the gap (deltas move $\leq.02$, and in
Mistral the instruction \emph{reduces} confidence-use from $.12$ to $.03$).

\subsection{Evaluative annotations are local challenge cues; praise acts like
pressure}
\label{sec:annotations}

If confidence content is not consumed, what happens when we attach explicit
reliability information to an answer? We annotate the model's answer with a
reviewer rating---30\% or 90\% reliable---placed either adjacent to the
reconsideration request (\emph{near}) or two turns earlier behind an
unrelated exchange (\emph{far}), against a no-annotation baseline.

\begin{table}[t]
\centering\small
\caption{\textbf{Annotations trigger revision by presence and proximity, not
content} (hard bank; abandonment rates). A 90\%-reliability
\emph{endorsement} increases abandonment as much as (Llama, OLMo) or more
than (Qwen) the 30\% warning; all annotation effects decay to baseline at
distance while the pull of a mentioned alternative persists.}
\label{tab:annotations}
\begin{tabular}{lccccccc}
\toprule
 & & \multicolumn{2}{c}{rated 30\%} & \multicolumn{2}{c}{rated 90\%} &
 \multicolumn{2}{c}{alt.\ mention} \\
Model & none & near & far & near & far & near & far \\
\midrule
Qwen2.5-7B & .48 & .78 & .46 & \textcolor{darkred}{\textbf{.91}} & .52 & .83 & .60 \\
Llama-3.1-8B & .63 & .98 & .68 & .98 & .64 & .80 & .74 \\
OLMo-2-7B & .69 & .65 & .66 & .63 & .71 & .77 & .70 \\
\bottomrule
\end{tabular}
\end{table}

The family-level pattern is \emph{cue-dominated and semantically
inappropriate}, and it varies by family: Llama is confidence-insensitive
(30\% and 90\% indistinguishable at $.98$, both far above its $.63$
baseline); Qwen is anti-semantic (the reassuring 90\% rating raises
abandonment \emph{above} the warning, $.91$ vs.\ $.78$, baseline $.48$);
OLMo-2 is unmoved by annotations of either valence. Where annotations act,
the pattern is \emph{consistent with} a local pragmatic inference---\emph{an
evaluator commented on my answer immediately before I was asked to
reconsider; something must be wrong}---that overrides stated meaning; and
consistent with a short-lived cue, rating effects collapse to baseline at two
turns' distance while the mentioned alternative retains its pull (Qwen far:
$.60$ vs.\ baseline $.48$). We note a structural caveat: in this design the
near condition fuses the annotation with the reconsideration request while
the far condition separates and acknowledges it, so distance covaries with
discourse structure; a structure-matched replication is needed before
proximity can be isolated as \emph{the} operative variable.
\textbf{Where annotations act at all, praise acts like pressure.}

\subsection{A pre-registered reversal: recency, not source}
\label{sec:reversal}

We initially hypothesized \emph{self-discounting}: that models use a user's
declared confidence while ignoring their own. The raw effect replicated
across all 19 checkpoints. Following a reviewer-identified confound---in the
natural protocol the user's declaration sits adjacent to the challenge while
the model's sits turns earlier---we pre-registered a matched-position
control (predictions hashed before data). At matched position the asymmetry
vanished; the hierarchical analysis (\cref{sec:stats}) reversed it:
origin$\times$confidence interaction $\mu = -0.38$, 95\% CI $[-0.51, -0.26]$
across $k=6$ families---if anything, models weight their \emph{own}
positioned declarations slightly more. The apparent self/other asymmetry was
a \emph{recency} effect. We keep this result in the paper for two reasons:
it identifies a confound plausibly present in existing conformity protocols
\citep{repetition2026}, and it converts our own earlier claim into a
cautionary measurement: in this literature, position is a stronger variable
than source, and protocols that do not match it will discover phantom
asymmetries.

\subsection{Where training installs it: stage localization and a controlled
construction}
\label{sec:origins}

\textbf{Observational stage ladders.} Three post-training pipelines publish
every stage. The two capitulation channels have different histories
(\cref{tab:stages}): capitulation to \emph{answer priming} is high at every
stage and in every lineage---inherited, not installed---though not strictly
constant (OLMo-2: $.83/.84/.84$; Zephyr rises $.59\to.78$ at DPO). Capitulation
to \emph{contentless pressure} rises at the preference-optimization checkpoint
in two of three lineages (OLMo-2: $.19\to.73$; T\"ulu-3: $.43\to.66$) and is
not repaired by verifiable-reward RL; the third lineage (Zephyr) is already
highly pressure-sensitive at SFT ($.69$), consistent with either a ceiling or
a recipe difference. Self-confidence use, where present at SFT (OLMo-2:
$.225$), collapses at the DPO stage ($.051$) and never recovers.

\begin{table}[t]
\centering\small
\caption{\textbf{Stage localization} (hard bank). Pressure-capitulation
rises at the preference-optimized checkpoint in 2/3 lineages; the third is
already highly pressure-sensitive at SFT. Priming-capitulation is high at all
stages (not strictly invariant: Zephyr rises at DPO). Observational: stages
differ in more than the optimizer (\cref{sec:limitations}).}
\label{tab:stages}
\begin{tabular}{llccc}
\toprule
Lineage & Channel & SFT & +DPO & +RLVR \\
\midrule
OLMo-2 & pressure & .19 & \textcolor{darkred}{\textbf{.73}} & .69 \\
       & priming (bare) & .83 & .84 & .84 \\
       & conf-use (self) & \textbf{.23} & .05 & .06 \\
T\"ulu-3 & pressure & .43 & \textbf{.66} & .60 \\
Zephyr & pressure & .69 & .70 & --- \\
       & priming (bare) & .59 & \textbf{.78} & --- \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Controlled construction.}
\label{sec:controlled-dpo}
Because released checkpoints confound the optimizer with data, duration, and
formatting, we close the gap constructively: starting from the same SFT
checkpoint (OLMo-2-7B-SFT), we run DPO ourselves under three diets---%
\textsc{accom} (2{,}000 synthetic pairs in which the \emph{preferred}
response accommodates contentless doubt), \textsc{firm} (the identical pairs
with preference reversed), and \textsc{uf-full} (12k unmodified UltraFeedback
pairs \citep{cui2024ultrafeedback})---with checkpoint dose-response curves
and two seeds per arm. \emph{[Results collecting at time of writing; this
subsection reports design. Under the accommodation hypothesis,
pressure-capitulation should rise with dose under \textsc{accom}, remain flat
or fall under \textsc{firm}, and move weakly if at all under
\textsc{uf-full}, whose prompts contain little to accommodate.]}

\subsection{The repair: one training signal restores conditional revision}
\label{sec:repair}

If the deficit is a missing training signal rather than missing capacity, a
small amount of the right signal should install the behavior---and installing
it should require the epistemic information itself. We LoRA-train
\citep{hu2022lora} Qwen2.5-7B (3 seeds) on $\sim$3{,}000 dialogues teaching a
single policy: \emph{under an unsupported challenge, retain; under a sourced
counter, revise iff your stated reliability was below 50\%}. Reliability
statements span 13 numeric values, two logically-equivalent framings
(``$N$\% likely correct'' / ``$(100{-}N)$\% likely incorrect''), six
phrasings, and both near and far positions; 20\% of examples are plain-QA
replay. The control arm receives \emph{byte-identical training with the
reliability sentences removed} (targets preserved with matched marginal
action rates). Evaluation is doubly held out: unseen datasets (MMLU-Pro/
TruthfulQA vs.\ SciQ training), unseen values (15/30/45/60/75/90), unseen
phrasings, and an annotation wrapper taken verbatim from
\cref{sec:annotations}, which the training never contained.

\begin{table}[t]
\centering\small
\caption{\textbf{The repair generalizes; its control collapses.} All
evaluation surface forms are unseen in training. Values are means over 3
seeds. ($\peff{}$ = effective stated reliability after resolving framing.)}
\label{tab:repair}
\begin{tabular}{lccc}
\toprule
 & Base & Control arm & \textbf{Confidence arm} \\
\midrule
Abandon @ $\peff \le 45$ (unseen values) & $\approx$const & 1.00 & \textbf{.92--1.0} \\
Abandon @ $\peff \ge 60$ (unseen values) & $\approx$const & 1.00 & \textbf{.00--.02} \\
Far-position behavior & decays & flat & \textbf{$=$ near} \\
Retain-correct (no note) & .10 & .00 & \textbf{.21--.41} \\
Accept valid correction & .97 & 1.00 & .66--.82 \\
Resist contentless pressure$^{\ast}$ & .40 & .98 & .99 \\
Post-challenge acc.\ (own conf.) & .51 & .30 & \textbf{.60--.66} \\
Held-out FC accuracy & \textbf{.86} & .70--.75 & .61--.73 \\
\bottomrule
\multicolumn{4}{l}{\footnotesize $^{\ast}$Pressure resistance is \emph{not}
specific to the confidence arm: generic training matches it.}
\end{tabular}
\end{table}

Four observations, two favorable and two not. First, the trained model
behaves as a \emph{rule}, not a lookup table: at reliability values,
phrasings, and wrappers never seen in training, revision follows a sharp
threshold at the trained boundary, and the far-position curve matches the
near curve---the persistence every base model lacks (\cref{sec:annotations}).
Second, the control arm is the decisive comparison for \emph{conditioning}:
with the reliability sentences deleted, identical training produces an
unconditional policy (here always-switch; in a pilot with different class
balance, always-retain)---without the epistemic channel there is nothing to
condition on, and training manufactures a reflex in whichever direction the
data tips. Third---and this bounds the claim---pressure resistance is
\emph{not} specific to the epistemic signal: the control also reaches $.98$.
Generic robustness is cheap; \emph{conditional} revision is what only the
reliability channel buys. Fourth, the costs are substantial: held-out
forced-choice accuracy drops 20 points (mean $.66$ vs.\ $.86$ base,
$n{=}80$/seed), so the current intervention demonstrates \emph{trainability},
not a deployable repair; and it teaches \emph{obedience to stated
reliability}, not honest reliability---verbalized overconfidence
($\geq$85\% declared on nearly all items) is untouched. The natural
follow-ups are capability-preserving recipes (higher replay, lower learning
rate) and closing the loop by conditioning on \emph{calibrated} confidence
derived from the model's own correctness estimates, jointly training
confidence honesty and confidence use.

% =====================================================================
\section{Discussion}
% =====================================================================

\textbf{One sentence.} Language-model revision is controlled by cues that
reconsideration is expected---a mentioned alternative, an expression of
doubt, the sheer act of annotation, proximity to the decision---rather than by
information about whether reconsideration is warranted; the one legitimate
signal grows more informative with scale while remaining behaviorally inert,
the pressure-sensitivity is installed at the preference-optimization stage,
and a small targeted training signal (but not prompting, and not generic
robustness training) begins to reverse it.

\textbf{Security (hypothesis).} Our experiments are single-turn factual QA,
not attack demonstrations, so we state this as a testable hypothesis rather
than a result: the decomposition suggests injection defenses that filter for
authoritative-sounding content miss the operative channel, because merely
\emph{mentioning} a target answer was 80--100\% as effective as attributing
it in our protocols, and endorsements were processed as challenges. Whether
this transfers to realistic agent pipelines is an open empirical question.

\textbf{Evaluation.} We propose reporting \kar{} and the
retention/correction pair (with harmonic mean) as standard numbers, the way
calibration papers report ECE. They are cheap, discriminative (the frontier
is empty), and resistant to gaming by either stubbornness or capitulation.

\textbf{Training.} The stage analysis plus the repair suggest a concrete
account: preference data teaches models that expressed doubt should be
accommodated (installed at DPO; consistent with \citealp{sharma2024towards}),
correctness-only RL contains no signal to remove it, and \emph{no} current
stage rewards consuming reliability information---which is why three thousand
examples of exactly that signal produce behavior no amount of scale produced
on its own. Confidence-integration is, on this evidence, a missing training
objective rather than an emergent capability.

% =====================================================================
\section{Limitations}
\label{sec:limitations}
% =====================================================================

Our claims are bounded in five ways. (1)~\emph{Injected claims}: most cells
place answers in the model's mouth; the elicited-own-answer cells replicate
the key patterns but cover fewer conditions. (2)~\emph{Single-turn
challenges} on English factual QA; multi-turn negotiation and non-factual
domains are untested. (3)~\emph{Stage ladders are observational}: released
checkpoints differ in data, duration, and formatting beyond the optimizer;
our controlled construction (\cref{sec:controlled-dpo}) addresses the
mechanism but was collecting at submission of this draft. (4)~\emph{The
repair} is demonstrated at one scale (7B), with a capability dip we have not
yet tuned away and no effect on verbalized overconfidence; it establishes
trainability, not a deployment recipe. (5)~\emph{Scale ceiling}: 32B
(quantized) is our largest model; frontier-scale behavior may differ, though
the 7B$\to$14B$\to$32B trend gives no indication that the phenomenon
resolves.

% =====================================================================
\section{Reproducibility and Ethics}
% =====================================================================

All experiments run on open-weight models with public datasets. The
repository contains the full harness, claim banks, per-trial outputs'
aggregates, analysis code, LoRA adapters, and both pre-registration files
with MD5 hashes and timestamps (including the failed prediction and its
reversal). Total compute: $\approx$120 GPU-hours on shared A5000/A6000
nodes. The capitulation results could inform adversarial prompting; we judge
the defensive value of characterizing the (already easily discoverable)
phenomenon to outweigh this, and the repair section is precisely a mitigation
path.

\bibliographystyle{iclr2027_conference}
% ================= REFERENCES =================
\begin{thebibliography}{40}

\bibitem[Sharma et~al.(2024)]{sharma2024towards}
Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, et~al.
\newblock Towards understanding sycophancy in language models.
\newblock In \emph{International Conference on Learning Representations
  (ICLR)}, 2024.

\bibitem[Perez et~al.(2023)]{perez2023discovering}
Ethan Perez, Sam Ringer, Kamil\.e Luko\v{s}i\={u}t\.e, et~al.
\newblock Discovering language model behaviors with model-written evaluations.
\newblock In \emph{Findings of ACL}, 2023.

\bibitem[Kadavath et~al.(2022)]{kadavath2022language}
Saurav Kadavath, Tom Conerly, Amanda Askell, et~al.
\newblock Language models (mostly) know what they know.
\newblock \emph{arXiv preprint arXiv:2207.05221}, 2022.

\bibitem[Lin et~al.(2022{\natexlab{a}})]{lin2022teaching}
Stephanie Lin, Jacob Hilton, and Owain Evans.
\newblock Teaching models to express their uncertainty in words.
\newblock \emph{Transactions on Machine Learning Research (TMLR)}, 2022.

\bibitem[Tian et~al.(2023)]{tian2023just}
Katherine Tian, Eric Mitchell, Allan Zhou, et~al.
\newblock Just ask for calibration: Strategies for eliciting calibrated
  confidence scores from language models fine-tuned with human feedback.
\newblock In \emph{Proceedings of EMNLP}, 2023.

\bibitem[Xiong et~al.(2024)]{xiong2024can}
Miao Xiong, Zhiyuan Hu, Xinyang Lu, et~al.
\newblock Can {LLMs} express their uncertainty? {A}n empirical evaluation of
  confidence elicitation in {LLMs}.
\newblock In \emph{ICLR}, 2024.

\bibitem[Huang et~al.(2024)]{huang2024large}
Jie Huang, Xinyun Chen, Swaroop Mishra, et~al.
\newblock Large language models cannot self-correct reasoning yet.
\newblock In \emph{ICLR}, 2024.

\bibitem[Rafailov et~al.(2023)]{rafailov2023direct}
Rafael Rafailov, Archit Sharma, Eric Mitchell, et~al.
\newblock Direct preference optimization: Your language model is secretly a
  reward model.
\newblock In \emph{NeurIPS}, 2023.

\bibitem[Ouyang et~al.(2022)]{ouyang2022training}
Long Ouyang, Jeffrey Wu, Xu Jiang, et~al.
\newblock Training language models to follow instructions with human feedback.
\newblock In \emph{NeurIPS}, 2022.

\bibitem[Cui et~al.(2024)]{cui2024ultrafeedback}
Ganqu Cui, Lifan Yuan, Ning Ding, et~al.
\newblock {UltraFeedback}: Boosting language models with scaled {AI} feedback.
\newblock In \emph{ICML}, 2024.

\bibitem[Tunstall et~al.(2024)]{tunstall2024zephyr}
Lewis Tunstall, Edward Beeching, Nathan Lambert, et~al.
\newblock Zephyr: Direct distillation of {LM} alignment.
\newblock In \emph{Conference on Language Modeling (COLM)}, 2024.

\bibitem[Lambert et~al.(2024)]{lambert2024tulu}
Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et~al.
\newblock T\"ulu 3: Pushing frontiers in open language model post-training.
\newblock \emph{arXiv preprint arXiv:2411.15124}, 2024.

\bibitem[OLMo Team(2025)]{olmo2}
Team OLMo, Pete Walsh, Luca Soldaini, et~al.
\newblock 2 {OLMo} 2 furious.
\newblock \emph{arXiv preprint arXiv:2501.00656}, 2025.

\bibitem[Hendrycks et~al.(2021)]{hendrycks2021measuring}
Dan Hendrycks, Collin Burns, Steven Basart, et~al.
\newblock Measuring massive multitask language understanding.
\newblock In \emph{ICLR}, 2021.

\bibitem[Wang et~al.(2024)]{wang2024mmlupro}
Yubo Wang, Xueguang Ma, Ge Zhang, et~al.
\newblock {MMLU-Pro}: A more robust and challenging multi-task language
  understanding benchmark.
\newblock In \emph{NeurIPS Datasets and Benchmarks}, 2024.

\bibitem[Lin et~al.(2022{\natexlab{b}})]{lin2022truthfulqa}
Stephanie Lin, Jacob Hilton, and Owain Evans.
\newblock {TruthfulQA}: Measuring how models mimic human falsehoods.
\newblock In \emph{Proceedings of ACL}, pp.\ 3214--3252, 2022.

\bibitem[Welbl et~al.(2017)]{welbl2017crowdsourcing}
Johannes Welbl, Nelson~F. Liu, and Matt Gardner.
\newblock Crowdsourcing multiple choice science questions.
\newblock In \emph{Proceedings of the 3rd Workshop on Noisy User-generated
  Text (W-NUT)}, pp.\ 94--106, 2017.

\bibitem[Hu et~al.(2022)]{hu2022lora}
Edward~J. Hu, Yelong Shen, Phillip Wallis, et~al.
\newblock {LoRA}: Low-rank adaptation of large language models.
\newblock In \emph{ICLR}, 2022.

\bibitem[Wei et~al.(2022)]{wei2022flan}
Jason Wei, Maarten Bosma, Vincent~Y. Zhao, et~al.
\newblock Finetuned language models are zero-shot learners.
\newblock In \emph{ICLR}, 2022.

\bibitem[Longpre et~al.(2023)]{longpre2023flan}
Shayne Longpre, Le Hou, Tu Vu, et~al.
\newblock The {Flan} collection: Designing data and methods for effective
  instruction tuning.
\newblock In \emph{ICML}, 2023.

\bibitem[Kwon et~al.(2023)]{kwon2023vllm}
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et~al.
\newblock Efficient memory management for large language model serving with
  {PagedAttention}.
\newblock In \emph{SOSP}, pp.\ 611--626, 2023.

\bibitem[Zhang et~al.(2024)]{zhang2024snowball}
Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah~A. Smith.
\newblock How language model hallucinations can snowball.
\newblock In \emph{ICML}, 2024.

\bibitem[Li et~al.(2023)]{li2023iti}
Kenneth Li, Oam Patel, Fernanda Vi\'egas, Hanspeter Pfister, and Martin
  Wattenberg.
\newblock Inference-time intervention: Eliciting truthful answers from a
  language model.
\newblock In \emph{NeurIPS}, 2023.

\bibitem[Liang \& Zeger(1986)]{liang1986longitudinal}
Kung-Yee Liang and Scott~L. Zeger.
\newblock Longitudinal data analysis using generalized linear models.
\newblock \emph{Biometrika}, 73(1):13--22, 1986.

\bibitem[DerSimonian \& Laird(1986)]{dersimonian1986meta}
Rebecca DerSimonian and Nan Laird.
\newblock Meta-analysis in clinical trials.
\newblock \emph{Controlled Clinical Trials}, 7(3):177--188, 1986.

\bibitem[Qwen Team(2025)]{qwen25}
An Yang, Baosong Yang, Beichen Zhang, et~al.
\newblock Qwen2.5 technical report.
\newblock \emph{arXiv preprint arXiv:2412.15115}, 2025.

\bibitem[Anonymous(2026)]{repetition2026}
[Repetition-confound study; verified arXiv:2607.05545].
\newblock On conformity protocols and answer repetition in {LLM} evaluation.
\newblock \emph{arXiv preprint arXiv:2607.05545}, 2026.

\bibitem[Anonymous(2025)]{retraction2025}
[Verified arXiv:2505.16170].
\newblock When do {LLMs} admit their mistakes? {U}nderstanding the role of
  model belief in retraction.
\newblock \emph{arXiv preprint arXiv:2505.16170}, 2025.

\end{thebibliography}

% =====================================================================
\appendix
\section{Appendix A: Exact prompts and templates}
All system prompts, claim/challenge templates (train and held-out), and
filler-turn text, verbatim. [See repository \texttt{harness/}.]

\section{Appendix B: Pre-registration receipts}
Both prediction files verbatim with MD5 hashes and commit timestamps: the
stage-ladder predictions (confirmed: DPO-stage damage rise in 3/3 lineages;
RLVR partial recovery 2/2) and the matched-position control predictions
(P2 failed and reversed; P1 split; P3, P4 confirmed).

\section{Appendix C: Null and negative results}
(i) Activation steering along the declared-confidence direction fails the
selectivity criterion: probe directions are perfectly decodable (a lexical
confound) but steering produces sign-inconsistent effects indistinguishable
from random-direction perturbation at matched norm---unlike belief-direction
steering \citep{retraction2025}, the declared-confidence representation is
not causally wired to revision through any single direction we found.
(ii) Prompted rescue fails (\cref{sec:scissors}). (iii) UltraFeedback
contains only 0.4\% accommodation-preferring pairs by our detector,
motivating the constructive design of \cref{sec:controlled-dpo}.

\section{Appendix D: Full per-checkpoint tables and sensitivity analyses}
\label{sec:appendixD}
All 19-checkpoint tables with cluster-bootstrap CIs; GEE coefficients per
checkpoint; heterogeneity statistics ($\tau^2$, $Q$), Hartung--Knapp
intervals, and leave-one-family-out sensitivity for the family-level
meta-analyses.

\end{document}
