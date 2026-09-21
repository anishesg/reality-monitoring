# Section 3 (proposed): A model of answer revision

Drafted 2026-09-21 (V). Goal: one formal object that every experiment in the paper measures a piece of, so the sections
read as tests of derived predictions rather than as a list of findings. LaTeX-ready; notation matches `paper/main.tex`.

## 3.1 Setup

An item is a question $x$ with a correct answer $a^\*$ and a distractor $\bar a$. A model $M$ has produced (or been given)
an answer $a \in \{a^\*, \bar a\}$; write $t = \mathbb{1}[a = a^\*]$. A challenge $c$ then enters the context. We decompose
$c$ into three kinds of feature:

- **surface cues** $s(c)$: whether an alternative is named, whether doubt is expressed, whether a source is invoked, the
  position of the cue in the dialogue (recency);
- **evidence** $e(c)$: the likelihood ratio the challenge actually carries for $\bar a$ over $a$ (a reason, a verifiable
  claim, a reliability rating that is *about* the answer);
- **own reliability** $r$: the model's probability, before the challenge, that $a$ is correct, elicited or latent.

The **revision policy** is $\pi_M(a, c) = \Pr[M \text{ switches away from } a \mid a, c]$. Outcomes are scored as
retain / switch to $\bar a$ / other, so $\pi$ is directly estimable per cell.

## 3.2 The normative benchmark

A Bayesian reviser with prior $r$ on $a$ and evidence likelihood ratio $\ell(e)$ for $\bar a$ switches iff
$$\frac{1-r}{r}\,\ell(e) > 1 \quad\Longleftrightarrow\quad \operatorname{logit}(1-r) + \log \ell(e) > 0 .$$
Two consequences define what "using" a signal means in the rest of the paper.

**Definition 1 (behavioral use).** For a signal $z \in \{s, e, r\}$, its use is the contrast
$U_z = \mathbb{E}[\pi \mid z = z_{\text{lo}}] - \mathbb{E}[\pi \mid z = z_{\text{hi}}]$ with the other two held fixed by
design (matched items, matched position, matched wording). $U_z$ is an average partial effect, identified only by the
matched construction; observational contrasts are reported separately as associations.

**Definition 2 (calibration).** $\mathrm{AUROC}(r \to t)$: how well $r$ ranks correct answers above incorrect ones.

**Proposition 1 (a calibrated signal must move a Bayesian reviser).** If $r$ is calibrated and $\ell(e)$ is fixed, then for
any two reliability levels $r_{\text{lo}} < r_{\text{hi}}$ with $\ell(e) \in \big(\tfrac{r_{\text{lo}}}{1-r_{\text{lo}}},
\tfrac{r_{\text{hi}}}{1-r_{\text{hi}}}\big)$ the Bayesian policy switches at $r_{\text{lo}}$ and holds at $r_{\text{hi}}$,
so $U_r = 1$ on that band, and $U_r > 0$ whenever the band has positive measure under the challenge distribution.
*Proof.* Immediate from the switching rule. $\square$

The proposition is trivial, and that is the point: it turns "the model ignores its own confidence" into a measurable
violation. Section 5's finding is $\mathrm{AUROC}(r\to t)$ rising from .49 to .70 with scale while $U_r$ stays within
$\pm .05$ under matched conditions, against a benchmark that predicts $U_r > 0$ for any calibrated $r$. The same
definition makes the surface/evidence contrast precise: Section 4 measures $U_s$ (naming an alternative, expressing doubt)
in the range .4–.9 and $U_e$ (adding a reason or a source) at most .16.

## 3.3 The mechanism: a corpus-conditional revision policy

Pretraining and supervised fine-tuning minimize cross-entropy on human dialogue. In the limit, the model's revision policy
equals the corpus conditional of revision given the *visible* context,
$$\pi_M(a,c) \;\to\; \Pr_{\text{corpus}}\big[\text{revise} \mid \text{visible}(a, c)\big],$$
and the prior reliability of the speaker, $r$, is not part of $\text{visible}(a,c)$: people's private confidence leaves no
token trace, and their *stated* confidence is cheap talk that barely predicts revision in text. Surface cues, by contrast,
are visible and are what human revisions in text actually condition on. This single assumption yields the paper's
predictions:

- **P-surface.** $U_s \gg U_e$: the policy responds to the visible cue, not to the evidence the cue happens to carry. (Section 4.)
- **P-flat.** $U_r \approx 0$ regardless of $\mathrm{AUROC}(r \to t)$, and the gap *widens* with scale because scale sharpens
  the fit to the corpus conditional, not the use of a private variable. (Section 5, the calibration–use gap.)
- **P-surfacing.** Writing $r$ into the context as a token does not create $U_r$, because the corpus conditional on a stated
  reliability is flat; only training that rewrites the conditional can. (Section 5 null; Section 6 repair.)
- **P-recency.** Position in the dialogue is a visible feature; source (who said it) is not once position is matched. The
  apparent self-versus-other asymmetry should vanish under matched position. (Section 5, pre-registered reversal.)
- **P-training.** Preference optimization on human agreement labels raises $\Pr[\text{revise} \mid \text{doubt cue}]$ because
  agreeable continuations are preferred; a reward on *post-challenge correctness* lowers it without touching $U_e$. (Section 6:
  stage ladder, FIRM/POISON, D2/STAND.)
- **P-propagation.** In a pipeline that forwards one agent's answer to the next, the forwarded answer is a named alternative,
  hence a surface cue: the receiver's switch rate should not depend on the sender's competence, and contamination should follow
  the Markov dynamics of Section 7 with $p_{ww} \approx \pi_M(\cdot, \text{mention})$. (Section 7.)

Each prediction was written down before its experiment (pre-registration files in `prereg/`), and two of them failed in
their first form: the self-versus-other asymmetry (reversed under matching, which P-recency explains) and the chain Markov
prediction (hop 1 differs from hops $\geq 2$ because the seed is a bare message; reported as a correction).

## 3.4 Propagation between agents

Let a pipeline forward agent $k$'s reply to agent $k+1$. Under the Markov assumption (agent $k+1$ conditions on the reply it
receives, which is what the architecture enforces), with $p_{ww} = \Pr[\text{wrong}_{k+1} \mid \text{wrong}_k]$ and
$p_{cw} = \Pr[\text{wrong}_{k+1} \mid \text{right}_k]$,
$$\pi_k = \pi_\infty + (\pi_1 - \pi_\infty)\lambda^{k-1}, \qquad \lambda = p_{ww} - p_{cw}, \qquad
\pi_\infty = \frac{p_{cw}}{p_{cw} + 1 - p_{ww}} .$$
$\pi_\infty$ is the long-run error of the pipeline from any seed; $\ln 2 / (-\ln \lambda)$ is the half-life of a seed's
influence. For $m$ independent chains combined by majority vote, $\Pr[\text{majority wrong at depth } k] =
\Pr[\mathrm{Bin}(m, \pi_k) > m/2]$, which increases in $m$ whenever $\pi_k > 1/2$: adding agents does not help once the
per-agent policy is contaminated. A trained agent at position $j$ with $p'_{ww} = q$ resets the chain to $\pi_j \approx q$.
Measured on 7–8B models, $p_{ww} \in [.975, .987]$: a wrong forwarded answer is retained with probability near one per hop,
so $\pi_\infty$ is set almost entirely by the drift rate $p_{cw}$ (.68 for Qwen2.5-7B, .84 for Llama-3.1-8B).

## 3.5 What the model does not claim

It is a description of the learned policy, not of an internal representation; the steering null (Appendix) is consistent
with it but not implied by it. It does not say the model *cannot* access $r$, only that the trained policy does not use it;
Section 6 shows the conditional can be rewritten by demonstrations. And the Markov assumption is tested, not assumed.
