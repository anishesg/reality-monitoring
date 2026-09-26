#!/usr/bin/env python3
"""Assemble paper/iclr/main.tex: Anish's paper/latex/main.tex body on the ICLR 2027 template + V's additions (frontier subsection, agent-chain
consequence, limitations, frontier appendix). Re-run after editing the sources; never hand-edit main.tex."""
import re, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
src = open(os.path.join(ROOT, "paper/latex/main.tex")).read()
NF = os.environ.get("NO_FRONTIER", "0") == "1"   # NO_FRONTIER=1: build the version without the Astra/Fable experiments (writes main_nofrontier.tex)
body = src[src.index("\\begin{abstract}"): src.index("\\bibliography{refs}")]
appx = src[src.index("\\appendix"): src.index("\\end{document}")]
def strip(p):  # drop-in files: remove % comment lines
    return "\n".join(l for l in open(os.path.join(ROOT, "paper/contrib", p)).read().splitlines() if not l.lstrip().startswith("%")).strip() + "\n"
frontier = strip("frontier_ident.tex").replace("\\paragraph{The gap at the frontier.}", "\\subsection{The gap at the frontier}\n\\label{sec:frontier}\n")
frontier = frontier.replace("(Section~X)", "(\\cref{sec:ident})")
frontier_fig = r'''
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figs/frontier_ident.pdf}
\caption{\textbf{The three cells at the frontier.} Same bank, same challenges as \cref{tab:cells}; GPT-6 Astra (red) and Claude Fable 5.1 (orange), each in one deterministic pass at low reasoning effort. Left: a correct injected answer is abandoned under a counter-argument. Middle: a wrong answer is left when a wrong alternative is named; the hatched part of the frontier bar is switches to the \emph{true} answer, which neither turn named. Right: a true correction is accepted. Error bars are 95\% question-cluster bootstrap intervals.}
\label{fig:frontier}
\end{figure}
'''
agents = strip("agents_consequence.tex").replace("Appendix~X", "\\cref{app:lean}").replace("../figures/fig6_contagion.pdf", "figs/frontier_contagion_open.pdf" if NF else "figs/frontier_contagion.pdf")
agents = agents.replace("Section~5 says otherwise", "\\cref{sec:control} says otherwise").replace("\\subsection{The gap propagates between agents}", "\\subsection{The gap propagates between agents}\n\\label{sec:agents}")
agents += "" if NF else r'''
The same protocol on GPT-6 Astra (300 questions, 14{,}259 trials; red in \cref{fig:contagion}) shows the sender invariance exactly: with a correct answer injected, Astra folds to a peer naming the alternative at $0.12$ / $0.11$ / $0.12$ / $0.12$ whether the peer is a 1.5B, 7B or 14B open model or Astra itself, and no more often than with no message at all ($0.15$ [$0.11,0.19$]); with its own reasoned answer in context it folds at $0.00$--$0.03$. The chain keeps its form with different constants: a bare wrong seed is rejected by the first agent $0.81$ of the time, but once an agent is wrong \emph{with its reasoning} the next is wrong with $\pww=0.96$ [$0.93,0.98$] against $\pcw=0.01$ from a correct predecessor, so $\lambda=0.95$, $\piinf=0.21$ and both chains sit at $0.19$ (wrong seed) and $0.10\to0.14$ (right seed) for eight hops. A reasoned wrong answer is as contagious at the frontier as at 7B; what changed is how often one is produced.
'''
# ---- edits to Anish's body
# Synthetic-monitor sweep (co-author's Exp H, 2026-09-26): the trained controller against the Bayesian curve
synth = r"""
\textbf{The controller is fixed; the monitor is now the limit.} To separate what the trained controller can do from what the model's own signals supply, we replace the stated reliability with a synthetic note calibrated to a chosen AUROC against the model's correctness and measure post-challenge accuracy under a sourced counter (\cref{tab:synth}). The untrained model ignores even a near-perfect monitor: at AUROC $0.99$ it stays at $0.24$ where the Bayesian reviser reaches $0.96$. The rule-trained adapter tracks the Bayesian curve, realising 91--99 percent of the achievable gain at every monitor quality, so what training installed is the threshold policy that \cref{prop:price-general}(iv) shows to be optimal, while the shuffled-label control stays flat. Real signals land where the curve says: self-reported confidence has AUROC $0.50$--$0.68$ on this bank, a range in which even the optimal controller gains at most a point, which is why surfacing the model's own confidence adds at most $0.02$ (\cref{sec:nulls}); the best real monitor available on this bank (AUROC $0.80$--$0.87$) lifts the rule adapter from $0.74$ to $0.81$.

\begin{table}[t]
\caption{Post-challenge accuracy under a sourced counter as a function of monitor quality. The monitor is a synthetic reliability note calibrated to the stated AUROC against the model's own correctness; the first row is the Bayesian threshold policy of \cref{prop:price-general}. Efficiency is the fraction of the achievable gain over keep-all realised at AUROC $\geq 0.8$. Own-confidence rows give the range over three seeds.}
\label{tab:synth}
\vskip 0.05in
\centering\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lcccccc|c}
\toprule
monitor AUROC & 0.5 & 0.7 & 0.8 & 0.9 & 0.95 & 0.99 & efficiency \\
\midrule
Bayesian reviser (best possible) & .744 & .754 & .780 & .850 & .885 & .960 & --- \\
rule-trained adapter & .743 & .746 & .781 & .848 & .878 & .959 & .91--.99 \\
own-confidence adapters (3 seeds) & .68--.77 & .71--.78 & .73--.82 & .79--.86 & .83--.88 & .87--.94 & .42--.93 \\
shuffled-label control & .755 & .752 & .753 & .756 & .758 & .758 & flat \\
untrained base & .235 & .237 & .234 & .238 & .235 & .237 & flat \\
\bottomrule
\end{tabular}
\end{table}

"""
_w = body.index("\\textbf{What the trained model is and is not.}"); _we = body.index("\n\n", _w)
what_para = body[_w:_we]; body = body[:_w] + "\\textbf{What the trained model is and is not.} It follows stated reliability as a rule, with a sharp threshold at the trained boundary; \\cref{app:repairdetail} characterises it." + body[_we:]
appx_repair = "\n\\section{The Trained Model in Detail}\n\\label{app:repairdetail}\n" + what_para.replace("\\textbf{What the trained model is and is not.} ", "") + "\n"
body = body.replace("\\section{Two Consequences}", synth + "\\section{Two Consequences}", 1)
body = body.replace("Scale buys the knowledge; only training buys its use.", "Scale buys the knowledge; only training buys its use, and once it does the trained controller realises 91 to 99 percent of the Bayes-optimal gain at every monitor quality, so the monitor becomes the binding limit.", 1)
_c = body.index("Two checks pass."); _ce = body.index("\n\n", _c)
checks = body[_c:_ce]; body = body[:_c] + body[_ce+2:]
appx_checks = "\n\\paragraph{Two checks.}" + checks[len("Two checks pass."):] + "\n"
body = body.replace("It does not. The dissociation has two sides, usually studied by separate communities that do not test each other's assumption.", "It does not.")
_i = body.index("The fix needs a third cell"); _e = body.index("\n\n", _i)
prop_ident = r"""\begin{proposition}[Identification]\label{prop:ident}
Let $\pi(t_{\mathrm{own}},t_{\mathrm{alt}})$ be the switch probability as a function of the truth of the model's claim and of the named alternative; a two-cell protocol observes $\pi(1,0)$ and $\pi(0,1)$ only. (i) A truth-tracker, $\pi=\mathbf{1}[t_{\mathrm{alt}}]$, and a self-doubter, $\pi=\mathbf{1}[\neg t_{\mathrm{own}}]$, are indistinguishable in those two cells and differ by one in the F$\to$F cell $\pi(0,0)$. (ii) For any observed pair $(a,b)\in[0,1]^2$ and any $\delta\in[b-1,b]$ there is a policy with values in $[0,1]$ realising $(a,b)$ whose alternative-truth effect $\pi(0,1)-\pi(0,0)$ equals $\delta$. (iii) The three cells determine both the alternative-truth effect and the own-falsity effect $\pi(0,0)-\pi(1,0)$.
\end{proposition}
Proofs are in \cref{app:theory}; (iii) makes the F$\to$F rate a measurement of mechanism."""
body = body[:_e] + "\n\n" + prop_ident + body[_e:]
old_price = "\\paragraph{What ignoring the monitor costs.} The gap has a price that follows from the decision alone. Consider a content-free challenge on a two-candidate item, a reviser whose reliability $r$ is calibrated ($\\Pr[\\text{correct}\\mid r]=r$) with pre-challenge accuracy $A=\\mathbb{E}[r]$, and a policy that switches with probability $\\pi$ independently of $r$, which is what the next section measures. Holding yields expected correctness $r$ and switching $1-r$, so a Bayesian reviser takes $\\max(r,1-r)$ and the cue-follower $(1-\\pi)r+\\pi(1-r)$; subtracting and taking expectations,\n\\begin{equation}"
assert body.count(old_price) == 1, body.count(old_price)
body = body.replace(old_price, "\\paragraph{What ignoring the monitor costs.} The gap has a price that follows from the decision alone.\n\\begin{proposition}[Price of the cue]\\label{prop:price}\nLet a content-free challenge arrive on a two-candidate item, let the reviser's reliability $r$ be calibrated ($\\Pr[\\text{correct}\\mid r]=r$) with pre-challenge accuracy $A=\\mathbb{E}[r]$, and let it switch with probability $\\pi$ independently of $r$ (the policy the next section measures). The expected post-challenge accuracy of the Bayesian reviser exceeds that of the cue-follower by\n\\begin{equation}")
old_tail = "\\label{eq:price}\n\\end{equation}\nThe first term is the value"
assert body.count(old_tail) == 1, body.count(old_tail)
body = body.replace(old_tail, "\\label{eq:price}\n\\end{equation}\n\\end{proposition}\n\\emph{Proof.} Holding yields expected correctness $r$ and switching $1-r$, so the Bayesian reviser takes $\\max(r,1-r)$ and the cue-follower $(1-\\pi)r+\\pi(1-r)$; subtract and take expectations. \\qed\\ \\Cref{prop:price-general} gives the general form. The first term is the value")
import re as _re
# reproducibility: no compute hours, no repository specifics; becomes the ICLR "Reproducibility statement" (does not count toward the limit)
body = _re.sub(r"\\section\*\{Reproducibility\}.*?(?=\Z)", "", body, flags=_re.S)
# move two control subsections to the appendix, leaving two-sentence summaries (page limit)
def cut(body, start_marker, end_marker):
    i = body.index(start_marker); j = body.index(end_marker, i); return body[:i] + body[j:], body[i:j]
body, nulls = cut(body, "\\subsection{The gap is not an access failure}", "\\subsection{A pre-registered reversal}")
body, reversal = cut(body, "\\subsection{A pre-registered reversal}", "\\subsection{Where the policy plausibly comes from}")
summary = r'''\label{sec:nulls}\label{sec:reversal}\textbf{Two controls} (\cref{app:controls}) rule out the easy explanations: writing the model's reliability into the context moves abandonment by at most $0.069$ with no dose response, and the apparent discounting of the model's own confidence relative to a user's is a position artifact that a pre-registered matched-position control eliminates and mildly reverses ($-0.38$, 95\% CI $[-0.54,-0.22]$).

'''
body = body.replace("\\subsection{Where the policy plausibly comes from}", summary + "\\subsection{Where the policy plausibly comes from}")
controls_app = "\n\\section{Two Controls in Full}\n\\label{app:controls}\n" + nulls.replace("\\subsection{The gap is not an access failure}\n\\label{sec:nulls}\n", "\\paragraph{The gap is not an access failure.}") + reversal.replace("\\subsection{A pre-registered reversal}\n\\label{sec:reversal}\n", "\\paragraph{A pre-registered reversal.}")
appx = appx + controls_app
# Bayesian reviser proposition after the price discussion; drop the closing sentence of that discussion
body = body.replace(" \\Cref{eq:price} is what makes the monitor's AUROC in \\Cref{tab:signals} a quantity with a cost rather than a curiosity: the model forgoes it every time it is challenged.", "")
_k = body.index("\\section{The Controller Ignores It}")
prop_bayes = r"""A Bayesian reviser with evidence of any finite strength still consults $r$ on a band of reliabilities (\cref{prop:bayes}), so no strength of a source makes the monitor irrelevant, and the best policy that ignores the monitor forgoes exactly the value of the monitor for the decision (\cref{prop:voi}).

"""
body = body[:_k] + prop_bayes + body[_k:]
# Table 1 (uncertainty-signal AUROCs) -> appendix (page budget, 2026-09-26)
i = body.index("\\begin{table}"); j = body.index("\\end{table}", i) + len("\\end{table}")
tab1 = body[i:j]; assert "AUROC" in tab1 and "tab:signals" in tab1, tab1[:200]
body = body[:i] + body[j:]
appx = appx + "\n\\section{Uncertainty Signals}\n\\label{app:signals}\n" + tab1
# page limit: move fig_forest and fig_repair to the appendix; 5.3 keeps its first paragraph; 5.4 becomes one sentence
def cutfig(body, filename):
    i = body.rfind("\\begin{figure", 0, body.index(filename)); j = body.index("\\end{figure", i); j = body.index("}", j) + 1
    return body[:i] + body[j:], body[i:j]
body, fig_forest = (body, "") if NF else cutfig(body, "figs/fig_forest.pdf")
body, fig_repair = cutfig(body, "figs/fig_repair.pdf")
body, quant = cut(body, "\\subsection{The policy, quantified}", "\\subsection{Progress is one-sided}")
quant_paras = quant.split("\n\n")
quant_main = "\n\n".join(quant_paras[:2]).rstrip() + " The raw confidence splits are larger than the regression coefficient and are difficulty in disguise; \\cref{app:quant} gives the analysis and the coefficient plot.\n\n"
body = body.replace("\\subsection{Progress is one-sided}", quant_main + "\\subsection{Progress is one-sided}")
body, progress = cut(body, "\\subsection{Progress is one-sided}", "\\label{sec:nulls}\\label{sec:reversal}\\textbf{Two controls}")
pass  # one-sided-progress paragraph lives in the appendix (app:quant) since 2026-09-26
appx = appx + "\n\\section{The Policy, Quantified: Full Analysis}\n\\label{app:quant}\n" + "\n\n".join(quant_paras[2:]).replace("\\label{sec:regression}", "") + "\n" + fig_forest.replace("[width=0.6\\linewidth]", "[width=0.7\\linewidth]") + "\n\\paragraph{Progress is one-sided.}\\label{sec:qwen3}" + progress.replace("\\subsection{Progress is one-sided}\n\\label{sec:qwen3}\n", "") + "\n" + fig_repair.replace("[width=0.55\\linewidth]", "[width=0.6\\linewidth]")
body = body.replace("[width=0.9\\linewidth]{figs/fig_cells", "[width=\\linewidth]{figs/fig_cells").replace("[width=\\textwidth]{figs/fig_cells", "[width=\\linewidth]{figs/fig_cells")
# 5.7 origin hypothesis -> compressed (the ladder section tests it)
body, origin = cut(body, "\\subsection{Where the policy plausibly comes from}", "\\section{Training Installs the Missing Link}")
origin_short = r'''\paragraph{Where the policy plausibly comes from.}\label{sec:origin}Our results fit a corpus-statistics account, stated as a hypothesis: in dialogue data the truth of a claim leaves traces while a speaker's private confidence is invisible, so the corpus teaches an accurate confidence estimator and a flat confidence-to-behavior mapping. It predicts the surfacing null and that training installs the mapping (\cref{sec:repair}); \cref{app:ladder} gives its pre-registered test.

'''
body = body.replace("\\section{Training Installs the Missing Link}", origin_short + "\\section{Training Installs the Missing Link}")
appx = appx + "\n\\section{The Corpus-Statistics Account in Full}\n\\label{app:origin}\n" + origin.replace("\\subsection{Where the policy plausibly comes from}\n\\label{sec:origin}\n", "")
# limitations: tighter
i = body.index("\\section{Limitations}"); j = body.index("\\section{Conclusion}")
body = body[:i] + r'''\section{Limitations}
\label{sec:limits}
The open-model identification experiments inject claims; the elicited design is run on the frontier models (\cref{sec:frontier}), and running it on open models requires an equivalence judge for their free-form answers, which we leave to future work. All experiments are single-challenge, English and factual; the largest open model is 14B, and the frontier model is measured in one deterministic pass at low reasoning effort with grading agreement reported in \cref{app:frontier}. The released-checkpoint comparisons are observational. The pre-registered training ladder that would make them causal (\cref{app:ladder}) is the natural next step; under our compute constraints we have not yet run it, and we plan to include it in a future version. The repair is demonstrated at one scale on one family, with rule obedience rather than genuine confidence use. The theory characterises the revision decision, its cost and its propagation; it does not derive the policy from training, and the account of where the policy comes from remains a hypothesis (\cref{app:origin}).

''' + body[j:]
# conclusion: tighter
i = body.index("\\section{Conclusion}"); j = body.index("\\end{abstract}") if False else len(body)
body = body[:i] + r'''\section{Conclusion}
\label{sec:conclusion}
Language models increasingly know when they might be wrong, and none of that knowledge governs what they do under challenge: a mentioned source moves a model to answers it believes less than its own at 99 percent rates and its stated confidence is worth $0.007$ in switch probability once content is controlled, and the structure is the same at the frontier. What helps is training: a few thousand demonstrations connect stated confidence to revision at no marginal capability cost. The gap between knowing and acting is not a capacity limit but a missing entry in the learned policy, and it can be written in.

'''
# two consequences -> one-sentence summaries in the main text, full paragraphs in the appendix
i = body.index("\\textbf{Survival beats stated confidence.}"); j = body.index("\\section{Limitations}")
conseq_full = body[i:j]
conseq_short = r'''\textbf{Two deployment consequences} (\cref{app:conseq}): survival under a challenge battery beats stated confidence as a confidence score in 16 of 19 checkpoints, and in-place reconsideration after a false counter collapses accuracy where re-asking in a fresh context restores it.

'''
body = body[:i] + conseq_short + body[j:]
appx = appx + "\n\\section{Two Deployment Consequences in Full}\n\\label{app:conseq}\n" + conseq_full
# related work and setup: compact in the main text, full versions in the appendix (page limit)
body, related_full = cut(body, "\\section{Related Work}", "\\section{Experimental Setup}")
body, setup_full = cut(body, "\\section{Experimental Setup}", "\\begin{figure*}")  # fig_gap float sits between Setup and Section 4
related_short = r'''\section{Related Work}
\label{sec:related}
Sycophancy is documented and traced to preference data \citep{sharma2024sycophancy, perez2023discovering}; repetition alone accounts for much apparent conformity \citep{hu2026conformity}, and stated answers are unstable under self-challenge \citep{saadat2026certainty}. Calibration work shows that verbalized confidence and self-evaluation predict correctness \citep{lin2022teaching, tian2023just, xiong2024can, kadavath2022language}; that literature measures the monitor, we measure whether the controller consumes it. \citet{yang2025retraction} find that a hidden-state belief causally drives retraction; every \emph{explicit} channel we test is inert. Intrinsic self-correction is unreliable \citep{huang2024large}. Staged releases \citep{olmo2025, lambert2024tulu, tunstall2024zephyr} place changes at training stages (\cref{app:stages}, \cref{app:related}).

'''
setup_short = r'''\section{Experimental Setup}
\label{sec:setup}
Every trial has the same shape: a claim enters as the model's prior assistant turn, optionally with epistemic metadata; a challenge arrives; the model answers in a fixed format; the outcome is scored by normalized string matching into retain, switch-to-alternative or switch-elsewhere (unparsed under 2\%). Decoding is greedy, so uncertainty comes from resampling questions. Three banks: 600 easy SciQ items \citep{welbl2017crowdsourcing}, 900 hard MMLU-Pro and TruthfulQA items \citep{wang2024mmlupro, lin2022truthfulqa} split into analysis and held-out halves, and an identification bank of 600 MMLU-Pro items with the true answer and two distinct distractors. Challenges are matched sentence types (sourced counter, bare counter, source only, pressure, weak suggestion; \cref{app:setup} gives the wording). Nineteen open checkpoints span the Qwen2.5 ladder (0.5B--32B), Llama-3.x, Mistral-7B, Phi-3.5-mini, OLMo-2-7B and the staged OLMo-2, Tulu-3 and Zephyr lineages; the identification experiments use six of them and, in \cref{sec:frontier}, one frontier model. Four uncertainty signals are measured per item in a clean context: verbalized confidence, P(True) \citep{kadavath2022language}, a completion-log-probability belief score, and eight-sample consistency. Within checkpoints we use the cluster bootstrap over questions; the identification analysis is a pooled logistic regression over 52{,}102 trials with model fixed effects and errors clustered by model-item. Two analyses were pre-registered with prediction files hashed before data collection.

\paragraph{Measures and choices.} Every challenge trial is scored into retain, switch-to-alternative or switch-elsewhere by normalized string match on the model's \texttt{FINAL} line (unparsed under 2\%). We report \emph{abandonment}, the probability of leaving a correct answer, and the switch rate in each cell of \cref{tab:cells}; the \emph{confidence effect}, abandonment at low minus high stated confidence with the challenge held fixed; the \emph{source effect}, abandonment under a sourced minus a bare counter; and $\Delta_b$, the change in F$\to$F switching between items whose clean-context belief favours the alternative versus the claim. Monitor quality is the AUROC of each signal against the model's own correctness on the same items; the regression reports average marginal effects. Decoding is greedy, so uncertainty comes from questions alone (cluster bootstrap); the identification bank carries two distractors per item so that the F$\to$F cell exists; no training item is ever evaluated.

\paragraph{Framework.} Write $\pi(x)$ for the probability that the model abandons its current answer in context $x$, with $x$ decomposed into the cue $u$ (that a challenge occurred, and its form), the monitor $s$ (stated confidence or any of the four signals), and the latent truth values $t_{\mathrm{own}}$ and $t_{\mathrm{alt}}$ of the current answer and of the named alternative. The monitor is good to the extent that $s$ predicts $t_{\mathrm{own}}$, which we report as AUROC; the controller uses it to the extent that $\pi$ depends on $s$ with $u$ held fixed. The confidence-use gap is the conjunction: AUROC well above chance with $\partial\pi/\partial s\approx 0$. \Cref{prop:ident} says which protocols identify the controller, \cref{prop:price} prices the gap, and \cref{prop:chain} propagates it through a population of agents.

'''
assert "\\begin{figure*}" in body and "\\section{The Monitor Works}" in body
body = body.replace("\\begin{figure*}", related_short + setup_short + "\\begin{figure*}", 1)  # right after the Introduction
appx = appx + "\n\\section{Extended Related Work}\n\\label{app:related}\n" + related_full.replace("\\section{Related Work}\n\\label{sec:related}\n", "") + "\n\\section{Setup Details}\n\\label{app:setup}\n" + setup_full.replace("\\section{Experimental Setup}\n\\label{sec:setup}\n", "")
body = body.replace("\\end{abstract}", "Three propositions make the claims exact: only a three-cell protocol identifies the mechanism, a policy that ignores its monitor is strictly suboptimal at a price we compute, and the propagation of a wrong answer through a chain of agents has a closed form that the data follow. \\end{abstract}")
if not NF: body = body.replace("\\end{abstract}", "The same structure holds at the frontier: two frontier models from different developers abandon a correct answer a fifth and a tenth as often as a 7B model and use their own confidence, and the source of a challenge, exactly as little; a wrong answer that one agent reasons its way to is passed down a chain of agents at the frontier as reliably as at 7B.\n\\end{abstract}")
body = body.replace("\\section{Conclusion}\n\\label{sec:conclusion}\n\n", "\\section{Conclusion}\n\\label{sec:conclusion}\n\n")
body = body.replace("\\section{Two Consequences}", "\\section{Three Consequences}")
ladder_app = r'''
\section{Pre-registered Causal Ladder (Design)}
\label{app:ladder}
\Cref{sec:origin} states the corpus-statistics account as a hypothesis and \cref{sec:repair} shows that a targeted rule can be written in; neither shows which training stage installs the cue-following policy, nor whether a reward alone removes it. We pre-registered a training ladder to answer both (predictions and the frozen run list were hashed before any arm was trained; the document is in the supplementary material). Under our compute constraints we have not yet run it; we anticipated it as the natural next step, report the design here so that the predictions are on record, and plan to include the results in a future version. Backbones: OLMo-2-7B-SFT and Tulu-3-8B-SFT, the stage before preference tuning. Arms: A0, the untouched SFT checkpoint; A1, the backbone's own published preference recipe (10k Tulu-3 preference pairs, DPO, released hyperparameters) with nothing about challenges in the data; A2, revision-DPO on post-challenge turns with the correct answer preferred, the same signal through the same optimizer as A3; A3, GRPO with a single reward, post-challenge correctness, so the model is never told to hold or to switch. Evaluation: the challenge battery of \cref{sec:setup} on 450 held-out questions plus MMLU, GSM8K and IFEval, three seeds per arm. Predictions: H1, A1 raises pressure-only abandonment by at least $0.15$ over A0 in both backbones (generic preference tuning installs the cue); H2/H3, A3 retains correct answers at $\geq 0.6$ and accepts valid corrections at $\geq 0.7$ at capability within one point of A0 while A2 does not reach both (a reward on the outcome, not a demonstration, installs the judgment); H5, no arm other than A1 moves capability by more than one point. A pre-registered chain experiment places one A3 agent at position two of a contaminated chain and predicts from \cref{prop:chain} a downstream error reduction of at least $0.40$.
'''
appx = appx + ladder_app
body = body.replace("\\section{Limitations}", agents.rstrip() + "\n\n\\section{Limitations}")
if not NF: body = body.replace("\\section{Training Installs the Missing Link}", frontier.rstrip() + "\n\n\\section{Training Installs the Missing Link}")
body = body.replace("The identification experiments inject claims rather than eliciting them; elicited-answer conditions in the survey experiments reproduce the main patterns, but the full three-cell design under elicitation is future work.",
 "The identification experiments on open models inject claims rather than eliciting them; the elicited three-cell design is run on the two frontier models (\\cref{sec:frontier}) and reproduces the injected pattern, and remains to be run on the open models.")
body = body.replace("The largest unquantized model is 14B.", "The largest open model is 14B; the two frontier models are measured in a single deterministic pass at low reasoning effort, and their responses are graded with an equivalence judge whose agreement with exact-match grading is reported in \\cref{app:frontier}.")
body = body.replace("[width=\\textwidth]{figs/fig_gap", "[width=\\linewidth]{figs/fig_gap").replace("[width=\\columnwidth]{figs/fig_forest", "[width=0.6\\linewidth]{figs/fig_forest").replace("[width=\\columnwidth]{figs/fig_repair", "[width=0.55\\linewidth]{figs/fig_repair").replace("[width=\\textwidth]{figs/fig_cells", "[width=0.9\\linewidth]{figs/fig_cells").replace("[width=\\columnwidth]{figs/fig_cells", "[width=0.9\\linewidth]{figs/fig_cells")
body = body.replace("\\label{tab:cells}\n\\vskip 0.1in\n\\centering\n\\small", "\\label{tab:cells}\n\\vskip 0.05in\n\\centering\n\\small")
body = body.replace("\\columnwidth", "\\linewidth").replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
appx = appx.replace("\\columnwidth", "\\linewidth")
# ---- frontier appendix
tables = open(os.path.join(ROOT, "paper/contrib/frontier_table.tex")).read()
tables = "\n".join(l for l in tables.splitlines() if not l.lstrip().startswith("%"))
appx_front = r'''
\section{Frontier Replication: Protocol, Grading, and Tables}
\label{app:frontier}
\textbf{Protocol.} GPT-6 Astra and Claude Fable 5.1 are run on the identical question banks, prompts, and challenge texts as the open models: the v17 decomposition (450 questions, all twelve cells, 10{,}800 trials per model), the three-cell identification bank (500 questions, injected and elicited claims), and the contagion protocol of \cref{sec:agents} (300 questions, the same Qwen2.5 1.5B/7B/14B peer messages, plus the model itself as a fourth sender). Requests go through the OpenAI Batch API at reasoning effort \emph{low}, one deterministic pass (seed 0), with the model snapshot recorded per trial. Fable 5.1 was run on the identification and decomposition protocols; the agent-chain protocol of \cref{sec:agents} was run on Astra only.

\textbf{Grading.} The open models copy the injected string almost verbatim, so exact matching on the \texttt{FINAL} line grades them reliably. Frontier models restate a retained answer in their own words (``Compounded quarterly'' becomes ``Quarterly compounding is more profitable''), which exact matching scores as a switch to something else. Every frontier trial the string grader marked \emph{switch-other} or \emph{ambiguous} is therefore re-graded by an LLM judge (GPT-6 Astra, sixteen output tokens) asked whether the final answer expresses the claim, the named alternative, the true answer, or none; the string outcome is kept alongside. For Astra 15{,}573 of 36{,}339 trials were re-graded; among the 3{,}686 decomposition rows, 1{,}092 were retained answers in new words and 1{,}050 were switches to the named alternative the string grader had missed. The judge also maps the model's free-form own answer onto the bank candidates for the elicited cells (Astra: 271 of 409 unmatched answers mapped, 130 match no candidate and are excluded; own-answer accuracy 0.72). All string-graded rates are reproducible with \texttt{RM\_GRADING=string}.

\textbf{Tables.} Rates with 95\% question-cluster bootstrap intervals. Open-model rows are string-graded; frontier rows are judge-graded.
\begin{center}\scriptsize
''' + tables + r'''
\end{center}
'''
appx_lean = r'''
\section{Chain Dynamics: Closed Form and Machine-Checked Proofs}
\label{app:lean}
Let agent $k{+}1$ condition only on the reply of agent $k$, with $\pww=\Pr[\text{wrong}_{k+1}\mid\text{wrong}_k]$ and $\pcw=\Pr[\text{wrong}_{k+1}\mid\text{right}_k]$, and let $\pi_k$ be the probability that agent $k$ is wrong.
\begin{proposition}[Chain dynamics]\label{prop:chain}
$\pi_{k+1}=\pcw+(\pww-\pcw)\,\pi_k$, hence $\pi_k=\piinf+(\pi_1-\piinf)\lambda^{k-1}$ with $\lambda=\pww-\pcw$ and $\piinf=\pcw/(\pcw+1-\pww)$ whenever $\pcw+1-\pww>0$. If $0<\pcw$ and $\pww<1$ then $|\lambda|<1$ and $\pi_k\to\piinf$; a repaired agent at position $j$ with retention $q$ resets $\pi_j$ to $1-q$ after which the tail re-converges at rate $\lambda$; and for a mixture of populations $\piinf$ is monotone in each population's $\pcw$.
\end{proposition}
The one-step law is the law of total probability; the closed form follows by induction on $k$. These statements, the decay bound and the firewall reset are formalised and checked in Lean~4 with Mathlib (repository directory \texttt{lean/}, \texttt{lake build} with no \texttt{sorry}); formalisation caught one gap in the informal statement, the hypothesis $\pcw>0$, without which $\lambda=1$ is admissible and nothing decays. \Cref{fig:contagion} plots the closed form from the measured first hop against the measured curve for every model.
'''
appx_theory = r"""
\section{Theory: Proofs and Further Statements}
\label{app:theory}
\textbf{Setting.} A finite set of items $i$ with weights $w_i\geq 0$, $\sum_i w_i=1$, and calibrated reliabilities $r_i\in[0,1]$ ($\Pr[\text{correct}\mid r]=r$). A reviser that switches with probability $\pi$ on an item of reliability $r$ has expected post-challenge correctness $\mathrm{acc}(r,\pi)=(1-\pi)r+\pi(1-r)$; the Bayesian reviser attains $\max(r,1-r)$. Write $x^{+}=\max(x,0)$ and $A=\sum_i w_i r_i$.

\begin{proposition}[Price of the cue, general form]\label{prop:price-general}
(i) For every $r$ and $\pi$, $\max(r,1-r)-\mathrm{acc}(r,\pi)=(1-\pi)(1-2r)^{+}+\pi(2r-1)^{+}$. (ii) For a policy that switches with the same $\pi$ on every item, $\sum_i w_i\max(r_i,1-r_i)-\sum_i w_i\,\mathrm{acc}(r_i,\pi)=\sum_i w_i(1-2r_i)^{+}+\pi(2A-1)$, which is \cref{eq:price}. (iii) If some item with positive weight has $r_i<\tfrac12$ and another has $r_j>\tfrac12$, this quantity is strictly positive for every $\pi\in[0,1]$. (iv) The threshold policy, switch iff $r<\tfrac12$, attains $\max(r,1-r)$ pointwise.
\end{proposition}
\emph{Proof.} (i) is a case split on $r\leq\tfrac12$. (ii) is linearity and $(2r-1)^{+}-(1-2r)^{+}=2r-1$. (iii) bounds each of the two sums in (i) below by its $i$-th or $j$-th term. (iv) is a case split. \qed

\begin{proposition}[Value of the monitor]\label{prop:voi}
Let $V=\sum_i w_i\max(r_i,1-r_i)-\max(A,1-A)$. Then $V\geq 0$, every policy that switches with the same $\pi$ on every item pays at least $V$ relative to the Bayesian reviser, and the best such policy (switch on every item iff $A<\tfrac12$) pays exactly $V$.
\end{proposition}
\emph{Proof.} $\max(r,1-r)=r+(1-2r)^{+}$ pointwise, so $\sum_i w_i\max(r_i,1-r_i)=A+\sum_i w_i(1-2r_i)^{+}$; substitute into \cref{prop:price-general}(ii) and split on $A\gtrless\tfrac12$. Nonnegativity uses $(1-2r)^{+}\geq 1-2r$. \qed

\begin{proposition}[Bayesian reviser]\label{prop:bayes}
Let the challenge carry evidence of likelihood ratio $\ell>0$ for the alternative. A Bayesian reviser with prior reliability $r\in(0,1)$ switches iff $\ell>r/(1-r)$. For every $\ell$ and every $r_{\mathrm{lo}}<r_{\mathrm{hi}}$ whose prior odds bracket $\ell$ it switches at $r_{\mathrm{lo}}$ and holds at $r_{\mathrm{hi}}$, and such a band exists for every pair $r_{\mathrm{lo}}<r_{\mathrm{hi}}$.
\end{proposition}
\emph{Proof.} The posterior odds of the alternative are $\ell(1-r)/r$; the switching rule is that they exceed one. Prior odds $r/(1-r)$ are strictly increasing in $r$, so the band between two of them is non-empty. \qed

\Cref{prop:ident} is proved by exhibiting the two policies in (i), the policy $\pi(1,\cdot)=a$, $\pi(0,1)=b$, $\pi(0,0)=b-\delta$ in (ii), and by definition of the two effects in (iii). Every proposition in the paper, including \cref{prop:chain}, is also formalised and checked in Lean~4 with Mathlib; the sources and build log are in the supplementary material.
"""
appx = appx + appx_repair
appx = appx.replace("\\section{Identification Bank and Robustness}", "\\section{Identification Bank and Robustness}" + appx_checks, 1) if "\\section{Identification Bank and Robustness}" in appx else appx + appx_checks
appx = appx + ("" if NF else appx_front.replace("\\textbf{Tables.}", frontier_fig + "\n\\textbf{Tables.}")) + appx_theory + appx_lean
head = r'''\documentclass{article}
\usepackage{iclr2027_conference,times}
\input{math_commands.tex}
\usepackage{hyperref}
\usepackage{url}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{multirow}
\usepackage[capitalize,noabbrev]{cleveref}
\usepackage{xcolor}
\newcommand{\pending}[1]{{\color{black!45}\itshape #1}}
\newtheorem{proposition}{Proposition}
\newcommand{\pww}{p_{\mathrm{ww}}}
\newcommand{\pcw}{p_{\mathrm{cw}}}
\newcommand{\piinf}{\pi_{\infty}}

\title{Confidence Without Control:\\Language Models Know When They Might Be Wrong\\and Revise Anyway}

\author{Anonymous authors\\Paper under double-blind review}

\begin{document}
\maketitle

'''
tail = r'''
\subsection*{AI use statement}
In this work we used generative AI tools (large language model coding assistants) for writing and refactoring experiment and analysis code, for drafting and editing prose, and for checking derivations, including the preparation of the Lean formalisation in \cref{app:lean}; all AI-assisted code was reviewed and run by the authors and every number in the paper is regenerated by the released analysis scripts from raw model outputs. We did not use generative AI tools to generate research ideas, to produce experimental results, or to write any part of the paper without author review. One of the evaluated systems (GPT-6 Astra) is also used as an equivalence judge for grading frontier responses, as described in \cref{app:frontier}; that use is part of the method, not of the writing. We take responsibility for the final content of this work, including text, claims and artifacts produced with the aid of generative AI.

\subsection*{Ethics statement}
This work evaluates and fine-tunes language models on factual question answering; it involves no human subjects, no personal data and no new data collection beyond model outputs. The behaviours it documents, capitulation to unsupported challenges and the propagation of wrong answers between agents, are failure modes with safety implications; documenting them and the conditions under which they appear is intended to reduce, not enable, harm.

\subsection*{Reproducibility statement}
All open models are open weight and all question banks are public. The supplementary material contains the harness, the three question banks, the analysis scripts that regenerate every number and figure in this paper from raw model outputs, the trained adapters, the pre-registration documents with their hashes and timestamps (including the prediction that failed), and the Lean sources for \cref{app:lean}. Frontier-model runs record the model snapshot, reasoning effort and grading for every trial (\cref{app:frontier}).

\bibliography{refs}
\bibliographystyle{iclr2027_conference}

\newpage
''' + appx + "\n\\end{document}\n"
out = head + body + tail
if NF:
    NFREP = [
        ("the identification experiments use six of them and, in \\cref{sec:frontier}, one frontier model.", "the identification experiments use six of them."),
        (", and the structure is the same at the frontier.", "."),
        (" One of the evaluated systems (GPT-6 Astra) is also used as an equivalence judge for grading frontier responses, as described in \\cref{app:frontier}; that use is part of the method, not of the writing.", ""),
        (" Frontier-model runs record the model snapshot, reasoning effort and grading for every trial (\\cref{app:frontier}).", ""),
        ("the elicited design is run on the frontier models (\\cref{sec:frontier}), and running it on open models requires an equivalence judge for their free-form answers, which we leave to future work.", "the elicited design, with the model's own reasoned answer as the challenged turn, requires an equivalence judge for free-form answers and is left to future work."),
        ("\\cref{app:quant} gives the analysis and the coefficient plot.", "\\cref{app:quant} gives the full analysis and \\cref{fig:forest} plots the coefficients."),
    ]
    for a, b in NFREP:
        assert out.count(a) == 1, ("NF replacement not found", a[:70], out.count(a))
        out = out.replace(a, b)
    out = re.sub(r"[^.]*the frontier model is measured in one deterministic pass[^.]*\.", "", out) if False else out
    m = re.search(r"; the largest open model is 14B, and the frontier model is measured[^.]*\\cref\{app:frontier\}\.", out); assert m, "limitations frontier sentence"
    out = out.replace(m.group(0), "; the largest model is 14B.")
    left = [l for l in out.splitlines() if re.search(r"Astra|Fable|sec:frontier|app:frontier|fig:structure|frontier model", l)]
    assert not left, ("frontier mentions remain", [l[:90] for l in left])
    open(os.path.join(HERE, "main_nofrontier.tex"), "w").write(out); print("wrote paper/iclr/main_nofrontier.tex", len(out.split()), "words")
else:
    open(os.path.join(HERE, "main.tex"), "w").write(out)
    print("wrote paper/iclr/main.tex", len(out.split()), "words")
