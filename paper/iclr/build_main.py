#!/usr/bin/env python3
"""Assemble paper/iclr/main.tex: Anish's paper/latex/main.tex body on the ICLR 2027 template + V's additions (frontier subsection, agent-chain
consequence, limitations, frontier appendix). Re-run after editing the sources; never hand-edit main.tex."""
import re, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
src = open(os.path.join(ROOT, "paper/latex/main.tex")).read()
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
agents = strip("agents_consequence.tex").replace("Appendix~X", "\\cref{app:lean}").replace("../figures/fig6_contagion.pdf", "figs/frontier_contagion.pdf")
agents = agents.replace("Section~5 says otherwise", "\\cref{sec:control} says otherwise").replace("\\subsection{The gap propagates between agents}", "\\subsection{The gap propagates between agents}\n\\label{sec:agents}")
agents += r'''
The same protocol on GPT-6 Astra (300 questions, 14{,}259 trials; red in \cref{fig:contagion}) shows the sender invariance exactly: with a correct answer injected, Astra folds to a peer naming the alternative at $0.12$ / $0.11$ / $0.12$ / $0.12$ whether the peer is a 1.5B, 7B or 14B open model or Astra itself, and no more often than with no message at all ($0.15$ [$0.11,0.19$]); with its own reasoned answer in context it folds at $0.00$--$0.03$. The chain keeps its form with different constants: a bare wrong seed is rejected by the first agent $0.81$ of the time, but once an agent is wrong \emph{with its reasoning} the next is wrong with $\pww=0.96$ [$0.93,0.98$] against $\pcw=0.01$ from a correct predecessor, so $\lambda=0.95$, $\piinf=0.21$ and both chains sit at $0.19$ (wrong seed) and $0.10\to0.14$ (right seed) for eight hops. A reasoned wrong answer is as contagious at the frontier as at 7B; what changed is how often one is produced.
'''
# ---- edits to Anish's body
import re as _re
# reproducibility: no compute hours, no repository specifics; becomes the ICLR "Reproducibility statement" (does not count toward the limit)
body = _re.sub(r"\\section\*\{Reproducibility\}.*?(?=\Z)", "", body, flags=_re.S)
# move two control subsections to the appendix, leaving two-sentence summaries (page limit)
def cut(body, start_marker, end_marker):
    i = body.index(start_marker); j = body.index(end_marker, i); return body[:i] + body[j:], body[i:j]
body, nulls = cut(body, "\\subsection{The gap is not an access failure}", "\\subsection{A pre-registered reversal}")
body, reversal = cut(body, "\\subsection{A pre-registered reversal}", "\\subsection{Where the policy plausibly comes from}")
summary = r'''\subsection{Two controls: access and position}
\label{sec:nulls}\label{sec:reversal}
Two controls, reported in full in \cref{app:controls}, rule out the easy explanations. \emph{Access}: writing the model's reliability into the context as a token it emits itself, at values from 20 to 95 percent, moves abandonment by at most $0.069$ with no dose response (Qwen2.5-7B abandons at $0.94$ when its stated reliability is 20 percent and at $0.98$ when it is 95), and instructing the model to weigh its stated confidence moves the confidence effect by at most $0.02$; the gap is not inaccessibility. \emph{Position}: an apparent discounting of the model's own confidence relative to a user's, replicated in all nineteen checkpoints, is a position artifact; a pre-registered matched-position control eliminates it and mildly reverses it (family-level speaker-by-confidence interaction $-0.38$, Hartung--Knapp 95\% CI $[-0.54,-0.22]$, every leave-one-family-out interval excluding zero). Recency, not speaker identity, is the operative variable.

'''
body = body.replace("\\subsection{Where the policy plausibly comes from}", summary + "\\subsection{Where the policy plausibly comes from}")
controls_app = "\n\\section{Two Controls in Full}\n\\label{app:controls}\n" + nulls.replace("\\subsection{The gap is not an access failure}\n\\label{sec:nulls}\n", "\\paragraph{The gap is not an access failure.}") + reversal.replace("\\subsection{A pre-registered reversal}\n\\label{sec:reversal}\n", "\\paragraph{A pre-registered reversal.}")
appx = appx + controls_app
# Table 1 (uncertainty-signal AUROCs) -> appendix
i = body.index("\\begin{table}"); j = body.index("\\end{table}", i) + len("\\end{table}")
tab1 = body[i:j]; assert "AUROC" in tab1 and "tab:signals" in tab1, tab1[:200]
body = body[:i] + body[j:]
appx = appx + "\n\\section{Uncertainty Signals}\n\\label{app:signals}\n" + tab1
# page limit: move fig_forest and fig_repair to the appendix; 5.3 keeps its first paragraph; 5.4 becomes one sentence
def cutfig(body, filename):
    i = body.rfind("\\begin{figure", 0, body.index(filename)); j = body.index("\\end{figure", i); j = body.index("}", j) + 1
    return body[:i] + body[j:], body[i:j]
body, fig_forest = cutfig(body, "figs/fig_forest.pdf")
body, fig_repair = cutfig(body, "figs/fig_repair.pdf")
body, quant = cut(body, "\\subsection{The policy, quantified}", "\\subsection{Progress is one-sided}")
quant_paras = quant.split("\n\n")
quant_main = "\n\n".join(quant_paras[:2]).rstrip() + " The raw confidence splits are larger than the regression coefficient and are difficulty in disguise; \\cref{app:quant} gives the analysis and the coefficient plot.\n\n"
body = body.replace("\\subsection{Progress is one-sided}", quant_main + "\\subsection{Progress is one-sided}")
body, progress = cut(body, "\\subsection{Progress is one-sided}", "\\subsection{Two controls: access and position}")
body = body.replace("\\subsection{Two controls: access and position}", "\\paragraph{Progress is one-sided.}\\label{sec:qwen3}Qwen3-8B, the newest checkpoint, keeps true answers under a sourced counter at $0.56$ against $0.80$--$0.94$ for every older model, yet its F$\\to$F rate is still $0.90$: it learned to weight the truth of its own claim and nothing about evaluating the alternative (\\cref{app:quant}).\n\n\\subsection{Two controls: access and position}")
appx = appx + "\n\\section{The Policy, Quantified: Full Analysis}\n\\label{app:quant}\n" + "\n\n".join(quant_paras[2:]).replace("\\label{sec:regression}", "") + "\n" + fig_forest.replace("[width=0.6\\linewidth]", "[width=0.7\\linewidth]") + "\n\\paragraph{Progress is one-sided, in full.}" + progress.replace("\\subsection{Progress is one-sided}\n\\label{sec:qwen3}\n", "") + "\n" + fig_repair.replace("[width=0.55\\linewidth]", "[width=0.6\\linewidth]")
body = body.replace("[width=0.9\\linewidth]{figs/fig_cells", "[width=\\linewidth]{figs/fig_cells").replace("[width=\\textwidth]{figs/fig_cells", "[width=\\linewidth]{figs/fig_cells")
# 5.7 origin hypothesis -> compressed (the ladder section tests it)
body, origin = cut(body, "\\subsection{Where the policy plausibly comes from}", "\\section{Training Installs the Missing Link}")
origin_short = r'''\paragraph{Where the policy plausibly comes from.}\label{sec:origin}Our results fit a corpus-statistics account, stated as a hypothesis: in dialogue data, whether an interlocutor's claim is true leaves traces (people object to false claims) while a speaker's private confidence is invisible, so the corpus teaches an accurate confidence estimator and a flat confidence-to-behavior mapping. The account predicts the surfacing null and that training installs the mapping (\cref{sec:repair}); \cref{sec:ladder} tests it causally and \cref{app:origin} states it in full.

'''
body = body.replace("\\section{Training Installs the Missing Link}", origin_short + "\\section{Training Installs the Missing Link}")
appx = appx + "\n\\section{The Corpus-Statistics Account in Full}\n\\label{app:origin}\n" + origin.replace("\\subsection{Where the policy plausibly comes from}\n\\label{sec:origin}\n", "")
# limitations: tighter
i = body.index("\\section{Limitations}"); j = body.index("\\section{Conclusion}")
body = body[:i] + r'''\section{Limitations}
\label{sec:limits}
The open-model identification experiments inject claims; the elicited design is run on the frontier model (\cref{sec:frontier}) and remains to be run on the open models. All experiments are single-challenge, English and factual; the largest open model is 14B, and the frontier model is measured in one deterministic pass at low reasoning effort with grading agreement reported in \cref{app:frontier}. The released-checkpoint comparisons are observational until \cref{sec:ladder} completes, and the repair is demonstrated at one scale on one family, with rule obedience rather than genuine confidence use.

''' + body[j:]
# conclusion: tighter
i = body.index("\\section{Conclusion}"); j = body.index("\\end{abstract}") if False else len(body)
body = body[:i] + r'''\section{Conclusion}
\label{sec:conclusion}
Language models increasingly know when they might be wrong, and none of that knowledge governs what they do under challenge: a mentioned source moves a model to answers it believes less than its own at 99 percent rates and its stated confidence is worth $0.007$ in switch probability once content is controlled, and the structure is the same at the frontier. What helps is training: a few thousand demonstrations connect stated confidence to revision at no marginal capability cost, and deleting the confidence sentences from the same data severs the connection. The gap between knowing and acting is not a capacity limit. It is a missing entry in the learned policy, and it can be written in.

'''
# two consequences -> one-sentence summaries in the main text, full paragraphs in the appendix
i = body.index("\\textbf{Survival beats stated confidence.}"); j = body.index("\\section{Limitations}")
conseq_full = body[i:j]
conseq_short = r'''\textbf{Two deployment consequences}, survival under a standardized battery of challenges as a confidence score that beats the model's stated confidence in 16 of 19 checkpoints, and the collapse of in-place reconsideration after a false counter (accuracy $0.00$--$0.08$ on questions first answered correctly, restored by re-asking in a fresh context), are given in full in \cref{app:conseq}.

'''
body = body[:i] + conseq_short + body[j:]
appx = appx + "\n\\section{Two Deployment Consequences in Full}\n\\label{app:conseq}\n" + conseq_full
# related work and setup: compact in the main text, full versions in the appendix (page limit)
body, related_full = cut(body, "\\section{Related Work}", "\\section{Experimental Setup}")
body, setup_full = cut(body, "\\section{Experimental Setup}", "\\begin{figure*}")  # fig_gap float sits between Setup and Section 4
related_short = r'''\section{Related Work}
\label{sec:related}
Sycophancy is documented and traced to preference data \citep{sharma2024sycophancy, perez2023discovering}; repetition of an alternative accounts for much apparent conformity without any speaker \citep{hu2026conformity}, and stated answers are unstable under self-challenge \citep{saadat2026certainty}. Calibration work shows that verbalized confidence and self-evaluation predict correctness \citep{lin2022teaching, tian2023just, xiong2024can, kadavath2022language}; that literature measures the monitor, we measure whether the controller consumes it. \citet{yang2025retraction} find that a hidden-state belief causally drives retraction, the contrast to our result that every \emph{explicit} channel is inert. Intrinsic self-correction is unreliable \citep{huang2024large}; contentless doubt alone flips $0.53$ to $0.91$ of correct answers. Staged releases \citep{olmo2025, lambert2024tulu, tunstall2024zephyr} place changes at training stages (\cref{app:stages}); we add causal interventions (full discussion in \cref{app:related}).

'''
setup_short = r'''\section{Experimental Setup}
\label{sec:setup}
Every trial has the same shape: a claim enters as the model's prior assistant turn, optionally with epistemic metadata; a challenge arrives; the model answers in a fixed format; the outcome is scored by normalized string matching into retain, switch-to-alternative or switch-elsewhere (unparsed under 2\%). Decoding is greedy, so uncertainty comes from resampling questions. Three banks: 600 easy SciQ items \citep{welbl2017crowdsourcing}, 900 hard MMLU-Pro and TruthfulQA items \citep{wang2024mmlupro, lin2022truthfulqa} split into analysis and held-out halves, and an identification bank of 600 MMLU-Pro items with the true answer and two distinct distractors. Challenges are matched sentence types (sourced counter, bare counter, source only, pressure, weak suggestion; \cref{app:setup} gives the wording). Nineteen open checkpoints span the Qwen2.5 ladder (0.5B--32B), Llama-3.x, Mistral-7B, Phi-3.5-mini, OLMo-2-7B and the staged OLMo-2, Tulu-3 and Zephyr lineages; the identification experiments use six of them and, in \cref{sec:frontier}, one frontier model. Four uncertainty signals are measured per item in a clean context: verbalized confidence, P(True) \citep{kadavath2022language}, a completion-log-probability belief score, and eight-sample consistency. Within checkpoints we use the cluster bootstrap over questions; the identification analysis is a pooled logistic regression over 52{,}102 trials with model fixed effects and errors clustered by model-item. Two analyses were pre-registered with prediction files hashed before data collection.

'''
assert "\\begin{figure*}" in body and "\\section{The Monitor Works}" in body
body = body.replace("\\begin{figure*}", related_short + setup_short + "\\begin{figure*}", 1)  # right after the Introduction
appx = appx + "\n\\section{Extended Related Work}\n\\label{app:related}\n" + related_full.replace("\\section{Related Work}\n\\label{sec:related}\n", "") + "\n\\section{Setup Details}\n\\label{app:setup}\n" + setup_full.replace("\\section{Experimental Setup}\n\\label{sec:setup}\n", "")
body = body.replace("\\end{abstract}", "The same structure holds at the frontier: two frontier models from different developers abandon a correct answer a fifth and a tenth as often as a 7B model and use their own confidence, and the source of a challenge, exactly as little; a wrong answer that one agent reasons its way to is passed down a chain of agents at the frontier as reliably as at 7B.\n\\end{abstract}")
body = body.replace("\\section{Conclusion}\n\\label{sec:conclusion}\n\n", "\\section{Conclusion}\n\\label{sec:conclusion}\n\n")
ladder = strip("della_pending.tex")
body = body.replace("\\section{Two Consequences}", ladder.rstrip() + "\n\n\\section{Three Consequences}")
body = body.replace("\\section{Limitations}", agents.rstrip() + "\n\n\\section{Limitations}")
body = body.replace("\\section{Training Installs the Missing Link}", frontier.rstrip() + "\n\n\\section{Training Installs the Missing Link}")
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
\begin{proposition}[Chain dynamics]
$\pi_{k+1}=\pcw+(\pww-\pcw)\,\pi_k$, hence $\pi_k=\piinf+(\pi_1-\piinf)\lambda^{k-1}$ with $\lambda=\pww-\pcw$ and $\piinf=\pcw/(\pcw+1-\pww)$ whenever $\pcw+1-\pww>0$. If $0<\pcw$ and $\pww<1$ then $|\lambda|<1$ and $\pi_k\to\piinf$; a repaired agent at position $j$ with retention $q$ resets $\pi_j$ to $1-q$ after which the tail re-converges at rate $\lambda$; and for a mixture of populations $\piinf$ is monotone in each population's $\pcw$.
\end{proposition}
The one-step law is the law of total probability; the closed form follows by induction on $k$. These statements, the decay bound and the firewall reset are formalised and checked in Lean~4 with Mathlib (repository directory \texttt{lean/}, \texttt{lake build} with no \texttt{sorry}); formalisation caught one gap in the informal statement, the hypothesis $\pcw>0$, without which $\lambda=1$ is admissible and nothing decays. \Cref{fig:contagion} plots the closed form from the measured first hop against the measured curve for every model.
'''
appx = appx + appx_front.replace("\\textbf{Tables.}", frontier_fig + "\n\\textbf{Tables.}") + appx_lean
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
open(os.path.join(HERE, "main.tex"), "w").write(out)
print("wrote paper/iclr/main.tex", len(out.split()), "words")
