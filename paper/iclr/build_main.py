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
\caption{\textbf{The three cells at the frontier.} Same bank, same challenges as \cref{tab:cells}; GPT-6 Astra in one deterministic pass through the batch API at low reasoning effort. Left: a correct injected answer is abandoned under a counter-argument. Middle: a wrong answer is left when a wrong alternative is named; the hatched part of the frontier bar is switches to the \emph{true} answer, which neither turn named. Right: a true correction is accepted. Error bars are 95\% question-cluster bootstrap intervals.}
\label{fig:frontier}
\end{figure}
'''
agents = strip("agents_consequence.tex").replace("Appendix~X", "\\cref{app:lean}").replace("../figures/fig6_contagion.pdf", "figs/frontier_contagion.pdf")
agents = agents.replace("Section~5 says otherwise", "\\cref{sec:control} says otherwise").replace("\\subsection{The gap propagates between agents}", "\\subsection{The gap propagates between agents}\n\\label{sec:agents}")
agents += r'''
The same protocol on GPT-6 Astra (300 questions, 14{,}259 trials; red in \cref{fig:contagion}) shows the sender invariance exactly: with a correct answer injected, Astra folds to a peer naming the alternative at $0.12$ / $0.11$ / $0.12$ / $0.12$ whether the peer is a 1.5B, 7B or 14B open model or Astra itself, and no more often than with no message at all ($0.15$ [$0.11,0.19$]); with its own reasoned answer in context it folds at $0.00$--$0.03$. The chain keeps its form with different constants: a bare wrong seed is rejected by the first agent $0.81$ of the time, but once an agent is wrong \emph{with its reasoning} the next is wrong with $\pww=0.96$ [$0.93,0.98$] against $\pcw=0.01$ from a correct predecessor, so $\lambda=0.95$, $\piinf=0.21$, both chains sit at $0.19$ (wrong seed) and $0.10\to0.14$ (right seed) for eight hops, and the closed form from the measured first hop places 8/8 hops inside the interval for both. A reasoned wrong answer is as contagious at the frontier as at 7B; what changed is how often one is produced.
'''
# ---- edits to Anish's body
body = body.replace("\\end{abstract}", "The same structure holds at the frontier: GPT-6 Astra abandons a correct answer a fifth as often as a 7B model and uses its own confidence, and the source of a challenge, exactly as little; a wrong answer that one agent reasons its way to is passed down a chain of agents at the frontier as reliably as at 7B.\n\\end{abstract}")
body = body.replace("\\section{Conclusion}\n\\label{sec:conclusion}\n\n", "\\section{Conclusion}\n\\label{sec:conclusion}\n\n")
ladder = strip("della_pending.tex")
body = body.replace("\\section{Two Consequences}", ladder.rstrip() + "\n\n\\section{Three Consequences}")
body = body.replace("\\section{Limitations}", agents.rstrip() + "\n\n\\section{Limitations}")
body = body.replace("\\section{Training Installs the Missing Link}", frontier.rstrip() + "\n\n\\section{Training Installs the Missing Link}")
body = body.replace("The identification experiments inject claims rather than eliciting them; elicited-answer conditions in the survey experiments reproduce the main patterns, but the full three-cell design under elicitation is future work.",
 "The identification experiments on open models inject claims rather than eliciting them; the elicited three-cell design is run on the two frontier models (\\cref{sec:frontier}) and reproduces the injected pattern, and remains to be run on the open models.")
body = body.replace("The largest unquantized model is 14B.", "The largest open model is 14B; the two frontier models are measured in a single deterministic pass at low reasoning effort, and their responses are graded with an equivalence judge whose agreement with exact-match grading is reported in \\cref{app:frontier}.")
body = body.replace("[width=\\textwidth]{figs/fig_gap", "[width=0.78\\linewidth]{figs/fig_gap").replace("[width=\\columnwidth]{figs/fig_forest", "[width=0.6\\linewidth]{figs/fig_forest").replace("[width=\\columnwidth]{figs/fig_repair", "[width=0.55\\linewidth]{figs/fig_repair").replace("[width=\\textwidth]{figs/fig_cells", "[width=0.9\\linewidth]{figs/fig_cells").replace("[width=\\columnwidth]{figs/fig_cells", "[width=0.9\\linewidth]{figs/fig_cells")
body = body.replace("\\columnwidth", "\\linewidth").replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
appx = appx.replace("\\columnwidth", "\\linewidth")
# ---- frontier appendix
tables = open(os.path.join(ROOT, "paper/contrib/frontier_table.tex")).read()
tables = "\n".join(l for l in tables.splitlines() if not l.lstrip().startswith("%"))
appx_front = r'''
\section{Frontier Replication: Protocol, Grading, and Tables}
\label{app:frontier}
\textbf{Protocol.} GPT-6 Astra is run on the identical question banks, prompts, and challenge texts as the open models: the v17 decomposition (450 questions, all twelve cells, 10{,}800 trials per model), the three-cell identification bank (500 questions, injected and elicited claims), and the contagion protocol of \cref{sec:agents} (300 questions, the same Qwen2.5 1.5B/7B/14B peer messages, plus the model itself as a fourth sender). Requests go through the OpenAI Batch API at reasoning effort \emph{low}, one deterministic pass (seed 0), with the model snapshot recorded per trial; the full set is 36{,}339 requests plus 15{,}573 judge requests, about \$145 at batch prices.

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
\newcommand{\pending}[1]{\textcolor{blue!65!black}{[\,#1\,]}}
\newtheorem{proposition}{Proposition}
\newcommand{\pww}{p_{\mathrm{ww}}}
\newcommand{\pcw}{p_{\mathrm{cw}}}
\newcommand{\piinf}{\pi_{\infty}}

\title{Confidence Without Control: Language Models Know\\When They Might Be Wrong and Revise Anyway}

\author{Anonymous authors\\Paper under double-blind review}

\begin{document}
\maketitle

'''
tail = r'''
\bibliography{refs}
\bibliographystyle{iclr2027_conference}

\newpage
''' + appx + "\n\\end{document}\n"
out = head + body + tail
open(os.path.join(HERE, "main.tex"), "w").write(out)
print("wrote paper/iclr/main.tex", len(out.split()), "words")
