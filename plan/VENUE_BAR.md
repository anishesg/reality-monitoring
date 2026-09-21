# The bar: recent ICLR / NeurIPS / ICML outcomes and the closest accepted work (2026-09-21)

## Venue statistics (most recent cycle of each)

| Venue | Submissions | Accepted | Rate | Spotlight | Oral |
|---|---|---|---|---|---|
| ICLR 2026 | 19,525 | 5,355 | 27.4% | n/a published (≈5% of submissions historically) | 225 (1.2%) |
| NeurIPS 2025 | 21,575 | 5,290 | 24.5% | 688 (3.2%) | 77 (0.36%) |
| ICML 2026 | 23,918 | 6,352 | 26.6% | 536 (2.2%) | 168 (0.7%) |

Reading: acceptance is roughly one in four; spotlight is roughly the top 10-13% of *accepted* papers; oral is the top 1-4% of
accepted. A spotlight at ICLR 2027 means being clearly above the median accepted alignment paper on one axis that reviewers
can name.

## Closest recent papers and what each carries

| Paper (venue / date) | Frontier models | Causal training intervention | Fix w/o capability loss | Multi-agent | Pre-registered | Mechanism |
|---|---|---|---|---|---|---|
| Sharma et al., Towards Understanding Sycophancy (ICLR 2024) | yes (Claude) | preference-model analysis | no | no | no | partial |
| Stengel-Eskin et al., Persuasion-Balanced Training (NAACL 2025) | no (Llama 8B) | yes (SFT/DPO) | partial | no | no | no |
| Okawa et al., Kindness or Sycophancy (NeurIPS 2025) | no | yes (synthetic games) | n/a | no | no | yes |
| A Few Bad Neurons (NeurIPS 2025) | no | yes (surgical edit) | claimed | no | no | yes |
| Do as We Do, Not as You Think: Conformity of LLMs (ICLR 2025) | no | no | no | yes (debate) | no | partial |
| Why Do Multi-Agent LLM Systems Fail? (NeurIPS 2025 D&B) | yes | no | no | yes (taxonomy) | no | no |
| Decomposing Factual Sycophancy: Size and Instruction Tuning (arXiv 2606.06306) | no | no | no | no | no | partial |
| Not Just RLHF: Alignment Alone Won't Fix Multi-Agent Sycophancy (arXiv 2605.12991) | ? | no | no | yes | no | no |
| From Spark to Fire: Error Cascades in LLM-MAS (arXiv 2603.04474) | ? | no | mitigation | yes (model) | no | model |
| Misinformation Propagation in Benign Multi-Agent Systems (arXiv 2606.16710) | ? | no | no | yes | no | no |
| Not All Flips Are Conformity (arXiv 2607.05545) | ? | no | no | yes (debate) | no | partial |
| ICML 2026 oral: misalignment concepts overinterpreted; insufficient causal interventions | yes | argues for them | n/a | no | n/a | yes |
| **Ours (Cues over Content), after plan/RESEARCH_PLAN.md** | E4 | E2 + FIRM | E3 | E1 | yes (3 files) | yes (corpus-conditional policy) |

Pattern in accepted alignment papers 2025-26: the ones that reach spotlight/oral carry a mechanism *and* a causal
intervention *and* evaluate on models people use. The ICML 2026 oral explicitly criticizes sycophancy work for
"insufficient causal interventions" and "non-robust datasets"; our pre-registration, matched-position control, and the
both-directions training result are the direct answer to that critique and should be framed as such in the intro.

## Where we stand today, honestly

| Axis | Ours today | After E1-E4 | Median accepted | Spotlight-level |
|---|---|---|---|---|
| Scale of evidence | 19 ckpts, 400k trials | + 5 models x 9.6k agent trials, + 4 frontier | 5-10 models | broad + frontier |
| Causal claim about training | observational + synthetic SFT both-directions | + real-recipe DPO (3 seeds x 2 backbones) | one intervention | intervention that transfers |
| Fix | doubt-only, -12 to -21 acc | STAND: mention-covering, ~0 acc cost (if it lands) | "we mitigate" | fix + why it works |
| Frontier relevance | none | 4 API models, same harness | 1-2 API models | frontier + open, same protocol |
| Multi-agent | none | pairwise + chain + firewall, Markov-tested | usually none | new failure mode + cure |
| Rigor | 2 preregs, GEE + meta, HK + LOO done | + judge agreement | bootstrap CIs | pre-registered, negative results kept |

Estimate (not a promise): accept ~35-40% today; ~50% with E1+E2+E4 clean; spotlight ~15-20% conditional on accept if E3
lands with zero capability cost and the frontier contagion result holds; oral ~5%.

## Sources
ICLR 2026 stats: blog.iclr.cc retrospective (2026-03-31), openaccept.org. NeurIPS 2025 stats: neurips.cc blog (2025-09-30),
aip.riken.jp. ICML 2026 stats: @icmlconf decision post; csconfstats. Paper list from openreview/arXiv searches 2026-09-21;
arXiv-only entries have unknown venue status and are listed because they compete for the same reviewers' attention.
