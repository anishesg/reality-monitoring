# Merge note for co-author A (2026-09-22)

Two full drafts existed after last night: your `paper/PAPER.md` ("Confidence Without Control", three acts, pooled AMEs, priced
repair) and my `paper/main_v2.tex` ("Cues over Content", formal model, agents, ICLR template). `paper/main_v3.tex` merges
them, taking YOUR spine and title as the base, because the monitoring/control framing with the identification design and the
priced repair is the stronger measurement narrative. What it adds from mine, and where:

- Section 2 "The gap, made precise": Definition (behavioural use as a matched contrast), Proposition 1 (a calibrated signal must
  move a Bayesian reviser), Proposition 2 (the accuracy cost of cue-following, with the Qwen-7B number), Assumption 1 (corpus
  conditional) stated once with six predictions that name their sections, Proposition 3 (chain dynamics). Props 1 and 3 are
  Lean-checked (Appendix B).
- Act four, Section 6 "The gap between agents": sender invariance on four models, eight-hop chains with the fitted closed form,
  the three design rules. One figure, one table.
- Act three: your repair sweep and FIRM/POISON as written; the two queued arms (real-recipe DPO, STAND) in one paragraph; the
  stage ladder demoted to an appendix pointer as you had it.
- Decomposition kept as an appendix (your call), recomputed harmful-only from your audit.
- Everything compiles on the ICLR 2027 template; main text ends on page 7 of 9 allowed, so there is room to restore any
  paragraph you want back from PAPER.md (e.g. the Qwen3 scaling paragraph is in; the self-correction paragraph of related work
  is compressed).

Things I could not fill from your files and marked \pending: P(True) for Qwen3-8B and R1; harmful-direction recompute of the
annotation cells; paraphrased re-asks under sampling; the elicited three-cell design.

Decisions for you: (1) title and framing = yours; agree? (2) is Act four in the main text acceptable (it is the one result
outside the single-model frame and the thing reviewers will retell)? (3) Proposition 2's number uses A=0.87 and pi=0.63 for
Qwen-7B pressure from the identification bank; confirm those are the values you want quoted.
