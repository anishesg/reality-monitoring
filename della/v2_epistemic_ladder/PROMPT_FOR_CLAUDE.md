# Prompt for the operator agent (paste into Claude)

You are operating on Princeton's **Della** HPC cluster to run a measurement study for an
ICLR submission: do LLMs use declared confidence/verification-status when revising prior
claims, across a 0.5B→72B scale ladder. The package `reality-monitoring-v2.zip` contains
everything; your job is careful EXECUTION, not redesign. Do not change prompts, conditions,
or metrics — the design is frozen (pre-registered). Fix only infrastructure issues.

## Ground rules (shared cluster — be a good citizen)
1. GPU work ONLY via sbatch (scripts provided). Never run models on login nodes.
2. Compute nodes have NO internet. All downloads happen in `prepare.sh` on the login node.
3. Don't poll the scheduler: check `squeue -u $USER` at most every 5 minutes; rely on log files.
4. All heavy files live in `/scratch/gpfs/$USER/reality-monitoring`. Never scan shared
   filesystems broadly; never `du`/`find` outside the project dir.
5. NEVER print, echo, or commit the contents of `secrets.env` (HF token). Source it silently.
6. If a job fails: read its log in `logs/`, fix the resource request or skip that model,
   resubmit once. Two failures on the same model = skip it and note why.

## Execution sequence
1. `unzip reality-monitoring-v2.zip && cd pkg && bash prepare.sh` — run inside `tmux`
   (model downloads ~450GB, can take 1–2h). If scratch quota is tight (`checkquota`),
   rerun with `SKIP_BIG=1`.
2. `cd /scratch/gpfs/$USER/reality-monitoring && bash submit_smoke.sh`.
   When done, check `results/smoke/summary.json`:
   - PASS: unparsed_frac < 0.10 → continue.
   - FAIL: read `logs/rm_smoke_*.out`, fix infra (mem/time/partition), resubmit smoke.
3. `bash submit_all.sh` (14 jobs, independent). If the queue rejects gres/partition,
   inspect `snodes` and add the right `--partition`/`--account` (PLI users: `pli-c`).
4. Monitor lazily. When all jobs are COMPLETED (sacct), run:
   `source venv/bin/activate && python analyze_v2.py results`
5. Report back with:
   - `results_summary.md` (verbatim — this is the emergence table),
   - each model's `summary.json` (unparsed_frac, baseline_fc_acc),
   - any models skipped and why,
   - `seff <jobid>` for two representative jobs.

## What the numbers mean (so you can sanity-check, not so you can editorialize)
- `delta_revision` should be strongly positive for competent models (they switch more when
  their claim was false). If it's ~0 for a 7B+ model, something is broken — check parsing.
- `state_use_self low_vs_high` is THE quantity: positive = the model abandons its own
  low-confidence claims more than its high-confidence ones. Prior finding: ~0 at 1.5B,
  ~+0.06 at 7B. The scale curve of this number is the headline.
- `baseline_fc_acc` should be >0.6 for 7B+ models on this bank. If ~0.5 (chance), flag it.
- unparsed_frac >0.15 for any model = flag it, don't silently include.

Report facts; do not adjust the analysis. If something looks scientifically odd, describe
it precisely and stop — the PI decides.
