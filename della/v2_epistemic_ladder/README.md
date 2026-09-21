# Reality Monitoring v2 — Della Cluster Package

**Study:** Do language models use *declared epistemic state* (confidence, verification status)
when deciding whether to defend or revise prior claims — and does use of their OWN declared
state differ from their use of a user's or another AI's? Measures a scale-emergence curve
across ~14 models (0.5B→72B, 4 families), 3 domains, 84 conditions/question, ~128k
generations per model.

## Prior result this scales up (Princeton ionic cluster, n=2 models)
- Qwen2.5-1.5B: uses USER's declared confidence (+0.08 [0.05,0.10]) but is BLIND to its own (−0.006 [−0.02,+0.01]).
- Qwen2.5-7B: own-confidence use emerges but is weak (+0.06) vs truth-driven revision (+0.81).
- Self-claims are cheaper to abandon than user claims everywhere (origin effect +0.12 to +0.30).

## Quickstart (in order)
```bash
# 1. copy zip to della and unzip
unzip reality-monitoring-v2.zip && cd pkg
# 2. everything network-y runs on the LOGIN node (compute nodes have no internet)
#    downloads ~450GB of models to /scratch/gpfs/$USER — run inside tmux, takes a while
bash prepare.sh                  # SKIP_BIG=1 bash prepare.sh to skip 32B/72B/70B
# 3. smoke test (<15 min)
cd /scratch/gpfs/$USER/reality-monitoring && bash submit_smoke.sh
#    verify: cat results/smoke/summary.json  -> unparsed_frac < 0.10
# 4. full ladder
bash submit_all.sh
# 5. when jobs finish
source venv/bin/activate && python analyze_v2.py results
cat results_summary.md
```

## Della notes
- Login: `ssh <netid>@della.princeton.edu`. GPU dev node: `della-gpu`.
- Compute nodes have **no internet**: jobs run with `HF_HUB_OFFLINE=1`; all downloads happen in `prepare.sh`.
- Storage: use `/scratch/gpfs/$USER` (fast, big, not backed up). Check quota: `checkquota`.
- GPUs: A100-80GB standard. If the account has PLI H100 access, add `--partition=pli-c --account=<acct>` to sbatch lines.
- If sbatch rejects: run `snodes` / `sacctmgr show assoc user=$USER format=account,partition,qos` and adjust partition/qos in submit_all.sh.
- Etiquette: don't poll `squeue` in a loop (check every ≥5 min); jobs write progress to `logs/`.

## Design (what each piece answers)
| Component | Reviewer attack it answers |
|---|---|
| 7-level epistemic state (incl. numeric %, verified/guess) | "source is authorship, not reliability" — separates confidence from status |
| 3 origins (self / user / other AI) | disentangles self-blindness from instruction-hierarchy effects |
| Genuinely-sampled own claims (observational) | "injected self-claims aren't really self" |
| 3 template families, C held out | prompt-overfitting / format confounds |
| Forced-choice knowledge baseline | knowledge stratification without paraphrase artifacts |
| 3 domains incl. TruthfulQA | generality; adversarial claims |
| Bidirectional metric (retract-false vs abandon-true) | "agreeable model wins" confound |

## Outputs
- `results/<tag>/trials.jsonl` (~90MB each), `baseline_fc.jsonl`, `summary.json`
- `all_results.json`, `results_summary.md` (the emergence table = the paper's Figure 1)

## Cost estimate
~45–60 A100-GPU-hours total for the full ladder. Each job is independent; kill/skip freely.
