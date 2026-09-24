# Reality-monitoring causal ladder: cluster bundle (2026-09-24)

This folder is self-contained. It trains and evaluates the post-training arms that fill Section 7 of the ICLR paper
("Where the policy comes from, causally"), plus one agent-chain experiment. Nothing here needs an API key or a Hugging Face token.

## What it runs
Two open models at the stage BEFORE preference tuning (`allenai/OLMo-2-1124-7B-SFT`, `allenai/Llama-3.1-Tulu-3-8B-SFT`), each trained
three ways with three random seeds, each followed by an evaluation job:
| arm | what | why |
|---|---|---|
| A1 | the lab's own published preference-tuning recipe (DPO on 10k generic Tulu-3 preference pairs), no data about pushback | does ordinary alignment training CAUSE models to cave under "are you sure?" (pre-registered H1) |
| A2 | DPO where the preferred reply after a challenge is the correct answer | control for A3: same signal, same optimizer |
| A3 | GRPO (RL) with one reward: is the final answer correct after the challenge | the fix: keep right answers, accept true corrections, lose no capability (H2/H3/H5) |
| firewall | one A3 model inserted into an 8-agent chain passing a wrong answer along | does the chain recover, as the theory predicts (P4) |
Evaluation = the paper's challenge protocol on 450 held-out questions + MMLU / GSM8K / IFEval. Predictions are frozen in `prereg/`.

## Run it (three commands on a login node)
```
unzip rm_ladder_bundle.zip && cd rm_ladder_bundle
bash RUN_ME.sh          # venv + model/data prefetch (~30 min, needs internet) + submits all jobs with dependencies
bash STATUS.sh          # any time: queue, finished arms, tails of running logs
```
Defaults are for della (A100 80GB: `CONSTRAINT=gpu80`, HF cache at `/scratch/gpfs/$USER/hf`). On another cluster:
`CONSTRAINT=<a100 constraint> PARTITION=<gpu partition> ACCOUNT=<allocation> bash RUN_ME.sh`. GPU jobs request 1 GPU, 80 GB RAM, 8 CPUs.

## On 48 GB GPUs (A6000, L40, A40)
```
GPU=a6000 CONSTRAINT=<your a6000 constraint> bash RUN_ME.sh      # add PARTITION=/ACCOUNT= if the cluster needs them
```
This switches GRPO jobs to two GPUs each (one serves rollouts with `trl vllm-serve`, one trains; same hyperparameters as the
80 GB template) and raises the wall-time caps (DPO 8 h, GRPO 30 h, eval 4 h). DPO and eval jobs use one GPU. If your partition's
maximum wall time is below 30 h, set `TIME_GRPO=<max>`; jobs requeue and resume from checkpoints.
Approximate times on A6000s (about 2.5x an A100-80GB): DPO 3-4 h, GRPO 15-20 h, eval 1.5-2 h. Total about 185 GPU-hours for the
whole queue. Wall clock: 8 GPUs about 24 h, 4 GPUs about 48 h. If you can only afford the single-seed core (A1 on both backbones, A3 and A2 on OLMo: about 40 GPU-hours, 20 h on 4 GPUs),
run the two lines below instead of `slurm/submit_core.sh` (RUN_ME.sh still does the setup; answer its prompt with Ctrl-C after prefetch or just let it run and cancel the extra seeds with `scancel`):
```
GPU=a6000 bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"
GPU=a6000 bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo" --arms "A3 A2"
```

## Timeline on 8 A100s
All 36 jobs are submitted at once; Slurm runs as many as it can in parallel. Per job: DPO training <= 4 h, GRPO training <= 12 h,
eval <= 2 h. The critical path is GRPO (12 h) then its eval (2 h): about 14-16 h wall clock with 8 GPUs, ~170 GPU-hours in total.
The first useful result (A1 seed 0 on both backbones) appears after about 6 h.

## Send back
```
bash COLLECT_RESULTS.sh     # writes rm_results_<date>.tgz (results + logs + training metadata, no weights; a few MB)
```
Send it whenever; partial results are useful too. Priority if time is short: `results_ladder/olmo/A1`, `results_ladder/tulu/A1`, then `results_ladder/olmo/A3` and `A2`.

## If something goes wrong
- A job died or was preempted: just run `bash RUN_ME.sh` again. Finished arms are skipped, trainers resume from their last checkpoint.
- `logs/<jobname>_<id>.out` holds each job's output. `FATAL: .venv missing` means step 1 of RUN_ME.sh did not finish.
- GRPO out of memory: resubmit that arm with `VLLM=0 SEED=<k> bash slurm/submit_ladder.sh --reuse-a0 --backbones olmo --arms A3` (slower rollouts, less memory).
- Wall-time limit lower than 12 h on your partition: `TIME_GRPO=08:00:00 bash slurm/submit_core.sh`; requeued jobs resume.
- Compute nodes are treated as offline (everything is prefetched by RUN_ME.sh). If a node does have internet, `RM_ONLINE=1` allows downloads.
