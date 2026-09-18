# train/ — causal post-training ladder (arms A0–A5)

Goal: start from an SFT-only checkpoint and *install* (A1) and *remove* (A2–A5) capitulation under challenge, evaluated with the
existing v17 harness on the held-out first 450 hard-bank questions. Predictions are frozen in `prereg/PREREG_causal_ladder.md`.

| arm | what | script |
|---|---|---|
| A0 | SFT backbone, no training (reuse `results/results_v17_cells/c_<tag>_sft` with `--reuse-a0`) | `eval_arm.py` |
| A1 | generic DPO on 10k Tulu-3 preference pairs (does ordinary preference tuning install pressure capitulation?) | `dpo.py` |
| A2 | revision-DPO: chosen = retain if right / switch if wrong, all 4 challenge kinds, both origins | `dpo.py` |
| A3 | revision-GRPO with verifiable reward (`reward.py`) | `grpo.py` |
| A4 | A3 + required CONFIDENCE line, Brier-calibration reward on the final answer (`--conf`) | `grpo.py` |
| A5 | A1 merged weights → revision-GRPO (does RLVR repair the DPO-installed defect?) | `grpo.py` |

Backbones: `allenai/OLMo-2-1124-7B-SFT` (tag `olmo`), `allenai/Llama-3.1-Tulu-3-8B-SFT` (tag `tulu`).
Training bank = SciQ (600) + hard-bank qids 450–899. Eval bank = hard-bank qids 0–449 (never trained on; `build_data.py` asserts this).
All paths are anchored to the git repo root (`harness/…`, `analysis/…`), never the cwd.

## 0. Smoke test (any machine, ~5 min, no GPU)
```
pip install trl peft datasets transformers torch pyyaml
export USE_TF=0
python3 train/reward.py                                                                 # unit checks
python3 train/build_data.py --mode injected --out data/smoke/revision.jsonl --n-questions 20 --per-cell 40
python3 train/build_data.py --mode generic  --out data/smoke/generic.jsonl  --n-generic 20
HF_DEVICE=cpu python3 train/dpo.py  --model Qwen/Qwen2.5-0.5B-Instruct --data data/smoke/revision.jsonl --out checkpoints/smoke/A2 --limit 16 --max-steps 5 --bsz 2 --grad-accum 1 --max-length 512 --lora-r 8 --merge
HF_DEVICE=cpu python3 train/grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --data data/smoke/revision.jsonl --out checkpoints/smoke/A3 --limit 4 --max-steps 1 --G 2 --bsz 2 --grad-accum 1 --max-completion 24 --lora-r 8
python3 train/eval_arm.py --model checkpoints/smoke/A2/merged --arm SMOKE --results-root results_ladder/smoke --n-questions 2 --force
```
Without vLLM/CUDA the data builder and eval fall back to `transformers` (slow, smoke only).

## 1a. Azure (one NC24ads_A100_v4, ~$3.67/h; NC96 = 4 GPUs, $14.69/h)
```
bash azure/launch_vm.sh --size Standard_NC24ads_A100_v4 --region eastus2          # dry run: prints price + quota
bash azure/launch_vm.sh --size Standard_NC24ads_A100_v4 --region eastus2 --yes    # creates VM; ~10 min cloud-init
bash azure/sync.sh up                                                              # ships the git layout to /mnt/work/reality-monitoring
ssh azureuser@$(cat azure/.last_ip)
  cd /mnt/work/reality-monitoring && echo "$HF_TOKEN" > .hftok && export HF_TOKEN
  tmux new -s ladder
  bash train/run_ladder.sh --reuse-a0 allenai/OLMo-2-1124-7B-SFT olmo             # A0(reused) A2 A1 A3 A4 A5, prints ETA table
  bash train/run_ladder.sh --reuse-a0 allenai/Llama-3.1-Tulu-3-8B-SFT tulu
bash azure/sync.sh down                                                            # pulls results_ladder/ to the laptop
bash azure/teardown.sh --yes                                                       # deallocate + delete VM and disks (stops billing)
```
`run_ladder.sh` is idempotent (skips arms whose `summary.json` exists). `QUICK=1` runs 4-step arms for a pipeline check.
`ELICITED=1` builds A2/A3 data on the model's own forced-choice answers instead of injected claims.

## 1b. della / ionic (Slurm)
```
cd /path/to/reality-monitoring            # git layout: harness/, analysis/, train/, slurm/
echo "$HF_TOKEN" > .hftok
python -m venv .venv && source .venv/bin/activate && pip install -r train/requirements.txt   # once; jobs reuse .venv
PARTITION=gpu GRES=gpu:1 CONSTRAINT=a100-80g bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu"
squeue -u $USER
```
`submit_ladder.sh` chains train → eval with `--dependency=afterok`, in the order A2, A1, A3, A4, A5 (A5 waits on A1).
Single stages: `sbatch --export=ALL,BACKBONE=...,DATA=...,OUT=... slurm/train_dpo.sbatch`, likewise `train_grpo.sbatch`, `eval_arm.sbatch`.

## 2. Reading results
`results_ladder/<tag>/<arm>/summary.json` (one row per arm) and `results_ladder/<tag>/ladder.json`. Headline columns:
`retain_correct`, `accept_valid_correction` (the frontier), `pressure_abandon` (the DPO-installed defect), `counter_bare_abandon`
(mere-mention priming), `excluded_frac`, and `capability` (mmlu / gsm8k / ifeval) when `--capability` was on.
`train/judge_check.py` samples 500 trials for an LLM-judge agreement check (needs cells with `resp`, i.e. cluster output).

## 3. Configs
`train/configs/*.yaml` share `base.yaml` via `extends:`; `python3 train/config.py train/configs/A4.yaml grpo.lr=1e-5` shows the merged
config (dotted CLI overrides, numeric coercion). The shell drivers take the same values as flags.

## Budget (one A100 80GB, LoRA r=64, 7–8B)
DPO 10k pairs × 2 epochs ≈ 1.5 h; GRPO G=8, 128-token rollouts, ~7k prompts × 1 epoch ≈ 6–9 h; v17 eval ≈ 15 min; capability ≈ 30 min.
Full ladder per backbone ≈ 25–32 GPU-hours sequential; two backbones ≈ 55–65 h (≈ $220 on NC24 pay-as-you-go).
