# Della package: what to run, in order, and what to send back

Paste-ready for whoever has della access. Every command is resumable; re-running skips finished work. All GPU work is one
80 GB A100 per job unless stated. Total if everything runs: ~175 GPU-hours (steps 1-4 ~90; steps 5-8 ~85); steps 1-2 alone
(~25 GPU-h) already give the paper two figures. Seed rule: trained arms 3 seeds; measured checkpoints one deterministic pass.

## 0. One-time setup (login node, ~30 min)
    cd /scratch/gpfs/$USER && git clone https://github.com/ANONYMIZED/reality-monitoring.git && cd reality-monitoring
    git log -1 --format=%h                       # must be 57b4e53 or later
    python -m venv .venv && source .venv/bin/activate && pip install -r train/requirements.txt
    touch .hftok                                  # put a Hugging Face token inside only if a download asks for one
    HF_HOME=/scratch/gpfs/$USER/hf bash slurm/prefetch.sh olmo tulu
    HF_HOME=/scratch/gpfs/$USER/hf bash slurm/prefetch.sh contagion        # Llama/Mistral are gated: token must have accepted their licenses
    sinfo -o "%P %G %l %D"                        # note PARTITION, GRES string, and the constraint for 80 GB A100s
    mkdir -p logs && nohup python3 train/results_server.py --port 8765 > logs/dashboard.log 2>&1 &   # live results page (ssh -L 8765:localhost:8765)

Export once per shell: `export PARTITION=<p> GRES=gpu:1 CONSTRAINT=<a100-80g-or-empty> HF_HOME=/scratch/gpfs/$USER/hf`

## 1. Agent-chain replication in bf16 (5 jobs x ~25 min)  -> Figure "contagion"
    bash slurm/submit_contagion.sh
Produces results_contagion/<tag>/{summary.json, contagion.png, report.txt} for qwen7b qwen14b llama8b mistral olmo.
Send back: the five contagion.png files and `python3 analysis/report_contagion.py results_contagion/*` output.

## 2. STAND fix + its DPO control, 3 seeds x 2 backbones (24 jobs; DPO ~1.5 h, GRPO ~3 h each)  -> Figure "frontier"
    for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A2 A3"; done
Produces results_ladder/<tag>/[s<seed>/]A{2,3}/summary.json with retain_correct, accept_valid_correction, pressure_abandon,
counter_bare_abandon and the capability suite (MMLU, GSM8K, IFEval). If GRPO OOMs on a 40 GB card: `VLLM=0` or the 80 GB
constraint. If the partition's walltime is under 12 h: `TIME_GRPO=11:30:00`.
Send back: `python3 analysis/figures_ladder.py` output (figures/ladder_frontier.png, figures/ladder_capability.png) and the
summary.json files. First milestone worth a message: results_ladder/olmo/A3/summary.json.

## 3. Firewall: the trained agent inside a chain (1 job, ~30 min; after 2)
    FIREWALL=checkpoints/olmo/A3/merged bash slurm/submit_contagion.sh olmo_fw
Send back: results_contagion/olmo_fw/contagion.png and report.txt (the "firewall" line in the chain block).

## 2. Real-recipe DPO: does the public preference recipe install the behavior? (12 jobs x ~1.5 h) -- FIRST training job: it makes the stage-ladder claim causal
    for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"; done
Send back: A1 summary.json files (pressure_abandon vs A0 is the number).

## 5. Scale: published 13B/32B stage checkpoints, eval only (6 jobs x ~2 h; 32B uses 2 GPUs)  -> stage-ladder figure at scale
    bash slurm/measure_stages.sh olmo13 olmo32

## 6. Contagion at 32B and 72B, unquantized (2 jobs; 72B needs 2 GPUs, ~2 h)  -> extends the chain result above 14B
    VLLM_TP=1 MODEL=Qwen/Qwen2.5-32B-Instruct TAG=qwen32b PEERS="weak=Qwen/Qwen2.5-1.5B-Instruct,same=Qwen/Qwen2.5-32B-Instruct,strong=Qwen/Qwen2.5-72B-Instruct" sbatch --gres=gpu:1 --mem=120G --partition=$PARTITION slurm/contagion.sbatch
    VLLM_TP=2 MODEL=Qwen/Qwen2.5-72B-Instruct TAG=qwen72b PEERS="weak=Qwen/Qwen2.5-1.5B-Instruct,same=Qwen/Qwen2.5-72B-Instruct,strong=Qwen/Qwen2.5-72B-Instruct" sbatch --gres=gpu:2 --mem=200G --partition=$PARTITION slurm/contagion.sbatch
(prefetch first: `python -c "from huggingface_hub import snapshot_download as d; [d(m, allow_patterns=['*.json','*.safetensors','tokenizer*']) for m in ['Qwen/Qwen2.5-32B-Instruct','Qwen/Qwen2.5-72B-Instruct']]"` on the login node, ~210 GB)

## 7. STAND at 32B, one seed (A0 + A2 + A3; 4 GPUs per job, ~25 GPU-h)  -> does the fix persist with scale
    bash slurm/prefetch.sh olmo32 && bash slurm/submit_ladder.sh --backbones olmo32 --arms "A0 A2 A3"

## 8. A2 hyperparameter sweep on the DEV split (5 runs x ~1.5 h)  -> appendix table; selection rule is pre-registered
    sbatch --gres=gpu:1 --mem=80G --partition=$PARTITION --time=10:00:00 --job-name=rm-sweep --wrap="source slurm/_common.sh; bash train/sweep_a2.sh"

## 9. the co-author's v2 epistemic ladder, anchor models + 32B/72B only (6 jobs, ~25 GPU-h; see della/v2_epistemic_ladder/CHANGES_2026-09-21.md)
    cd della/v2_epistemic_ladder && PROJ=/scratch/gpfs/$USER/rm_v2 bash prepare.sh   # SKIP_BIG=1 to skip 72B; then edit submit_all.sh to the 6 models and run it

## 10. Elicited three-cell identification (6 jobs x ~40 min)  -> closes the paper's stated limitation
    python experiments/claimbank_3cell.py            # login node, builds claims3.jsonl (same bank as the co-author's identification runs)
    CLAIMS=$PWD/claims3.jsonl bash slurm/submit_ident_elicited.sh
Send back: results_ident_elicited/*/summary.json and ident.jsonl (responses are stored for the judge check).

## Monitoring
`squeue -u $USER`, `tail -f logs/rm-*.out`, or the dashboard at http://localhost:8765 through the SSH tunnel. Each finished
arm writes a summary.json; nothing needs to be babysat. If a job fails twice the same way, send the log rather than retrying.
