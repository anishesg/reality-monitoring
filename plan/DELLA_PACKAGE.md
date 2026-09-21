# Della package: what to run, in order, and what to send back

Paste-ready for whoever has della access. Every command is resumable; re-running skips finished work. All GPU work is one
80 GB A100 per job unless stated. Total if everything runs: ~95 GPU-hours; the first two items (~5 GPU-h) already give the
paper two figures.

## 0. One-time setup (login node, ~30 min)
    cd /scratch/gpfs/$USER && git clone https://github.com/anishesg/reality-monitoring.git && cd reality-monitoring
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

## 4. Real-recipe DPO: does the public preference recipe install the behavior? (12 jobs x ~1.5 h)
    for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A1"; done
Send back: A1 summary.json files (pressure_abandon vs A0 is the number).

## 5. Scale (only if time): published 13B/32B stage checkpoints, eval only (6 jobs x ~2 h; 32B needs 2 GPUs)
    bash slurm/measure_stages.sh olmo13 olmo32

## Monitoring
`squeue -u $USER`, `tail -f logs/rm-*.out`, or the dashboard at http://localhost:8765 through the SSH tunnel. Each finished
arm writes a summary.json; nothing needs to be babysat. If a job fails twice the same way, send the log rather than retrying.
