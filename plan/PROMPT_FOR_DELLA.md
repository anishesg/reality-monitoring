# Paste this into Claude Code on the della login node (after unzipping the package)

You are running pre-registered experiments for an ICLR submission on Princeton's della cluster. The zip
`reality-monitoring-della-<hash>.zip` contains everything; your job is careful execution, not redesign. Do not change
prompts, data, rewards, or metrics; fix only environment and scheduler issues, and tell me exactly what you changed.

Ground rules: GPU work only through sbatch (scripts provided); never run models on the login node. Compute nodes have no
internet, so all downloads happen on the login node in step 1. Check `squeue` at most every 5 minutes. Keep everything under
/scratch/gpfs/$USER. Never print the contents of `.hftok`.

1. Setup (login node, inside tmux).
   cd /scratch/gpfs/$USER && unzip reality-monitoring-della-*.zip && cd reality-monitoring
   ls train slurm harness analysis prereg plan   # confirm they exist; read plan/DELLA_PACKAGE.md and train/README.md
   python -m venv .venv && source .venv/bin/activate && pip install -r train/requirements.txt
   (if pip fails on torch/CUDA: install torch 2.6 matching `nvidia-smi`'s CUDA first, then rerun; vLLM 0.8.5 needs CUDA 12.x)
   echo "<hf token>" > .hftok     # only if you have one; needed for Llama/Mistral weights (licenses must be accepted on HF)
   export HF_HOME=/scratch/gpfs/$USER/hf
   bash slurm/prefetch.sh olmo tulu          # backbones + builds training data (~15 min)
   bash slurm/prefetch.sh contagion          # 6 instruct models (~60 GB)
   sinfo -o "%P %G %l %D"                    # then: export PARTITION=<gpu partition> GRES=gpu:1 CONSTRAINT=<80GB A100 constraint, or unset>
   mkdir -p logs && nohup python3 train/results_server.py --port 8765 > logs/dashboard.log 2>&1 &     # results page; tell me `hostname`

2. Smoke test (1 hour, one GPU). salloc --gres=gpu:1 --time=01:00:00 --mem=80G [--partition/--constraint as above], then:
   source .venv/bin/activate && QUICK=1 VLLM=1 bash train/run_ladder.sh --reuse-a0 allenai/OLMo-2-1124-7B-SFT olmo
   This trains 4 steps of every arm, merges, and runs the real evaluation path. PASS = results_ladder/olmo/A2/summary.json and
   A3/summary.json exist. If vLLM rollouts OOM, rerun with VLLM=0 and tell me. When it passes: rm -rf results_ladder/olmo checkpoints/olmo

3. Submit, in this order (each is resumable; re-running skips finished work):
   bash slurm/submit_contagion.sh                                                     # 5 jobs, ~25 min each
   for S in 0 1 2; do SEED=$S bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo tulu" --arms "A2 A3"; done   # 24 jobs
   If the partition walltime is under 12 h: prefix TIME_GRPO=11:30:00. If a job fails twice the same way, skip it and send me the log.

4. When results_ladder/olmo/A3/summary.json exists (first milestone, ~4 h after the first real submission), send me its five
   numbers next to results_ladder/olmo/A0/summary.json, then:
   FIREWALL=checkpoints/olmo/A3/merged bash slurm/submit_contagion.sh olmo_fw            # 1 job
   Keep going without waiting for me.

5. Report as things finish: `python3 analysis/report_contagion.py results_contagion/*` output and the contagion.png files;
   `python3 analysis/figures_ladder.py` output and figures/ladder_frontier.png, figures/ladder_capability.png; every
   summary.json under results_ladder/ and results_contagion/; job IDs; partition used; anything you changed; `seff <jobid>`
   for one DPO and one GRPO job. Zip results_ladder results_contagion figures logs and send it back.
