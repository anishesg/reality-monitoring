# Prompt for the co-author's Claude Code session (Princeton della)
Verified on GitHub at commit 231efc4 or later. Paste everything below the line into Claude Code on the della login node.

---

Goal: run our pre-registered post-training ladder on della GPUs, publish a live results page I can open through an SSH tunnel, and report the numbers. Everything you need is in the repo; do not invent files or steps that are not there.

1. Fresh clone, verify, read.
cd /scratch/gpfs/$USER && git clone https://github.com/anishesg/reality-monitoring.git rm-ladder && cd rm-ladder
Run `git log -1 --format=%h` and confirm it prints 04690ce or a later hash. Run `ls train slurm prereg harness analysis` and confirm these exist: train/README.md, train/run_ladder.sh, train/build_data.py, train/dpo.py, train/grpo.py, train/reward.py, train/eval_arm.py, train/results_server.py, train/requirements.txt, slurm/submit_ladder.sh, slurm/measure_stages.sh, slurm/train_dpo.sbatch, slurm/train_grpo.sbatch, slurm/eval_arm.sbatch, prereg/PREREG_causal_ladder.md, harness/run_cells_v17.py, harness/claims_hard.jsonl, analysis/analyze_v17.py. If anything is missing, stop and tell me the hash; do not improvise. Then read train/README.md and prereg/PREREG_causal_ladder.md in full. Do not use the old flattened copy of the repo elsewhere on the cluster; this clone is the only layout the scripts support.

What this is: six arms per backbone, A0 (SFT baseline, reused from existing results), A1 (generic DPO on 10k Tulu-3 preference pairs), A2 (revision DPO on our challenge dialogues), A3 (revision GRPO with a verifiable reward), A4 (A3 plus a calibrated confidence line), A5 (A1 weights then the A3 recipe). Backbones: allenai/OLMo-2-1124-7B-SFT (tag olmo) and allenai/Llama-3.1-Tulu-3-8B-SFT (tag tulu). Every arm is evaluated with the frozen harness harness/run_cells_v17.py on the held-out first 450 questions of harness/claims_hard.jsonl. Never edit anything under harness/, analysis/, results/, or prereg/. Do not commit or push unless I say so.

2. Environment (once).
export HF_HOME=/scratch/gpfs/$USER/hf
touch .hftok            # models are ungated; put a Hugging Face token here only if a download asks for one
module load anaconda3 2>/dev/null || true
python -m venv .venv && source .venv/bin/activate && pip install -r train/requirements.txt
If pip fails on CUDA/torch, install torch 2.6 matching `nvidia-smi`'s CUDA first, then rerun the requirements install. vLLM 0.8.5 needs CUDA 12.x.

3. Start the results dashboard and open it through an SSH tunnel (you do this; it is how we both watch results).
On the della login node, inside the repo: mkdir -p logs && nohup python3 train/results_server.py --port 8765 > logs/dashboard.log 2>&1 &
It is stdlib-only, binds to localhost only, and auto-refreshes every 60 s with every finished arm's numbers, capability scores, recent job logs, and figures. Confirm it is serving with `curl -s localhost:8765 | head -3` and note the login node's hostname (`hostname`), because della has several login nodes and the tunnel must target the one running the server.
Then, from your own laptop in a separate terminal: ssh -N -L 8765:localhost:8765 <your-netid>@<that-hostname>.princeton.edu  (or @della.princeton.edu if it lands on the same node), and open http://localhost:8765 in a browser. Confirm you see the page titled "STAND ladder" saying "No summary.json yet". Keep that tunnel terminal open for the duration of the runs. If the login node kills the server (some clusters reap long-running processes), restart it with the same nohup command inside a `tmux` session. Send me a screenshot of the dashboard once it is up, and again each time a new arm appears on it.

4. Find the GPU partition.
sinfo -o "%P %G %l %D" and scontrol show partition. Note the partition name, the gres string (e.g. gpu:1), and any constraint that selects 80 GB A100s. Tell me what you found.

5. Smoke test on a real GPU before spending hours.
salloc --gres=gpu:1 --time=01:00:00 --mem=80G (add --partition/--constraint as needed), then inside it:
source .venv/bin/activate && QUICK=1 bash train/run_ladder.sh --reuse-a0 allenai/OLMo-2-1124-7B-SFT olmo
This runs 4 training steps of every arm plus the full eval path (vLLM, TRL, LoRA merge, lm-eval) and should finish in under an hour. Fix environment problems here and tell me exactly what you changed. When it passes, delete its outputs: rm -rf results_ladder/olmo checkpoints/olmo.

6. Submit the real ladder.
PARTITION=<from step 4> GRES=<from step 4> CONSTRAINT=<from step 4, or omit> bash slurm/submit_ladder.sh --reuse-a0 --backbones "olmo"
then the same command with --backbones "tulu". It submits train and eval jobs chained with SLURM afterok dependencies, in the order A2, A1, A3, A4, A5, and prints the job IDs. Timing on one A100: A2 and A1 about 1.5 h train plus 20 min eval each; A3, A4, A5 about 6 to 9 h each. If the partition's max walltime is under 12 h, lower grpo.epochs in train/configs/base.yaml (e.g. 0.5) rather than dropping an arm. If GRPO OOMs on a 40 GB GPU, resubmit with VLLM=0 or a constraint for 80 GB nodes.

7. Scale rung, after step 6 is queued.
bash slurm/measure_stages.sh olmo13 olmo32     (eval-only on the published 13B and 32B SFT/DPO/Instruct checkpoints, 1 to 3 h each)
bash slurm/submit_ladder.sh --backbones "olmo32" --arms "A0 A2 A1"     (needs 4 x 80 GB per job)
Do not start 32B GRPO (slurm/train_grpo_32b.sbatch) unless I confirm the 7B A3 result is positive.

8. Monitor and report.
squeue -u $USER, tail -f logs/rm-*.out, and the dashboard. Each finished arm writes results_ladder/<tag>/<arm>/summary.json containing retain_correct, accept_valid_correction, pressure_abandon, counter_bare_abandon, excluded_frac; those five numbers per arm are the deliverable. The first milestone is results_ladder/olmo/A2/summary.json: as soon as it exists, send me its five numbers next to results_ladder/olmo/A0/summary.json, then run `git add results_ladder && git commit -m "olmo A2 results" && git push`, and keep going without waiting for me. Both drivers skip arms whose summary.json already exists, so rerunning after a failure is safe. eval_arm.py refuses adapter-only checkpoints and wants <arm>/merged. A5 depends on A1's merged weights. If an arm fails twice for the same reason, skip it, continue the others, and quote the log verbatim. Never delete results/ or results_ladder/.

9. Answer-contagion experiment (prereg/PREREG_contagion.md; inference only, ~15-25 min per model on one A100). Prefetch the
weights on the login node first (Qwen2.5-1.5B/7B/14B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, OLMo-2-1124-7B-Instruct)
so compute nodes can run offline, then:
PARTITION=<p> GRES=<g> CONSTRAINT=<c> bash slurm/submit_contagion.sh
Five jobs, results in results_contagion/<tag>/summary.json; the report is python3 analysis/report_contagion.py results_contagion/*.
Then the firewall run with the FIRM model from experiments/train_firmness.py (its merged weights on scratch):
FIREWALL=<path to merged firm model> bash slurm/submit_contagion.sh qwen7b_fw
Send me the printed report for every finished run; it lists the pre-registered predictions P1-P4 next to the numbers.

Final report format: commit hash; contagion reports; dashboard hostname, port, and a screenshot; partition/gres/constraint used; smoke test outcome and any fixes; job IDs; per-arm status; the five numbers for every finished arm; anything you changed in the repo.
