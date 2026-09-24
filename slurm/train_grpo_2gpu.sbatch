#!/bin/bash
#SBATCH --job-name=rm-grpo2
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=30:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x_%j.out
# 7-8B GRPO on 2 GPUs of 48 GB (A6000 / L40 / A40): GPU 0 serves rollouts via `trl vllm-serve`, GPU 1 trains the LoRA policy.
# Same hyperparameters as the single-80GB-GPU template (bsz 8, grad-accum 2, G=8), so results are comparable across clusters.
# env: BACKBONE, DATA, OUT, EXTRA (e.g. "--conf", "--seed 1")
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
PORT=${PORT:-$((8000 + SLURM_JOB_ID % 1000))}
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "$BACKBONE" --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 2048 --port $PORT > "logs/vllm-serve_${SLURM_JOB_ID}.out" 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null; rm -rf $WORK" EXIT
for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1 && break; sleep 10; done
curl -s "http://127.0.0.1:$PORT/health/" >/dev/null || { echo "vllm-serve did not come up; see logs/vllm-serve_${SLURM_JOB_ID}.out"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python train/grpo.py --model "$BACKBONE" --data "$DATA" --out "$OUT" --merge --vllm-server 127.0.0.1:$PORT --save-steps "${SAVE_STEPS:-25}" ${EXTRA:-} &
wait $!
echo "DONE-GRPO $OUT"
