#!/bin/bash
#SBATCH --job-name=rm-grpo32
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
# 32B GRPO on 4 GPUs: GPUs 0-1 serve rollouts via `trl vllm-serve` (tensor parallel 2); GPUs 2-3 hold the sharded policy.
# env: BACKBONE, DATA, OUT, EXTRA (e.g. "--conf --device-map auto --bsz 1 --grad-accum 16")
source "${RM_ROOT:-$SLURM_SUBMIT_DIR}/slurm/_common.sh"
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model "$BACKBONE" --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 2048 --port $PORT > "logs/vllm-serve_${SLURM_JOB_ID}.out" 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null; rm -rf $WORK" EXIT
for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1 && break; sleep 10; done
curl -s "http://127.0.0.1:$PORT/health/" >/dev/null || { echo "vllm-serve did not come up; see logs/vllm-serve_${SLURM_JOB_ID}.out"; exit 1; }
CUDA_VISIBLE_DEVICES=2,3 python train/grpo.py --model "$BACKBONE" --data "$DATA" --out "$OUT" --merge --vllm-server 127.0.0.1:$PORT --bsz 4 --grad-accum 4 ${EXTRA:-}
echo "DONE-GRPO32 $OUT"
