# Sourced by every sbatch template. Git layout, preemption-safe defaults, offline Hugging Face access on compute nodes.
set -euo pipefail
ROOT="${RM_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
WORK=/tmp/${USER}_${SLURM_JOB_ID:-local}; mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
# Models/datasets live on the big scratch filesystem and are PREFETCHED on a login node (slurm/prefetch.sh): compute nodes are
# treated as offline (set RM_ONLINE=1 to allow downloads if the cluster permits them).
export HF_HOME="${HF_HOME:-${SCRATCH_HF:-/scratch/gpfs/$USER/hf}}"; [ -d "$HF_HOME" ] || export HF_HOME="$ROOT/.hf"
[ "${RM_ONLINE:-0}" = "1" ] || export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK" USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_ENABLE_HF_TRANSFER=1 WANDB_MODE="${WANDB_MODE:-offline}"
[ -f "$ROOT/.hftok" ] && { export HF_TOKEN=$(cat "$ROOT/.hftok"); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; }
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; { module load anaconda3/2024.02 || module load anaconda3 || true; } 2>/dev/null
[ -d "$ROOT/.venv" ] || { echo "FATAL: $ROOT/.venv missing. Build it once on a login node: python -m venv .venv && pip install -r train/requirements.txt"; exit 3; }
source "$ROOT/.venv/bin/activate"
# Requeue/preemption bookkeeping: Slurm sends the signal in --signal before the kill; we only log it. Trainers checkpoint every
# --save-steps and resume from <out>/trainer/checkpoint-*; evaluators resume from <arm>/cells.jsonl. Re-running the same sbatch is safe.
_on_signal() { echo "SIGNAL $(date -u +%FT%TZ) job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0}: checkpoint state is on disk; requeue will resume" >&2; }
trap _on_signal USR1 TERM
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0} root=$ROOT hf_home=$HF_HOME offline=${HF_HUB_OFFLINE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd "$ROOT"
