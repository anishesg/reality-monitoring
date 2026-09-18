# Sourced by every sbatch template. Mirrors harness/jobf.sh conventions (venv in $WORK, .hftok, module load), git layout.
set -euo pipefail
ROOT="${RM_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
WORK=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$WORK"; trap "rm -rf $WORK" EXIT
export HF_HOME="${HF_HOME:-$ROOT/.hf}" PIP_CACHE_DIR="$WORK/pip" TMPDIR="$WORK" USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_ENABLE_HF_TRANSFER=1
[ -f "$ROOT/.hftok" ] && { export HF_TOKEN=$(cat "$ROOT/.hftok"); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; }
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; { module load anaconda3/2024.02 || module load anaconda3 || true; } 2>/dev/null
if [ -d "$ROOT/.venv" ]; then source "$ROOT/.venv/bin/activate"; else
  python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"; pip install -q --upgrade pip; pip install -q -r "$ROOT/train/requirements.txt"; fi
echo "node=$(hostname) job=$SLURM_JOB_ID root=$ROOT"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd "$ROOT"
