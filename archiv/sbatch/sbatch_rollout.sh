#!/bin/bash
# =============================================================================
# sbatch_rollout.sh  —  Task 16c: autoregressive Rollout-Eval (k=1..4)
# =============================================================================
# NUR Inferenz, KEIN Training, KEIN /dev/shm-Staging (config_rollout.yaml hat
# stage_to_shm:false -> liest Val direkt von BeeGFS). Beide 16b.9-Checkpoints
# parallel (baseline vs smoothl1_minimal) auf je einer GPU.
#
# IMMER PLAIN absetzen:  ssh head 'bash -lc "cd ~/Code_final && sbatch sbatch_rollout.sh"'
# NIE CLI --export (16b.6-Lektion).
# =============================================================================
#SBATCH --job-name=rollout16c
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rollout16c_%j.out

cd ~/Code_final
mkdir -p logs predictions/task16c
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env 'bevwm' inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

CFG="config_rollout.yaml"
PHASE="cell"
K=4
LABELS=(baseline smoothl1_minimal)             # GPU 0, 1
declare -A CKPT=(
  [baseline]="checkpoints/task16b/headline/baseline/phase2/best_miou.pt"
  [smoothl1_minimal]="checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt"
)

for L in "${LABELS[@]}"; do
    [ -f "${CKPT[$L]}" ] || { echo "FEHLER: Checkpoint fehlt: ${CKPT[$L]}"; exit 1; }
done

echo "[rollout] Start $(date +%T)  (K=${K}, beide Konfigs parallel)"
FAIL=0; PIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; OUT="predictions/task16c/rollout_${L}.json"
    LOG="logs/${SLURM_JOB_ID}_rollout_${L}.log"
    echo "[rollout] GPU $g  $L  -> $OUT"
    CUDA_VISIBLE_DEVICES="$g" python -u rollout_eval.py --config "$CFG" \
        --checkpoint "${CKPT[$L]}" --phase "$PHASE" --k "$K" \
        --output "$OUT" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1; do
    wait "${PIDS[$g]}" || { echo "[rollout] FEHLGESCHLAGEN: ${LABELS[$g]}" >&2; FAIL=1; }
done
echo "[rollout] Ende $(date +%T)  (fail=${FAIL})"

echo "[done] Rollout-Kurven (mIoU / std-Ratio / Persistenz je k):"
for L in "${LABELS[@]}"; do
    J="predictions/task16c/rollout_${L}.json"
    if [ -f "$J" ]; then
        echo "--- $L ---"
        python -c "import json;d=json.load(open('$J'));[print('  k=%s mIoU=%.4f std=%.4f pers=%.4f (n=%s)'%(k,c['mIoU'],c['std_ratio'],c['mIoU_pers'],c['n_windows'])) for k,c in d['curve'].items()]"
    else
        echo "  $L: kein JSON"
    fi
done
exit "$FAIL"
