#!/bin/bash
# =============================================================================
# sbatch_headline.sh  —  Task 16b.9 Headline (voller Val, OHNE Early-Stop)
# =============================================================================
# ZWEI Laeufe parallel (2 GPU): Baseline (mse + Baseline-Lambdas) vs
# SmoothL1-Minimal (smooth_l1 + std, Rest 0). miou_early_stop=false in beiden
# Configs -> Training bis Val-Loss-Konvergenz. Danach eval_full_val.py auf ALLEN
# Val-Samples (n_subset=None) je Lauf -> der Headline-mIoU.
#
# IMMER PLAIN absetzen:  ssh head 'bash -lc "cd ~/Code_final && sbatch sbatch_headline.sh"'
# NIE CLI --export (zerstoert /dev/shm-Staging -- 16b.6-Lektion).
# =============================================================================
#SBATCH --job-name=headline
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=44
#SBATCH --mem=500G
#SBATCH --time=22:00:00            # kein Early-Stop -> bis ~50 Ep. moeglich + Full-Val-Eval
#SBATCH --output=logs/headline_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env 'bevwm' inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

PHASE="cell"
SHM_STAGING="Code/shm_staging.py"
LABELS=(baseline smoothl1_minimal)             # Reihenfolge = GPU 0,1
declare -A CFG=(
  [baseline]="configs/task16b/headline/config_headline_baseline.yaml"
  [smoothl1_minimal]="configs/task16b/headline/config_headline_smoothl1_minimal.yaml"
)
declare -A CKDIR=(
  [baseline]="checkpoints/task16b/headline/baseline"
  [smoothl1_minimal]="checkpoints/task16b/headline/smoothl1_minimal"
)
BASE_CFG="${CFG[baseline]}"                     # gleiche Train-Packs -> EIN Staging genuegt

# Configs vorhanden?
for L in "${LABELS[@]}"; do
    [ -f "${CFG[$L]}" ] || { echo "FEHLER: Config fehlt: ${CFG[$L]}"; exit 1; }
done

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

# --- 1. Pre-Staging (plain) ---
echo "[stage] Pre-Staging $(date +%T)"
python -u "$SHM_STAGING" --stage --config "$BASE_CFG" || { echo "FEHLER: Pre-Staging."; exit 1; }

# --- 2. Training: 2 parallel, OHNE Early-Stop ---
echo "[train] Start $(date +%T)"
FAIL=0; PIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_headline_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t16b9_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1; do
    wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN: ${LABELS[$g]}" >&2; FAIL=1; }
done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"

# --- 3. Full-Val-Eval: 2 parallel, ALLE Val-Samples (n_subset=None) ---
echo "[eval] Full-Val $(date +%T)"
EPIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; CK="${CKDIR[$L]}/phase2/best_miou.pt"
    OUT="${CKDIR[$L]}/phase2/miou_fullval.json"; ELOG="logs/${SLURM_JOB_ID}_eval_${L}.log"
    if [ -f "$CK" ]; then
        echo "[eval] GPU $g  $L  -> $OUT"
        CUDA_VISIBLE_DEVICES="$g" python -u eval_full_val.py --config "${CFG[$L]}" \
            --checkpoint "$CK" --phase "$PHASE" --output "$OUT" > "$ELOG" 2>&1 &
        EPIDS[$g]=$!
    else
        echo "[eval] SKIP $L: $CK fehlt (Training gescheitert?)" >&2; EPIDS[$g]=""
    fi
done
for g in 0 1; do
    [ -n "${EPIDS[$g]}" ] && { wait "${EPIDS[$g]}" || echo "[eval] FEHLGESCHLAGEN: ${LABELS[$g]}" >&2; }
done

# --- 4. Zusammenfassung ---
echo "[done] HEADLINE Full-Val-mIoU (OHNE Early-Stop):"
for L in "${LABELS[@]}"; do
    J="${CKDIR[$L]}/phase2/miou_fullval.json"
    if [ -f "$J" ]; then
        python -c "import json;d=json.load(open('$J'));print('  %-18s mIoU=%.4f  (full_val=%s, n=%s)'%('$L',d['mIoU'],d['full_val'],d['n_samples']))"
    else
        echo "  $L: kein JSON"
    fi
done
exit "$FAIL"
