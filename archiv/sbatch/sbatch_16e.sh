#!/bin/bash
# =============================================================================
# sbatch_16e.sh  —  Task 16e: mIoU-Hebel 4-Frames + EMA (Headline)
# =============================================================================
# ZWEI Laeufe parallel (2 GPU), Loss = smooth_l1+std (16b.9-Sieger), ego off,
# ohne mIoU-Early-Stop (Val-Loss-Patience bis Konvergenz). Vergleich vs off=0.6946.
#   GPU0 nf4: n_frames=4 (mehr temporaler Kontext, sonst identisch zur Baseline)
#   GPU1 ema: n_frames=3 + EMA-Weight-Averaging (best_miou_ema.pt zusaetzlich)
# Danach eval_full_val.py: fuer beide best_miou.pt; fuer ema ZUSAETZLICH
# best_miou_ema.pt (der eigentliche EMA-Test).
#
# IMMER PLAIN absetzen (sbatch sbatch_16e.sh). NIE CLI --export (16b.6-Lektion).
# =============================================================================
#SBATCH --job-name=mlever16e
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=44
#SBATCH --mem=500G
#SBATCH --time=20:00:00
#SBATCH --output=logs/mlever16e_%j.out

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
LABELS=(nf4 ema)                             # GPU 0, 1
declare -A CFG=( [nf4]="config_nf4.yaml" [ema]="config_ema.yaml" )
declare -A CKDIR=( [nf4]="checkpoints/task16e/nf4" [ema]="checkpoints/task16e/ema" )
BASE_CFG="${CFG[nf4]}"

for L in "${LABELS[@]}"; do [ -f "${CFG[$L]}" ] || { echo "FEHLER: Config fehlt ${CFG[$L]}"; exit 1; }; done

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

echo "[stage] $(date +%T)"
python -u "$SHM_STAGING" --stage --config "$BASE_CFG" || { echo "FEHLER: Pre-Staging."; exit 1; }

echo "[train] Start $(date +%T)"
FAIL=0; PIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_16e_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t16e_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1; do wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN ${LABELS[$g]}" >&2; FAIL=1; }; done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"

echo "[eval] Full-Val best_miou.pt (beide, parallel) $(date +%T)"
EPIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; CK="${CKDIR[$L]}/phase2/best_miou.pt"
    OUT="${CKDIR[$L]}/phase2/miou_fullval.json"; ELOG="logs/${SLURM_JOB_ID}_eval_${L}.log"
    if [ -f "$CK" ]; then
        CUDA_VISIBLE_DEVICES="$g" python -u eval_full_val.py --config "${CFG[$L]}" \
            --checkpoint "$CK" --phase "$PHASE" --output "$OUT" > "$ELOG" 2>&1 &
        EPIDS[$g]=$!
    else echo "[eval] SKIP $L: $CK fehlt" >&2; EPIDS[$g]=""; fi
done
for g in 0 1; do [ -n "${EPIDS[$g]}" ] && { wait "${EPIDS[$g]}" || echo "[eval] FEHL ${LABELS[$g]}" >&2; }; done

# --- Zusatz-Eval: EMA-Checkpoint (der eigentliche EMA-Test) ---
CK_EMA="${CKDIR[ema]}/phase2/best_miou_ema.pt"
OUT_EMA="${CKDIR[ema]}/phase2/miou_fullval_ema.json"
if [ -f "$CK_EMA" ]; then
    echo "[eval] Full-Val best_miou_ema.pt (EMA) $(date +%T)"
    CUDA_VISIBLE_DEVICES="0" python -u eval_full_val.py --config "${CFG[ema]}" \
        --checkpoint "$CK_EMA" --phase "$PHASE" --output "$OUT_EMA" \
        > "logs/${SLURM_JOB_ID}_eval_ema_shadow.log" 2>&1 \
        || echo "[eval] FEHL ema-shadow" >&2
else echo "[eval] SKIP ema-shadow: $CK_EMA fehlt" >&2; fi

echo "[done] 16e Full-Val-mIoU (Vergleich vs off=0.6946):"
for L in "${LABELS[@]}"; do
    J="${CKDIR[$L]}/phase2/miou_fullval.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-12s mIoU=%.4f (n=%s)'%('${L} (raw)',d['mIoU'],d['n_samples']))" || echo "  $L (raw): kein JSON"
done
[ -f "$OUT_EMA" ] && python -c "import json;d=json.load(open('$OUT_EMA'));print('  %-12s mIoU=%.4f (n=%s)'%('ema (shadow)',d['mIoU'],d['n_samples']))" || echo "  ema (shadow): kein JSON"
exit "$FAIL"
