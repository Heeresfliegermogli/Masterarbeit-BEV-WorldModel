#!/bin/bash
# =============================================================================
# sbatch_16f_cap.sh — Task 16f/H2+H3: Kapazitaets-Check + Residual (3 Laeufe)
# =============================================================================
# DREI Trainings from scratch parallel (3 GPU), smooth_l1+std, ego off,
# Val-Loss-Patience bis Konvergenz (kein mIoU-Early-Stop):
#   GPU0 cap_l8:   n_layers 4->8            (Kapazitaet: Tiefe)
#   GPU1 cap_d384: d_model 256->384, d_ff 1536 (Kapazitaet: Breite)
#   GPU2 residual: output_mode=residual     (hartes Residual statt Gate)
# Danach eval_full_val.py je Lauf. Vergleich vs off=0.6946.
# Zeit: l8 ~2x Epochenzeit der Baseline (~10h Lauf) -> 22h Limit mit Puffer.
#
# IMMER PLAIN absetzen (sbatch sbatch_16f_cap.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=cap16f
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=22:00:00
#SBATCH --output=logs/cap16f_%j.out

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
LABELS=(cap_l8 cap_d384 residual)             # GPU 0, 1, 2
declare -A CFG=( [cap_l8]="config_cap_l8.yaml" [cap_d384]="config_cap_d384.yaml" [residual]="config_residual.yaml" )
BASE_CFG="${CFG[cap_l8]}"

for L in "${LABELS[@]}"; do [ -f "${CFG[$L]}" ] || { echo "FEHLER: Config fehlt ${CFG[$L]}"; exit 1; }; done

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

echo "[stage] $(date +%T)"
python -u "$SHM_STAGING" --stage --config "$BASE_CFG" || { echo "FEHLER: Pre-Staging."; exit 1; }

echo "[train] Start $(date +%T)"
FAIL=0; PIDS=()
for g in 0 1 2; do
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_16f_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t16f_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1 2; do wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN ${LABELS[$g]}" >&2; FAIL=1; }; done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"

echo "[eval] Full-Val $(date +%T)"
EPIDS=()
for g in 0 1 2; do
    L="${LABELS[$g]}"; CK="checkpoints/task16f/${L}/phase2/best_miou.pt"
    OUT="checkpoints/task16f/${L}/phase2/miou_fullval.json"; ELOG="logs/${SLURM_JOB_ID}_eval_${L}.log"
    if [ -f "$CK" ]; then
        CUDA_VISIBLE_DEVICES="$g" python -u eval_full_val.py --config "${CFG[$L]}" \
            --checkpoint "$CK" --phase "$PHASE" --output "$OUT" > "$ELOG" 2>&1 &
        EPIDS[$g]=$!
    else echo "[eval] SKIP $L: $CK fehlt" >&2; EPIDS[$g]=""; fi
done
for g in 0 1 2; do [ -n "${EPIDS[$g]}" ] && { wait "${EPIDS[$g]}" || echo "[eval] FEHL ${LABELS[$g]}" >&2; }; done

echo "[done] 16f KAPAZITAET+RESIDUAL Full-Val-mIoU (Vergleich vs off=0.6946):"
for L in "${LABELS[@]}"; do
    J="checkpoints/task16f/${L}/phase2/miou_fullval.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-9s mIoU=%.4f (n=%s)'%('$L',d['mIoU'],d['n_samples']))" || echo "  $L: kein JSON"
done
exit "$FAIL"
