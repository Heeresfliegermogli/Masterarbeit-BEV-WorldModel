#!/bin/bash
# =============================================================================
# sbatch_16f_taskloss.sh — Task 16f/H1: Decoder-Task-Loss-Finetune (3 Werte)
# =============================================================================
# DREI Finetunes parallel (3 GPU) auf dem 16b.9-Headline-Checkpoint (0.6946):
# lambda_task 0.1 / 0.5 / 1.0, je 15 Epochen, lr 2e-5, smooth_l1+std weiter an.
# --init_weights laedt NUR Gewichte (frischer Optimizer, Epoche 0).
# Danach eval_full_val.py je Lauf. Vergleich vs off=0.6946.
#
# IMMER PLAIN absetzen (sbatch sbatch_16f_taskloss.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=tl16f
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=logs/tl16f_%j.out

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
INIT_CKPT="checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt"
LABELS=(tl_01 tl_05 tl_10)                    # GPU 0, 1, 2
declare -A CFG=( [tl_01]="config_taskloss_01.yaml" [tl_05]="config_taskloss_05.yaml" [tl_10]="config_taskloss_10.yaml" )
BASE_CFG="${CFG[tl_01]}"

[ -f "$INIT_CKPT" ] || { echo "FEHLER: Init-Checkpoint fehlt: $INIT_CKPT"; exit 1; }
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
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" \
        --init_weights "$INIT_CKPT" > "$LOG" 2>&1 &
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

echo "[done] 16f TASK-LOSS Full-Val-mIoU (Vergleich vs off=0.6946):"
for L in "${LABELS[@]}"; do
    J="checkpoints/task16f/${L}/phase2/miou_fullval.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-6s mIoU=%.4f (n=%s)'%('$L',d['mIoU'],d['n_samples']))" || echo "  $L: kein JSON"
done
exit "$FAIL"
