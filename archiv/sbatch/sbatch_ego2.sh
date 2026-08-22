#!/bin/bash
# =============================================================================
# sbatch_ego.sh  —  Task 16d: Ego-Motion-Conditioning (state + action, Headline)
# =============================================================================
# ZWEI konditionierte Laeufe parallel (2 GPU), ohne Early-Stop bis Epoche 50,
# Loss = smooth_l1+std (16b.9-Sieger), FiLM-Conditioning. "off" = das vorhandene
# SmoothL1-Minimal-Headline (16b.9, mIoU 0.6946), wird NICHT neu trainiert.
# Danach eval_full_val.py auf allen Val-Samples je Lauf.
#
# IMMER PLAIN absetzen (sbatch sbatch_ego.sh). NIE CLI --export (16b.6-Lektion).
# =============================================================================
#SBATCH --job-name=ego16d2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=44
#SBATCH --mem=500G
#SBATCH --time=22:00:00
#SBATCH --output=logs/ego16d2_%j.out

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
LABELS=(gate token)                          # GPU 0, 1
declare -A CFG=( [gate]="config_ego_gate.yaml" [token]="config_ego_token.yaml" )
declare -A CKDIR=( [gate]="checkpoints/task16d/ego_gate" [token]="checkpoints/task16d/ego_token" )
BASE_CFG="${CFG[gate]}"

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
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_ego_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t16d_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1; do wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN ${LABELS[$g]}" >&2; FAIL=1; }; done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"

echo "[eval] Full-Val $(date +%T)"
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

echo "[done] EGO-CONDITIONING Full-Val-mIoU (Vergleich vs off=0.6946):"
for L in "${LABELS[@]}"; do
    J="${CKDIR[$L]}/phase2/miou_fullval.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-8s mIoU=%.4f (n=%s)'%('$L',d['mIoU'],d['n_samples']))" || echo "  $L: kein JSON"
done
exit "$FAIL"
