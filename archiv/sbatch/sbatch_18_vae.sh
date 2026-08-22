#!/bin/bash
# =============================================================================
# sbatch_18_vae.sh — Task 18/B1: CVAE-Kopf, beta-Sweep (3 Werte, from scratch)
# =============================================================================
# DREI CVAE-Trainings parallel (3 GPU), smooth_l1+std+beta*KL, ego off,
# Headline-Regime (50 Ep, Val-Loss-Patience, kein mIoU-Early-Stop):
#   GPU0 lambda_kl=0.001 | GPU1 0.01 | GPU2 0.1
# From scratch (kein Warm-Start: der beguenstigt Posterior-Collapse).
# Danach eval_full_val.py je Lauf mit --z_mode mean (mIoU-Guard vs off=0.6946).
# Collapse-Diagnose im Log: KL=...-Spalte (KL->0 = z wird ignoriert).
#
# IMMER PLAIN absetzen (sbatch sbatch_18_vae.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=vae18
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=16:00:00
#SBATCH --output=logs/vae18_%j.out

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
LABELS=(vae_0.001 vae_0.01 vae_0.1)           # GPU 0, 1, 2
declare -A CFG=( [vae_0.001]="config_vae_0.001.yaml" [vae_0.01]="config_vae_0.01.yaml" [vae_0.1]="config_vae_0.1.yaml" )
BASE_CFG="${CFG[vae_0.001]}"

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
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_18_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t18_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1 2; do wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN ${LABELS[$g]}" >&2; FAIL=1; }; done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"

echo "[eval] Full-Val z_mode=mean $(date +%T)"
EPIDS=()
for g in 0 1 2; do
    L="${LABELS[$g]}"; CK="checkpoints/task18/${L}/phase2/best_miou.pt"
    OUT="checkpoints/task18/${L}/phase2/miou_fullval.json"
    ELOG="logs/${SLURM_JOB_ID}_eval_${L}.log"
    if [ -f "$CK" ]; then
        CUDA_VISIBLE_DEVICES="$g" python -u eval_full_val.py --config "${CFG[$L]}" \
            --checkpoint "$CK" --phase "$PHASE" --z_mode mean --output "$OUT" > "$ELOG" 2>&1 &
        EPIDS[$g]=$!
    else echo "[eval] SKIP $L: $CK fehlt" >&2; EPIDS[$g]=""; fi
done
for g in 0 1 2; do [ -n "${EPIDS[$g]}" ] && { wait "${EPIDS[$g]}" || echo "[eval] FEHL ${LABELS[$g]}" >&2; }; done

echo "[done] 18/B1 CVAE Full-Val-mIoU z=mean (Guard vs off=0.6946):"
for L in "${LABELS[@]}"; do
    J="checkpoints/task18/${L}/phase2/miou_fullval.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-10s mIoU=%.4f (n=%s)'%('$L',d['mIoU'],d['n_samples']))" || echo "  $L: kein JSON"
done
exit "$FAIL"
