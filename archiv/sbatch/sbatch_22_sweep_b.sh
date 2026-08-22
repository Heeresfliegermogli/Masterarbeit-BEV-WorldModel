#!/bin/bash
# =============================================================================
# sbatch_22_sweep.sh — Task 22: Selten-Klassen- & Dynamik-Hebel (5 Varianten)
# =============================================================================
# FUENF Laeufe parallel (5 GPU, 1 Welle, EIN /dev/shm-Staging), Basis =
# 16b.9-Minimal (smooth_l1+std), Sweep-Protokoll (miou_early_stop true,
# 300-Proxy-Inference). Anker = 16b.7-Sweepwert smooth_l1 (same-seed det.).
#   GPU0 rare_w2   GPU1 rare_w4   GPU2 rare_w8   (22.1 zellgewichtete Loss)
#   GPU3 samp_rare (22.2 CBGS-Analogon)          GPU4 samp_dyn (22.3 Dynamik)
# IMMER PLAIN absetzen (sbatch sbatch_22_sweep.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=sweep22b
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=44
#SBATCH --mem=500G
#SBATCH --time=16:00:00
#SBATCH --output=logs/sweep22b_%j.out

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
LABELS=(samp_rare samp_dyn)
declare -A CFG=()
for L in "${LABELS[@]}"; do CFG[$L]="config_22_${L}.yaml"; done

for L in "${LABELS[@]}"; do [ -f "${CFG[$L]}" ] || { echo "FEHLER: Config fehlt ${CFG[$L]}"; exit 1; }; done
[ -f /mnt/beegfs/ssd/lrt81-students/lrt81-vima/gt_masks_seg/gt_masks_train.npz ] \
    || { echo "FEHLER: gt_masks_train.npz fehlt auf BeeGFS."; exit 1; }
[ -f predictions/task22/dynamics_train.json ] \
    || { echo "FEHLER: dynamics_train.json fehlt."; exit 1; }

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

echo "[stage] $(date +%T)"
python -u "$SHM_STAGING" --stage --config "${CFG[samp_rare]}" || { echo "FEHLER: Pre-Staging."; exit 1; }

echo "[train] Start $(date +%T)"
FAIL=0; PIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_22_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t22_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1; do wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN ${LABELS[$g]}" >&2; FAIL=1; }; done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"

echo "[inference] 300-Proxy (parallel) $(date +%T)"
IPIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; CK="checkpoints/task22/${L}/phase2/best_miou.pt"
    ILOG="logs/${SLURM_JOB_ID}_inf_${L}.log"
    if [ -f "$CK" ]; then
        CUDA_VISIBLE_DEVICES="$g" python -u inference.py --config "${CFG[$L]}" \
            --checkpoint "$CK" --phase "$PHASE" --max_samples 300 --no-save \
            > "$ILOG" 2>&1 &
        IPIDS[$g]=$!
    else echo "[inference] SKIP $L: $CK fehlt" >&2; IPIDS[$g]=""; fi
done
for g in 0 1; do [ -n "${IPIDS[$g]}" ] && { wait "${IPIDS[$g]}" || echo "[inference] FEHL ${LABELS[$g]}" >&2; }; done

echo "[done] Task-22-Proxy-Ergebnisse (Anker: 16b.7 smooth_l1-Sweepwert):"
for L in "${LABELS[@]}"; do
    RS="checkpoints/task22/${L}/phase2/run_summary.json"
    [ -f "$RS" ] && python -c "import json;d=json.load(open('$RS'));print('  %-10s best_miou=%.4f stop=%s ep=%s'%('$L',d['best_miou'],d['stop_reason'],d['epochs_run']))" || echo "  $L: kein run_summary"
done
exit "$FAIL"
