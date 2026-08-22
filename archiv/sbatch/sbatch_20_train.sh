#!/bin/bash
# =============================================================================
# sbatch_20_train.sh — Task 20.2: Det-Transfer-Test "from six to two"
# =============================================================================
# ZWEI Det-Trainings parallel (2 GPU), Val-Loss-Selektion (Decode-Validierung
# lokal via Docker-Injection, Task 20.1):
#   GPU0 det_minimal:  smooth_l1 + lambda_std   (Seg-Sieger-Rezept)
#   GPU1 det_baseline: 6-Term (mse/cos0.1/mean0.1/std1/ssim0.1)
# ERSTLAUF-BESONDERHEIT: ensure_packed baut die Det-Packs (train 467G + val
# 100G) idempotent auf BeeGFS, DANN /dev/shm-Staging (567G -> --mem 750G!).
# Kein In-Job-Eval — Bewertung laeuft lokal (eval_dump_latents + Injection).
# ZEIT: Det-Epoche ~2.5-3x Seg (6075 Tokens) geschaetzt -> erste Epochen sind
# die MESSUNG; Resume via --resume moeglich, Snapshots alle 10 Epochen.
#
# IMMER PLAIN absetzen (sbatch sbatch_20_train.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=det20
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=750G
#SBATCH --time=34:00:00
#SBATCH --output=logs/det20_%j.out

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
LABELS=(det_minimal det_baseline)             # GPU 0, 1
declare -A CFG=( [det_minimal]="config_det_minimal.yaml" [det_baseline]="config_det_baseline.yaml" )
BASE_CFG="${CFG[det_minimal]}"

for L in "${LABELS[@]}"; do [ -f "${CFG[$L]}" ] || { echo "FEHLER: Config fehlt ${CFG[$L]}"; exit 1; }; done

echo "[pack] ensure_packed (idempotent) $(date +%T)"
python - <<'PYEOF' || { echo "FEHLER: ensure_packed."; exit 1; }
import sys
sys.path.insert(0, "Code")
import train_linux as T
import pack_latents
cfg = T.load_config("config_det_minimal.yaml", "cell")
pack_latents.ensure_packed(cfg, dtype_str="float16")
print("[pack] fertig")
PYEOF

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

echo "[stage] $(date +%T)"
python -u "$SHM_STAGING" --stage --config "$BASE_CFG" || { echo "FEHLER: Pre-Staging."; exit 1; }

echo "[train] Start $(date +%T)"
FAIL=0; PIDS=()
for g in 0 1; do
    L="${LABELS[$g]}"; LOG="logs/${SLURM_JOB_ID}_20_${L}.log"
    echo "[train] GPU $g  $L  -> $LOG"
    CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t20_${L}" \
        python -u train_linux.py --config "${CFG[$L]}" --phase "$PHASE" > "$LOG" 2>&1 &
    PIDS[$g]=$!
done
for g in 0 1; do wait "${PIDS[$g]}" || { echo "[train] FEHLGESCHLAGEN ${LABELS[$g]}" >&2; FAIL=1; }; done
echo "[train] Ende $(date +%T)  (fail=${FAIL})"
echo "[done] Checkpoints unter checkpoints/task20/{det_minimal,det_baseline}/phase2/"
exit "$FAIL"
