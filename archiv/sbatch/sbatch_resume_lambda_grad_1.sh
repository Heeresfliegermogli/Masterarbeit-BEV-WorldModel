#!/bin/bash
# =============================================================================
# sbatch_resume_lambda_grad_1.sh  —  Task 16b.4b (Cluster, Resume + Nachernte)
# =============================================================================
# Job 122859 starb am Zeitlimit mitten in lambda_grad=1.0 (Epoch 26/~28,
# Plateau 2/3). Die vier Welle-1-Werte {0, 0.05, 0.1, 0.3} sind fertig, aber
# Inference+Harvest (Schritt 4/5 im urspr. sbatch) liefen NIE, weil der Job
# vorher starb. Dieses Skript:
#   1. Setzt lambda_grad=1.0 per --resume vom letzten best_miou.pt fort
#      (Task-16b.4b-Patch: mIoU-Plateau-Zustand wird jetzt mitgeladen, kein
#      Reset auf best_miou=-1.0 mehr -> stoppt ehrlich, ueberschreibt nicht
#      versehentlich Epoch 23 mit einem schlechteren Wert).
#   2. Faehrt danach serielle Inference (16b.3-Methodik, --no-save, 300
#      Val-Samples) fuer ALLE 5 Werte nach -- die ist fuer keinen bisher
#      gelaufen.
#   3. CSV-Harvest ueber alle 5 Werte -> der Sweep ist danach vollstaendig.
# 1 GPU reicht: train_linux.py staged sich sein /dev/shm selbst
# (data.stage_to_shm=true in der Config) und raeumt am normalen Ende auch
# selbst wieder auf (train_linux.py:1398-1400). trap unten ist nur
# Sicherheitsnetz bei hartem Kill.
# =============================================================================

#SBATCH --job-name=resume_lambda_grad_1.0
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G                             # 286G shm-Pack (Solo, job_shared ungenutzt) + Worker/pinned
#SBATCH --time=02:00:00                        # ~1 Epoche erwartet (Plateau steht 2/3) + Staging + Puffer
#SBATCH --output=logs/resume_%x_%j.out

cd ~/Code_final
mkdir -p logs

REGLER="lambda_grad"
VALUES=(0 0.05 0.1 0.3 1.0)
RESUME_VALUE="1.0"

SWEEP_ROOT="$HOME/Code_final/checkpoints/task16b/${REGLER}_sweep"
CFG_DIR="$HOME/Code_final/configs/task16b/${REGLER}_sweep"
PRED_ROOT="$HOME/Code_final/predictions/task16b/${REGLER}_sweep"
PHASE="cell"
N_INFER=300

# --- conda VOR set -u (bewaehrter Block) ------------------------------------
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /usr/local/anaconda3; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || {
    echo "FEHLER: env 'bevwm' nicht aktiv (import torch schlaegt fehl)."; exit 1; }
echo "[env] python: $(command -v python)"
export WANDB_MODE=offline

set -uo pipefail

# --- Deploy-Guard: 16b.4b-Patch wirklich eingespielt? -----------------------
grep -q "resume_ckpt" train_linux.py || {
    echo "FEHLER: train_linux.py ohne 16b.4b-Resume-Patch -- neuen Stand einspielen."; exit 1; }
grep -q "best_miou" Code/checkpointing.py || {
    echo "FEHLER: Code/checkpointing.py ohne 16b.4b-Patch -- neuen Stand einspielen."; exit 1; }

# --- Sicherheitsnetz: /dev/shm raeumen falls hart gekillt -------------------
SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u Code/shm_staging.py --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT


# ==========================================================================
# 1. Resume-Training: nur lambda_grad=1.0
# ==========================================================================
RESUME_CFG="$CFG_DIR/config_${REGLER}_${RESUME_VALUE}.yaml"
RESUME_CKPT="$SWEEP_ROOT/${REGLER}_${RESUME_VALUE}/phase2/best_miou.pt"

[ -f "$RESUME_CFG" ]  || { echo "FEHLER: $RESUME_CFG fehlt."; exit 1; }
[ -f "$RESUME_CKPT" ] || { echo "FEHLER: $RESUME_CKPT fehlt (Job 122859 hat keinen best_miou.pt gespeichert?)."; exit 1; }

echo "[resume] Setze fort: ${REGLER}=${RESUME_VALUE}  von ${RESUME_CKPT}"
echo "[resume] Start: $(date +%T)"
python -u train_linux.py --config "$RESUME_CFG" --phase "$PHASE" \
    --resume "$RESUME_CKPT" 2>&1 | tee "logs/${SLURM_JOB_ID}_resume_${REGLER}_${RESUME_VALUE}.log"
RESUME_STATUS=${PIPESTATUS[0]}
echo "[resume] Ende:  $(date +%T)  (exit=${RESUME_STATUS})"

if [ "$RESUME_STATUS" -ne 0 ]; then
    echo "FEHLER: Resume-Training fehlgeschlagen (exit=${RESUME_STATUS}). Inference wird trotzdem versucht (alter Checkpoint)." >&2
fi


# ==========================================================================
# 2. Serielle Inference ueber ALLE 5 Werte (lief bisher fuer KEINEN)
# ==========================================================================
mkdir -p "$PRED_ROOT"
for L in "${VALUES[@]}"; do
    CKPT="$SWEEP_ROOT/${REGLER}_${L}/phase2/best_miou.pt"
    OUT="$PRED_ROOT/${REGLER}_${L}"
    CFG="$CFG_DIR/config_${REGLER}_${L}.yaml"
    if [ -f "$CKPT" ]; then
        echo "[infer] ${REGLER}=${L}"
        python -u inference.py --config "$CFG" --phase "$PHASE" \
            --checkpoint "$CKPT" --output "$OUT" \
            --max_samples "$N_INFER" --no-save --device cuda:0 || \
            echo "[warn] Inference ${REGLER}=${L} fehlgeschlagen." >&2
    else
        echo "[infer] SKIP ${REGLER}=${L}: $CKPT fehlt." >&2
    fi
done


# ==========================================================================
# 3. CSV-Harvest ueber alle 5 Werte
# ==========================================================================
CSV_OUT="$PRED_ROOT/../sweep_${REGLER}.csv"
python -u harvest_sweep.py --regler "$REGLER" --values "${VALUES[@]}" \
    --sweep-root "$SWEEP_ROOT" --pred-root "$PRED_ROOT" --out "$CSV_OUT" || \
    echo "[warn] Harvest fehlgeschlagen -- CSV spaeter lokal nachziehen." >&2

echo "[done] Resume + Vollernte fertig.  CSV: $CSV_OUT"
exit "$RESUME_STATUS"
