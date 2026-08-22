#!/bin/bash
# =============================================================================
# sbatch_18_flow30.sh — Task 18/B2.1: Flow-Kalibrierungs-Nachlauf
# =============================================================================
# (1) FM-Kopf 30 Epochen (statt 15) auf frozen 16b.9-Backbone from scratch.
# (2) Sample-Eval (K=8, steps=1) mit SKALEN-SWEEP s in {1.0, 0.8, 0.6, 0.5}
#     (post-hoc Residual-Skalierung) -> Ziel: std-Ratio ~1.0 bei minimalem
#     mIoU-Verlust. KERNFRAGE: schliesst laengeres Training + Skalierung den
#     -0.011-Kalibrierungsrest (Best-of-8 >= mean)?
# IMMER PLAIN absetzen. NIE CLI --export.
# =============================================================================
#SBATCH --job-name=flow30
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=logs/flow30_%j.out

cd ~/Code_final
mkdir -p logs predictions/task18
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env bevwm inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

PHASE="cell"
SHM_STAGING="Code/shm_staging.py"
INIT_CKPT="checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt"
[ -f "$INIT_CKPT" ] || { echo "FEHLER: Init-Checkpoint fehlt"; exit 1; }

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

echo "[stage] $(date +%T)"
python -u "$SHM_STAGING" --stage --config config_flow30.yaml || { echo "FEHLER: Staging."; exit 1; }

echo "[train] Start $(date +%T)"
python -u train_linux.py --config config_flow30.yaml --phase "$PHASE" \
    --init_weights "$INIT_CKPT" > "logs/${SLURM_JOB_ID}_18_flow30.log" 2>&1 \
    || { echo "[train] FEHLGESCHLAGEN" >&2; exit 1; }
echo "[train] Ende $(date +%T)"

CK="checkpoints/task18/flow30/phase2/best_miou.pt"
[ -f "$CK" ] || { echo "FEHLER: $CK fehlt."; exit 1; }

echo "[eval] Skalen-Sweep $(date +%T)"
for SC in 1.0 0.8 0.6 0.5; do
    python -u eval_vae.py --config config_flow_eval.yaml \
        --checkpoint "$CK" --phase "$PHASE" --k 8 --n_subset 300 \
        --flow_steps 1 --flow_scale "$SC" \
        --output "predictions/task18/eval_flow30_sc${SC}.json" \
        > "logs/${SLURM_JOB_ID}_eval_sc${SC}.log" 2>&1 || echo "[eval] FEHL sc=$SC" >&2
done

echo "[done] 18/B2.1 Kalibrierung (30 Ep, steps=1):"
for SC in 1.0 0.8 0.6 0.5; do
    J="predictions/task18/eval_flow30_sc${SC}.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  s=%-4s std_s=%.3f sh_s=%.3f div=%.4f miou_s=%.4f best8=%.4f (mean=%.4f)'%('$SC',d['std_ratio_sample'],d['sharpness_ratio_sample'],d['diversity_mask'],d['miou_sample'],d['miou_best_of_k'],d['miou_mean']))" || echo "  s=$SC: kein JSON"
done
