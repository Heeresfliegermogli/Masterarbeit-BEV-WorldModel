#!/bin/bash
# =============================================================================
# sbatch_18_flow.sh — Task 18/B2: Flow-Matching-Residual-Kopf (1 Lauf)
# =============================================================================
# EIN Training (1 GPU): FM-Kopf (~6.5M, Paritaet zum Backbone) auf dem
# EINGEFRORENEN 16b.9-Headline-Backbone (--init_weights), 15 Epochen,
# L_FM = ||v - (r-eps)||^2, Val-Loss = fix geseedeter FM-Loss, per-Epoche-mIoU
# = 1 Flow-Sample (flow_val_steps=10). Danach Sample-Eval (eval_vae.py, K=8)
# mit Schritt-Ablation steps in {10, 1} + Panel. mIoU-Guard ist per
# Konstruktion das Backbone (0.6946) — der Test sind die SAMPLES.
#
# IMMER PLAIN absetzen (sbatch sbatch_18_flow.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=flow18
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=logs/flow18_%j.out

cd ~/Code_final
mkdir -p logs predictions/task18
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
CFG="config_flow.yaml"

[ -f "$INIT_CKPT" ] || { echo "FEHLER: Init-Checkpoint fehlt: $INIT_CKPT"; exit 1; }
[ -f "$CFG" ] || { echo "FEHLER: Config fehlt: $CFG"; exit 1; }

SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT

echo "[stage] $(date +%T)"
python -u "$SHM_STAGING" --stage --config "$CFG" || { echo "FEHLER: Pre-Staging."; exit 1; }

echo "[train] Start $(date +%T)"
python -u train_linux.py --config "$CFG" --phase "$PHASE" \
    --init_weights "$INIT_CKPT" > "logs/${SLURM_JOB_ID}_18_flow.log" 2>&1 \
    || { echo "[train] FEHLGESCHLAGEN" >&2; exit 1; }
echo "[train] Ende $(date +%T)"

CK="checkpoints/task18/flow/phase2/best_miou.pt"
[ -f "$CK" ] || { echo "FEHLER: $CK fehlt nach Training."; exit 1; }

echo "[eval] Sample-Eval steps=10 und steps=1 $(date +%T)"
for S in 10 1; do
    python -u eval_vae.py --config config_flow_eval.yaml \
        --checkpoint "$CK" --phase "$PHASE" --k 8 --n_subset 300 \
        --flow_steps "$S" \
        --output "predictions/task18/eval_flow_s${S}.json" \
        --panel_npz "predictions/task18/panel_flow_s${S}.npz" --panel_n 3 \
        > "logs/${SLURM_JOB_ID}_evalflow_s${S}.log" 2>&1 \
        || echo "[eval] FEHL steps=${S}" >&2
done

echo "[done] 18/B2 Flow Sample-Eval:"
for S in 10 1; do
    J="predictions/task18/eval_flow_s${S}.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  steps=%-3s sh_mean=%.3f sh_sample=%.3f div_mask=%.4f miou_mean=%.4f miou_sample=%.4f'%('$S',d['sharpness_ratio_mean'],d['sharpness_ratio_sample'],d['diversity_mask'],d['miou_mean'],d['miou_sample']))" || echo "  steps=$S: kein JSON"
done
