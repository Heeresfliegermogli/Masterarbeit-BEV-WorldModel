#!/bin/bash
# =============================================================================
# sbatch_24_dump_train.sh — Task 24.0 NACHZUEGLER: Det-Train-Dump (stride 2)
# =============================================================================
# (Erster Versuch in Job 127214 scheiterte an einem sbatch-Zusammenbaufehler —
# Backslash-Fortsetzung beim head-Schnitt; Val-Dump lief dort korrekt.)
# IMMER PLAIN. NIE --export.
# =============================================================================
#SBATCH --job-name=dump24t
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dump24t_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

BG=/mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task24
mkdir -p ${BG}
python -u eval_dump_latents.py --config config_det_baseline_eval.yaml \
    --checkpoint checkpoints/task20/det_baseline/phase2/best_val_loss.pt \
    --phase cell --split train --no_fill --stride 2 \
    --output_dir ${BG}/dump_train_det_baseline
echo "[done train] $(ls ${BG}/dump_train_det_baseline | wc -l) Dateien (~13519 erwartet)"
