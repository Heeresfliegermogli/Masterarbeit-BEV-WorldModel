#!/bin/bash
# =============================================================================
# sbatch_20_dump_baseline.sh — Task 20: Vorhersage-Dump det_minimal (Val)
# =============================================================================
# 1 GPU, ohne shm-Staging (mmap BeeGFS). Dump -> predictions/task20/dump_det_minimal
# (5743 Vorhersagen + 276 Real-Fill, Roh-Skala fp16). IMMER PLAIN. NIE --export.
# =============================================================================
#SBATCH --job-name=dump20m
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dump20m_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

python -u eval_dump_latents.py --config config_det_minimal_eval.yaml \
    --checkpoint checkpoints/task20/det_minimal/phase2/best_val_loss.pt \
    --phase cell --output_dir predictions/task20/dump_det_minimal
echo "[done] $(ls predictions/task20/dump_det_minimal | wc -l) Dateien (erwartet 6019)"
