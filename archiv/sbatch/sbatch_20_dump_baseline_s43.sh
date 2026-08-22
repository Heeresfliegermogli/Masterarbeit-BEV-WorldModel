#!/bin/bash
# =============================================================================
# sbatch_20_dump_baseline_s43.sh — Task 20: Vorhersage-Dump det_baseline (Val)
# =============================================================================
# 1 GPU, ohne shm-Staging (mmap BeeGFS). Dump -> /mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task20/dump_det_baseline_s43
# (5743 Vorhersagen + 276 Real-Fill, Roh-Skala fp16). IMMER PLAIN. NIE --export.
# =============================================================================
#SBATCH --job-name=dump20bs43
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dump20bs43_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

python -u eval_dump_latents.py --config config_det_baseline_eval.yaml \
    --checkpoint checkpoints/task20/det_baseline_s43/phase2/best_val_loss.pt \
    --phase cell --output_dir /mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task20/dump_det_baseline_s43
echo "[done] $(ls /mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task20/dump_det_baseline_s43 | wc -l) Dateien (erwartet 6019)"
