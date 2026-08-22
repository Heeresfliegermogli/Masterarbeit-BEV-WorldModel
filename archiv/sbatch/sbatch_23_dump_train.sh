#!/bin/bash
# =============================================================================
# sbatch_23_dump_train.sh — Task 23.0: Train-Split-Vorhersagen (minimal-Modell)
# =============================================================================
# Dump der ~26k Train-Fenster-Vorhersagen (OHNE Real-Fill — der adaptierte
# Kopf soll nur echte Vorhersagen sehen) nach BeeGFS (~215G). 1 GPU, mmap.
# IMMER PLAIN. NIE --export.
# =============================================================================
#SBATCH --job-name=dump23
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/dump23_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

BG=/mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task23
mkdir -p ${BG}
python -u eval_dump_latents.py --config config_seg_dump_eval.yaml \
    --checkpoint checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt \
    --phase cell --split train --no_fill \
    --output_dir ${BG}/dump_train_minimal
echo "[done] $(ls ${BG}/dump_train_minimal | wc -l) Dateien"
