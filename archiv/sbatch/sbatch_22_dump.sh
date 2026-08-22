#!/bin/bash
# =============================================================================
# sbatch_22_dump.sh — Task 22: Vorhersage-Dumps der GT-Eval-Kandidaten
# =============================================================================
# rare_w2 + rare_w8 + samp_dyn sequenziell (1 GPU, mmap BeeGFS, ~10 min/Dump).
# Ziel IMMER BeeGFS (Speicher-Disziplin). Platz-Vorpruefung im Dumper aktiv.
# IMMER PLAIN absetzen. NIE --export.
# =============================================================================
#SBATCH --job-name=dump22
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dump22_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

BG=/mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task22
mkdir -p ${BG}

for L in rare_w2 rare_w8 samp_dyn; do
    python -u eval_dump_latents.py --config config_seg_dump_eval.yaml \
        --checkpoint checkpoints/task22/${L}/phase2/best_miou.pt \
        --phase cell --output_dir ${BG}/dump_22_${L}
    echo "[done ${L}] $(ls ${BG}/dump_22_${L} | wc -l) Dateien (erwartet 6019)"
done
