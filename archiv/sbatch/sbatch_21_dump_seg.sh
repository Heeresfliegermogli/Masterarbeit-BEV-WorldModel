#!/bin/bash
# =============================================================================
# sbatch_21_dump_seg.sh — Task 21.1: Seg-Vorhersage-Dumps (Val) fuer GT-Eval
# =============================================================================
# Dumpt die Vorhersage-Latents der beiden 16b.9-Headline-Modelle (SmoothL1-
# Minimal + 6-Term-Baseline) als npy in ROH-Skala (latent_scale=1.0 bei Seg,
# Rueckskalierung = Identitaet) -> Injection-Format fuer tools/test.py --eval
# map (21.2). Ziel IMMER BeeGFS (Speicher-Disziplin 29.07.), NIE /home.
# ~50G je Variante (6019 x 8.4MB fp16). 1 GPU, sequenziell, mmap von BeeGFS.
# IMMER PLAIN absetzen. NIE --export.
# =============================================================================
#SBATCH --job-name=dump21seg
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dump21seg_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

BG=/mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task21

python -u eval_dump_latents.py --config config_seg_dump_eval.yaml \
    --checkpoint checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt \
    --phase cell --output_dir ${BG}/dump_seg_minimal
echo "[done minimal] $(ls ${BG}/dump_seg_minimal | wc -l) Dateien (erwartet 6019)"

python -u eval_dump_latents.py --config config_seg_dump_eval.yaml \
    --checkpoint checkpoints/task16b/headline/baseline/phase2/best_miou.pt \
    --phase cell --output_dir ${BG}/dump_seg_baseline
echo "[done baseline] $(ls ${BG}/dump_seg_baseline | wc -l) Dateien (erwartet 6019)"
