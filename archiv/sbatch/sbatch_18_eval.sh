#!/bin/bash
# =============================================================================
# sbatch_18_eval.sh — Task 18/B1: Sample-Eval (eval_vae.py) fuer alle 3 beta
# =============================================================================
# 1 GPU, sequenziell, OHNE shm-Staging (config_vae_*_eval.yaml, mmap BeeGFS —
# Lektion 16d-Eval: tmpfs-Staging zaehlt gegen --mem -> OOM). Je Lauf:
# 300 Subset-Samples x K=8 Prior-Samples -> Schaerfe/Diversitaet/mIoU-JSON;
# Panel-npz (real/mean/3 Samples) fuer alle drei.
# IMMER PLAIN absetzen. NIE CLI --export.
# =============================================================================
#SBATCH --job-name=vae18eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/vae18eval_%j.out

cd ~/Code_final
mkdir -p logs predictions/task18
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env 'bevwm' inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

for V in 0.001 0.01 0.1; do
    CK="checkpoints/task18/vae_${V}/phase2/best_miou.pt"
    [ -f "$CK" ] || { echo "[eval_vae] SKIP ${V}: $CK fehlt"; continue; }
    echo "[eval_vae] beta=${V}  Start $(date +%T)"
    python -u eval_vae.py --config "config_vae_${V}_eval.yaml" \
        --checkpoint "$CK" --phase cell --k 8 --n_subset 300 \
        --output "predictions/task18/eval_vae_${V}.json" \
        --panel_npz "predictions/task18/panel_${V}.npz" --panel_n 3 \
        > "logs/${SLURM_JOB_ID}_evalvae_${V}.log" 2>&1 \
        || echo "[eval_vae] FEHL ${V}" >&2
done

echo "[done] Sample-Eval-Ergebnisse:"
for V in 0.001 0.01 0.1; do
    J="predictions/task18/eval_vae_${V}.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  b=%-6s sh_mean=%.3f sh_sample=%.3f div_mask=%.4f miou_mean=%.4f miou_sample=%.4f'%('$V',d['sharpness_ratio_mean'],d['sharpness_ratio_sample'],d['diversity_mask'],d['miou_mean'],d['miou_sample']))" || echo "  $V: kein JSON"
done
