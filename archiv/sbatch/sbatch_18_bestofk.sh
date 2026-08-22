#!/bin/bash
# =============================================================================
# sbatch_18_bestofk.sh — Task 18: Best-of-K-Nachmessung (CVAE x3 + Flow x2)
# =============================================================================
# eval_vae.py (jetzt mit miou_best_of_k) ueber alle 5 Varianten, 1 GPU,
# OHNE shm-Staging (mmap BeeGFS). Outputs als *_v2.json (Originale bleiben).
# IMMER PLAIN absetzen. NIE CLI --export.
# =============================================================================
#SBATCH --job-name=bok18
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bok18_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env bevwm inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

for V in 0.001 0.01 0.1; do
    python -u eval_vae.py --config "config_vae_${V}_eval.yaml" \
        --checkpoint "checkpoints/task18/vae_${V}/phase2/best_miou.pt" \
        --phase cell --k 8 --n_subset 300 \
        --output "predictions/task18/eval_vae_${V}_v2.json" \
        > "logs/${SLURM_JOB_ID}_bok_vae_${V}.log" 2>&1 || echo "[bok] FEHL vae_${V}" >&2
done
for S in 10 1; do
    python -u eval_vae.py --config config_flow_eval.yaml \
        --checkpoint "checkpoints/task18/flow/phase2/best_miou.pt" \
        --phase cell --k 8 --n_subset 300 --flow_steps "$S" \
        --output "predictions/task18/eval_flow_s${S}_v2.json" \
        > "logs/${SLURM_JOB_ID}_bok_flow_s${S}.log" 2>&1 || echo "[bok] FEHL flow_s${S}" >&2
done

echo "[done] Best-of-K:"
for J in predictions/task18/eval_vae_0.001_v2.json predictions/task18/eval_vae_0.01_v2.json predictions/task18/eval_vae_0.1_v2.json predictions/task18/eval_flow_s10_v2.json predictions/task18/eval_flow_s1_v2.json; do
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-28s miou_mean=%.4f miou_sample=%.4f miou_BESTof8=%.4f div=%.4f'%('$J'.split('/')[-1],d['miou_mean'],d['miou_sample'],d['miou_best_of_k'],d['diversity_mask']))" || echo "  $J fehlt"
done
