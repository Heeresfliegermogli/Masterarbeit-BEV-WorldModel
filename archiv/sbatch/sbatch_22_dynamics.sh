#!/bin/bash
# =============================================================================
# sbatch_22_dynamics.sh — Task 22.0b: Dynamik-Scores (val+train, CPU-only)
# =============================================================================
# Ein sequenzieller Lese-Pass ueber alle Seg-Latents (mmap BeeGFS, ~290G
# gelesen, nichts Grosses geschrieben — Ausgabe sind 2 kleine JSONs im Repo).
# IMMER PLAIN absetzen. NIE --export.
# =============================================================================
#SBATCH --job-name=dyn22
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dyn22_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
export WANDB_MODE=offline
set -uo pipefail

python -u compute_dynamics_scores.py --config config_seg_dump_eval.yaml
