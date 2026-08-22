#!/bin/bash
#SBATCH --job-name=15_3B1_anchor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/vima/Code_final/slurm_logs/anchor_%j.log
#SBATCH --error=/home/vima/Code_final/slurm_logs/anchor_%j.log

# NACHTRAG nach Job 122729 (OOM-Killed waehrend Validierungsphase, Epoch 0):
# --mem=64G war zu knapp fuer num_workers=8. Fix: num_workers auf 4 gesenkt
# (Config), --mem grosszuegig auf 200G erhoeht (Knoten hat ~2TB, kein Engpass).

# =============================================================================
# Task 15.3B.1 -- Job A: Anker-Lauf (use_packed=true) bis Konvergenz.
# Zweck: aeusserer Korrektheits-Check (mIoU/std-Ratio gegen 15.2-Rauschboden)
#        + liefert eigenes data_time_ratio/Epochenzeit fuer den Vergleich
#        gegen Job B (nopack).
#
# Cluster-Fallstricke aus ZWISCHENBERICHT_CLUSTER_INFRASTRUKTUR.md beachtet:
#   - --mem explizit (sonst Default = kompletter Knotenspeicher -> Job wartet
#     effektiv auf komplett freien Knoten)
#   - GRES-Typ "a100" klein geschrieben
#   - WANDB_MODE=offline (sonst UsageError bei non-interaktivem | tee)
#   - Logs nach ~/Code_final/slurm_logs/ (NICHT /tmp -- das ist node-lokal)
#
# Start: sbatch sbatch_anchor.sh
# Status: squeue -u vima
# Log live verfolgen: tail -f /home/vima/Code_final/slurm_logs/anchor_<jobid>.log
# =============================================================================

set -euo pipefail

mkdir -p /home/vima/Code_final/slurm_logs

echo "=== Job A (packed anchor) gestartet: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Conda in nicht-interaktiver Shell aktivieren (conda activate allein
# funktioniert in sbatch-Skripten i.d.R. nicht ohne dieses Sourcing)
source /home/vima/miniconda3/etc/profile.d/conda.sh
conda activate bevwm

export WANDB_MODE=offline

cd /home/vima/Code_final

python -u train_linux.py \
    --config config_nuscenes_full_cluster.yaml \
    --phase cell

echo "=== Job A (packed anchor) beendet: $(date) ==="
