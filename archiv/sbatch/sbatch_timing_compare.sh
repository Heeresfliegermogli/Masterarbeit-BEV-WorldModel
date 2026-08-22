#!/bin/bash
#SBATCH --job-name=15_3B1_timing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/vima/Code_final/slurm_logs/timing_%j.log
#SBATCH --error=/home/vima/Code_final/slurm_logs/timing_%j.log

# NACHTRAG nach Job 122730 (OOM-Killed, epoch 2) UND Job 122729 (OOM-Killed,
# Job A, Validierungsphase Epoch 0): --mem=64G/150G war zu knapp fuer
# num_workers=8. Fix: num_workers auf 4 gesenkt (Config, konsistent mit
# Job A -- stellt auch die Fairness des Vergleichs wieder her), --mem
# grosszuegig auf 200G erhoeht (Knoten hat ~2TB, kein Engpass).

# =============================================================================
# Task 15.3B.1 -- Job B: Timing-Vergleich (use_packed=false, reines .npy
# ueber BeeGFS-Netzwerk). NUR fuer data_time_ratio/Epochenzeit gegen Job A --
# keine eigene mIoU-Validierung noetig (der .npy-Pfad ist schon aus
# Task 13/14 verifiziert). Bewusst kurz (8 Epochen, siehe
# config_nuscenes_full_cluster_nopack.yaml) -- data_time_ratio pendelt sich
# schon nach 2-3 Epochen ein.
#
# Vergleich fair, weil ALLES bis auf use_packed/epochs/patience/checkpoints.dir
# identisch zu Job A ist (per diff gegen config_nuscenes_full_cluster.yaml
# verifiziert) -- gleicher Seed, gleiche Lambdas, gleiche Hardware-Anfrage.
#
# Start: sbatch sbatch_timing_compare.sh
# Kann PARALLEL zu Job A laufen (verschiedene GPU-Allokation, verschiedene
# Checkpoint-Verzeichnisse -- keine Kollision).
# =============================================================================

set -euo pipefail

mkdir -p /home/vima/Code_final/slurm_logs

echo "=== Job B (nopack timing) gestartet: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /home/vima/miniconda3/etc/profile.d/conda.sh
conda activate bevwm

export WANDB_MODE=offline

cd /home/vima/Code_final

python -u train_linux.py \
    --config config_nuscenes_full_cluster_nopack.yaml \
    --phase cell

echo "=== Job B (nopack timing) beendet: $(date) ==="
