#!/bin/bash
#SBATCH --job-name=15_3B2_fp16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/vima/Code_final/slurm_logs/fp16_%j.log
#SBATCH --error=/home/vima/Code_final/slurm_logs/fp16_%j.log

# =============================================================================
# Task 15.3B.2+15.3B.3 -- float16 + /dev/shm KOMBINIERT.
# Zweck: fp16 + Staging produktiv. mIoU lokal schon validiert (~0.004 unter
# fp32-Anker, weit unter Rauschboden). Hier geht es um den echten Cluster-Lauf
# mit halbem Speicherbedarf. ACHTUNG: fp16-Packs muessen auf BeeGFS existieren
# -- alte fp32-Packs ggf. vorher loeschen (anderer Pfad: latents_packed_fp16/).
#
# --mem=400G: fp16-gepackt (train ~220G + val ~48G = 268G) in /dev/shm +
# Puffer. HALBIERT ggue. dem fp32-Staging-Lauf (700G) -> findet VIEL leichter
# einen freien Slot (700G-Jobs kamen bei Cluster-Last nicht durch bzw. OOM auf
# teilbelegten Knoten). Das ist der eigentliche Grund fuer float16 hier.
#
# CLEANUP: doppelt abgesichert --
#   1) train_linux.py ruft am NORMALEN Ende shm_staging.cleanup(cfg)
#   2) der trap unten feuert bei JEDEM Skript-Ende (auch scancel/Timeout/OOM,
#      wo Python nicht mehr laeuft) und loescht das job-eigene /dev/shm-Dir.
#      Verhindert liegengebliebenen RAM-Muell auf dem geteilten Knoten.
#
# Start: sbatch sbatch_shm_stage.sh
# =============================================================================

set -uo pipefail

mkdir -p /home/vima/Code_final/slurm_logs

# --- Cleanup-trap: feuert bei EXIT (normal, Fehler, SIGTERM durch scancel/
# Timeout). Loescht das job-eigene /dev/shm-Verzeichnis. Idempotent/harmlos,
# falls train_linux.py schon selbst aufgeraeumt hat. ---
cleanup_shm() {
    echo "=== [trap] /dev/shm-Cleanup: $(date) ==="
    source /home/vima/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate bevwm 2>/dev/null
    python -u /home/vima/Code_final/Code/shm_staging.py \
        --cleanup --config /home/vima/Code_final/config_nuscenes_full_cluster_fp16.yaml \
        || echo "[trap] cleanup-Aufruf fehlgeschlagen (evtl. schon geraeumt)"
}
trap cleanup_shm EXIT

echo "=== Job 15.3B.3 (shm staging) gestartet: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--- /dev/shm auf diesem Rechenknoten (VOR Staging) ---"
df -h /dev/shm

source /home/vima/miniconda3/etc/profile.d/conda.sh
conda activate bevwm

export WANDB_MODE=offline

cd /home/vima/Code_final

python -u train_linux.py \
    --config config_nuscenes_full_cluster_fp16.yaml \
    --phase cell

echo "=== Job 15.3B.3 (shm staging) beendet: $(date) ==="
# (trap raeumt danach /dev/shm auf)
