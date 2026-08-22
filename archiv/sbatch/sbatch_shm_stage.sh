#!/bin/bash
#SBATCH --job-name=15_3B3_shm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=620G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/vima/Code_final/slurm_logs/shm_%j.log
#SBATCH --error=/home/vima/Code_final/slurm_logs/shm_%j.log

# =============================================================================
# Task 15.3B.3 -- /dev/shm-Staging-Test (fp32, verlustfrei).
# Zweck: den I/O-Effekt SAUBER isolieren. mIoU MUSS gleich Job A (Anker,
# 15.3B.1) bleiben (bit-identischer Datenpfad) -- was wir messen wollen, ist
# NUR data% und Epochenzeit. Erwartung: data% faellt drastisch von ~89%
# (BeeGFS, Job A) Richtung einstellig, Epochenzeit von ~3100s deutlich runter.
#
# --mem=620G: fp32-gepackt (train 440G + val 95G = 535G) landet in /dev/shm
# und zaehlt gegen das Cgroup-Limit -> grosszuegig + Puffer fuer Worker/pinned
# memory. Der Knoten hat ~1TB /dev/shm (auf Rechenknoten verifiziert).
# HINWEIS: 620G --mem kann die Queue-Zeit erhoehen (grosse Anforderung) --
# mit float16 (15.3B.2) spaeter halbierbar auf ~350G.
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
        --cleanup --config /home/vima/Code_final/config_nuscenes_full_cluster_shm.yaml \
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
    --config config_nuscenes_full_cluster_shm.yaml \
    --phase cell

echo "=== Job 15.3B.3 (shm staging) beendet: $(date) ==="
# (trap raeumt danach /dev/shm auf)
