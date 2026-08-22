#!/bin/bash
# =============================================================================
# sbatch_timing_check_3ep.sh -- TASK 16a.1B Nachtrag: Steady-State-Verifikation
# =============================================================================
# Regulaerer train_linux.py-Lauf mit epochs=3 (config_timing_check_3ep.yaml,
# per sed aus config_nuscenes_full_cluster_fp16.yaml, eigenes checkpoint-dir).
# Zweck: prueft, ob die 1d-Kurzlauf-Projektion (~372 s Train-Anteil/Epoche,
# data_wait /7.1) im Epochen-Steady-State haelt -- mit BEIDEN Loadern aktiv
# (train+val, persistent_workers) statt Profiling-Einzel-Loader.
# Ablesen: pro-Epoche-Tabellenzeile (Spalten "Train" und "data%").
# Referenz VOR 1d (15.3B, Job 122767-Aera): data% 74-76, Train-Anteil ~1146 s.
# Messpunkte: Epoche 2 (sauber) und 3 (enthaelt Decode-Val, alle 3 Epochen).
# =============================================================================
#SBATCH --job-name=timing_3ep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=logs/timing_check_%j.out
# Falls eure bisherigen Skripte eine Partition setzen: hier ergaenzen.

cd ~/Code_final
mkdir -p logs

# --- conda VOR set -u (bashrc greift in Batch-Shells nicht zuverlaessig) ---
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /usr/local/anaconda3; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || {
    echo "FEHLER: env 'bevwm' nicht aktiv (import torch schlaegt fehl)."
    exit 1
}
echo "[env] python: $(command -v python)"
export WANDB_MODE=offline

set -u

[ -f config_timing_check_3ep.yaml ] || {
    echo "FEHLER: config_timing_check_3ep.yaml fehlt -- erst per sed erzeugen (siehe Anleitung)."
    exit 1
}

# /dev/shm-Cleanup bei jedem Exit (15.3B-Schema)
trap 'rm -rf "/dev/shm/${USER}_${SLURM_JOB_ID}"' EXIT

python -u train_linux.py --config config_timing_check_3ep.yaml --phase cell

echo "[timing_check] fertig. Auswertung:"
echo "  grep -E 'Epoch|data%' logs/timing_check_${SLURM_JOB_ID}.out"
