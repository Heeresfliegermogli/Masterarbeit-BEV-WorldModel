#!/bin/bash
# =============================================================================
# sbatch_loader_diag.sh -- 16b-Baustein (vorgezogen): Rest-Engpass-Diagnose
# =============================================================================
# Nach 1d bleibt auf dem Cluster data% ~58 (Job 122810: data 333s, compute
# 240s). Hauptverdaechtiger: der EINE pin_memory-Thread des DataLoaders
# (268 MB/Batch seriell pinnen ~ 90ms ~ gemessene 98.6ms data_wait/Step;
# Indiz 2: 4->12 Worker brachte in 15.3B nichts -> serieller Engpass).
#
# Drei Zellen, je 1 Epoche, sequenziell in EINEM Job (Staging einmal):
#   W16   : num_workers 8->16, pin an     -> Kontrollarm: Worker-These
#   P0    : pin_memory=false, 8 Worker    -> Taeter-Test: pin-Thread-These
#   P0W16 : beides                        -> beste Kombi, falls P0 gewinnt
# Referenz (KEINE eigene Zelle noetig): Job 122810 = 8 Worker + pin an:
#   Train-Anteil 574s (data 333 + compute 240), data% 57-58, Epoche ~695s.
# Ablesen je Zelle: Epochen-Tabellenzeile (Train-Loss egal, "Zeit" + "data%").
# ACHTUNG P0: ohne pinned memory wird h2d langsamer/synchron -> der Gewinn
# beim Warten kann teils in compute_time wandern; es zaehlt die ZEIT-Spalte.
# =============================================================================
#SBATCH --job-name=loader_diag
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --output=logs/loader_diag_%j.out
# Falls eure bisherigen Skripte eine Partition setzen: hier ergaenzen.

cd ~/Code_final
mkdir -p logs

# --- conda VOR set -u (bewaehrter Block aus timing_check/A-B) ---
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

for f in config_diag_W16.yaml config_diag_P0.yaml config_diag_P0W16.yaml; do
    [ -f "$f" ] || { echo "FEHLER: $f fehlt -- erst per sed erzeugen (siehe Anleitung)."; exit 1; }
done
grep -q "pin_memory" Code/bev_dataloader.py || {
    echo "FEHLER: Code/bev_dataloader.py ohne pin_memory-Flag -- neuen Stand einspielen."
    exit 1
}

trap 'rm -rf "/dev/shm/${USER}_${SLURM_JOB_ID}"' EXIT

for CELL in W16 P0 P0W16; do
    echo "==================================================================="
    echo "[diag] Zelle ${CELL} -- config_diag_${CELL}.yaml (1 Epoche)"
    echo "==================================================================="
    python -u train_linux.py --config "config_diag_${CELL}.yaml" --phase cell
done

echo "[diag] fertig. Auswertung (Zeit + data% je Zelle, Referenz 122810: ~695s / 58):"
echo "  grep -E 'Zelle|data%' logs/loader_diag_${SLURM_JOB_ID}.out"
