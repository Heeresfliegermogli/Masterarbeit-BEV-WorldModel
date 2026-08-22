#!/bin/bash
# =============================================================================
# sbatch_profile_1d_ab.sh -- TASK 16a.1B: 1d-Wirkung auf dem Cluster messen
# =============================================================================
# A/B in EINEM Job (ein Queue-Slot, identischer Knoten):
#   Lauf A = VOR 1d:  alte bev_dataset.py aus backup_pre_16a1b/ (Loader upcastet
#            fp16->fp32 auf der CPU; das .float() im neuen profile_train ist auf
#            fp32-Rueckgabe ein No-op -> exakt der alte Datenpfad).
#   Lauf B = NACH 1d: neue bev_dataset.py (fp16 durch collate/pin/h2d,
#            Upcast erst auf der GPU).
# 1b (fused AdamW, cudnn.benchmark) ist in BEIDEN Laeufen aktiv -> A/B isoliert
# exakt den 1d-Effekt. Kennzahlen: data_wait + h2d (Vergleich A vs B).
# /dev/shm-Staging laeuft einmalig in Lauf A (~10 min, fp16-Pack ~268G),
# Lauf B skippt idempotent (gleiche SLURM_JOB_ID -> gleiches Verzeichnis).
# =============================================================================
#SBATCH --job-name=prof_1d_ab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --output=logs/profile_1d_ab_%j.out
# Falls eure bisherigen Skripte eine Partition setzen (sbatch_fp16_cluster.sh
# als Referenz), hier ergaenzen:  #SBATCH --partition=...

cd ~/Code_final
mkdir -p logs predictions/task16a_1b_cluster

# --- conda VOR set -u aktivieren. source ~/.bashrc greift in nicht-inter-
#     aktiven sbatch-Shells oft NICHT (Interaktiv-Guard am bashrc-Anfang
#     bricht ab, bevor die conda-Init geladen ist) -> direkt das conda-Profil
#     sourcen, gaengige Pfade durchprobieren. Falls euer bewaehrtes
#     sbatch_fp16_cluster.sh eine andere Zeile nutzt: die hier einsetzen.
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /usr/local/anaconda3; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true

# Fail-fast statt stillem System-Python: ohne torch ist alles Weitere sinnlos.
python -c "import torch" 2>/dev/null || {
    echo "FEHLER: env 'bevwm' nicht aktiv (import torch schlaegt fehl)."
    echo "        conda-Aktivierungszeile aus sbatch_fp16_cluster.sh uebernehmen."
    exit 1
}
echo "[env] python: $(command -v python)"
export WANDB_MODE=offline

set -u

# Struktur-Guards: bev_dataset.py lebt im Unterverzeichnis Code/ (grosses C --
# Vorsicht, es gab mal einen code/Code-Case-Bug, vgl. 15.3B §9).
[ -f Code/bev_dataset.py ] || { echo "FEHLER: Code/bev_dataset.py nicht gefunden -- Struktur pruefen"; exit 1; }
[ -f backup_pre_16a1b/bev_dataset.py ] || { echo "FEHLER: backup_pre_16a1b/bev_dataset.py fehlt -- Backup-Schritt nachholen"; exit 1; }

# --- Doppelter Sicherheits-trap (greift bei JEDEM Exit, auch OOM/Timeout/scancel):
#     (1) Code/bev_dataset.py-Restore, falls der Job mitten in Lauf A stirbt
#         (sonst blieb die ALTE Version aktiv liegen)
#     (2) /dev/shm-Cleanup nach 15.3B-Schema /dev/shm/<user>_<jobid>
trap 'cp -f bev_dataset_16a1b_NEU.py Code/bev_dataset.py 2>/dev/null; \
      rm -f bev_dataset_16a1b_NEU.py; \
      rm -rf "/dev/shm/${USER}_${SLURM_JOB_ID}"' EXIT

CFG=config_nuscenes_full_cluster_fp16.yaml

echo "==================================================================="
echo "[A/B] Lauf A: VOR 1d (alte bev_dataset.py) -- inkl. einmaligem Staging"
echo "==================================================================="
cp Code/bev_dataset.py bev_dataset_16a1b_NEU.py          # neue Version sichern
cp backup_pre_16a1b/bev_dataset.py Code/bev_dataset.py   # alte Version aktivieren
python -u profile_train.py --config "$CFG" --phase cell --active 25 \
       --no-ssim-ab --no-trace \
       --out predictions/task16a_1b_cluster/A_vor_1d

echo "==================================================================="
echo "[A/B] Lauf B: NACH 1d (neue bev_dataset.py) -- Staging-Skip erwartet"
echo "==================================================================="
cp bev_dataset_16a1b_NEU.py Code/bev_dataset.py          # neue Version zurueck
rm -f bev_dataset_16a1b_NEU.py
python -u profile_train.py --config "$CFG" --phase cell --active 25 \
       --no-ssim-ab --no-trace \
       --out predictions/task16a_1b_cluster/B_nach_1d

echo "[A/B] fertig. Reports:"
echo "      predictions/task16a_1b_cluster/A_vor_1d/profile_report.md"
echo "      predictions/task16a_1b_cluster/B_nach_1d/profile_report.md"
