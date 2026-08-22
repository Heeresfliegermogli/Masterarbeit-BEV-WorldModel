#!/bin/bash
# =============================================================================
# sbatch_parallel_test.sh -- 16b-Baustein: 3 parallele Trainings, EIN Staging
# =============================================================================
# Frage: Koennen mehrere GPUs dieselbe /dev/shm-Kopie nutzen (memmap read-only
# = geteilte Seiten, EINE RAM-Kopie), und wie gross ist die gegenseitige
# Verlangsamung? 3x286 GB separat wuerde nicht in /dev/shm passen (858>1008)
# -- Sharing ist fuer >=3 parallele Prozesse ohnehin Pflicht.
#
# Design: 3 IDENTISCHE 1-Epochen-Laeufe (W16-Basis) auf GPU 0/1/2 gleichzeitig.
#   - Solo-Referenz: Zelle W16 aus Job 122811 = 663.6 s, data% 54.2
#   - Delta der drei Zeit-Spalten gegen 663.6 s = Konkurrenz-Kosten
#   - gleicher Seed = identische Zugriffsmuster -> realistisch fuer lambda-
#     Sweeps (dort variiert nur lambda, nicht die Datenreihenfolge)
# Mechanik:
#   - shm_cleanup: job_shared -> Prozesse raeumen NICHT auf (Guard in
#     shm_staging), Verzeichnis bleibt /dev/shm/<user>_<jobid> -> trap raeumt
#     am JOB-Ende. Staging laeuft VOR dem Parallel-Start einmal sequenziell
#     (Skip-Check ist bei gleichzeitigem Kaltstart nicht race-sicher);
#     die 3 Prozesse muessen dann "Skip-Copy" loggen.
#   - --mem=500G: /dev/shm zaehlt ins Job-cgroup (286G Pack) + 3x Worker-/
#     pinned-Puffer (~35G je) + Overhead.
# =============================================================================
#SBATCH --job-name=par3_test
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH --output=logs/par3_%j.out
# Falls eure bisherigen Skripte eine Partition setzen: hier ergaenzen.

cd ~/Code_final
mkdir -p logs

# --- conda VOR set -u (bewaehrter Block) ---
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

for i in 0 1 2; do
    [ -f "config_par_gpu${i}.yaml" ] || { echo "FEHLER: config_par_gpu${i}.yaml fehlt (sed-Anleitung)."; exit 1; }
done
grep -q "job_shared" Code/shm_staging.py || {
    echo "FEHLER: Code/shm_staging.py ohne job_shared-Strategie -- neuen Stand einspielen."
    exit 1
}

trap 'rm -rf "/dev/shm/${USER}_${SLURM_JOB_ID}"' EXIT

# --- Pre-Staging: einmal sequenziell, danach treffen alle Prozesse den Skip ---
echo "[prestage] Staging nach /dev/shm/${USER}_${SLURM_JOB_ID} ..."
python -u - <<'PYEOF'
from train_linux import load_config          # setzt sys.path auf Code/
import shm_staging
cfg = load_config("config_par_gpu0.yaml", phase_override="cell")
shm_staging.stage_and_rewrite(cfg)           # kopiert; KEIN cleanup hier
print("[prestage] fertig -- Prozesse sollten 'Skip-Copy' loggen.")
PYEOF

# --- 3 Trainings parallel, je eigene GPU + eigenes Log ---
echo "[parallel] Start: $(date +%T)"
pids=()
for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i python -u train_linux.py \
        --config "config_par_gpu${i}.yaml" --phase cell \
        > "logs/par_${SLURM_JOB_ID}_gpu${i}.log" 2>&1 &
    pids+=($!)
    echo "[parallel] GPU${i} gestartet (PID ${pids[$i]})"
done

fail=0
for i in 0 1 2; do
    wait "${pids[$i]}" || { echo "[parallel] GPU${i} FEHLGESCHLAGEN -> logs/par_${SLURM_JOB_ID}_gpu${i}.log"; fail=1; }
done
echo "[parallel] Ende:  $(date +%T)  (fail=${fail})"

echo "[parallel] Epochen-Zeilen der drei Laeufe:"
grep -H "data%" logs/par_${SLURM_JOB_ID}_gpu*.log || true
echo "[parallel] Staging-Verhalten (muss 3x Skip-Copy sein):"
grep -H "Skip-Copy\|kopiert (" logs/par_${SLURM_JOB_ID}_gpu*.log || true
exit ${fail}
