#!/bin/bash
# =============================================================================
# sbatch_eval_film.sh  —  Task 16d: Full-Val-mIoU der FiLM-Modelle state/action
# =============================================================================
# Holt den Full-Val nach (Job 123599 wurde abgebrochen -> nur Per-Epochen-mIoU
# vorhanden). OHNE /dev/shm-Staging (config_ego_*_eval.yaml: stage_to_shm=false)
# -> Dataset mmap't packed.npy direkt von BeeGFS; RAM-Footprint winzig, kein
# OOM gegen --mem (die frueheren OOMs kamen vom 286G-tmpfs-Staging). Ein GPU,
# state dann action SEQUENZIELL (kein Parallel-RAM-Druck).
#
# IMMER PLAIN absetzen (sbatch sbatch_eval_film.sh). NIE CLI --export.
# =============================================================================
#SBATCH --job-name=eval_film
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_film_%j.out

cd ~/Code_final
mkdir -p logs
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env 'bevwm' inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

for L in state action; do
    CK="checkpoints/task16d/ego_${L}/phase2/best_miou.pt"
    OUT="checkpoints/task16d/ego_${L}/phase2/miou_fullval.json"
    if [ ! -f "$CK" ]; then echo "[eval] SKIP $L: $CK fehlt"; continue; fi
    echo "[eval] $L  Start $(date +%T)"
    python -u eval_full_val.py --config "config_ego_${L}_eval.yaml" \
        --checkpoint "$CK" --phase cell --output "$OUT" \
        > "logs/eval_film_${L}.log" 2>&1
    echo "[eval] $L  fertig $(date +%T): $(python -c "import json;print('mIoU=%.4f (n=%s)'%(lambda d:(d['mIoU'],d['n_samples']))(json.load(open('$OUT'))))" 2>/dev/null || echo 'kein JSON')"
done

echo "[done] FiLM Full-Val (Vergleich vs off=0.6946):"
for L in state action; do
    J="checkpoints/task16d/ego_${L}/phase2/miou_fullval.json"
    [ -f "$J" ] && python -c "import json;d=json.load(open('$J'));print('  %-8s mIoU=%.4f'%('$L',d['mIoU']))" || echo "  $L: kein JSON"
done
