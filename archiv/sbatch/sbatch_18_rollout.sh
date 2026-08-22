#!/bin/bash
# =============================================================================
# sbatch_18_rollout.sh — Task 18/B2: Rollout-Vergleich det vs. Flow-Sample
# =============================================================================
# ZWEI Rollouts (k=1..4) auf DENSELBEN 1000 seed-42-Fenstern, 1 GPU sequenziell,
# OHNE shm-Staging: (a) mode=det (deterministischer Mittelwert, rekursiv),
# (b) mode=sample (je Schritt EIN Flow-Sample, steps=1, fix geseedet).
# KERNFRAGE (BEVWorld/VFMF-Argument): degradiert der rekursive Mittelwert
# staerker als die Sample-Trajektorie (mIoU(k), std-Ratio(k))?
# IMMER PLAIN absetzen. NIE CLI --export.
# =============================================================================
#SBATCH --job-name=roll18
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/roll18_%j.out

cd ~/Code_final
mkdir -p logs predictions/task18
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || { echo "FEHLER: env bevwm inaktiv."; exit 1; }
export WANDB_MODE=offline
set -uo pipefail

CK="checkpoints/task18/flow/phase2/best_miou.pt"
[ -f "$CK" ] || { echo "FEHLER: $CK fehlt"; exit 1; }

for M in det sample; do
    python -u rollout_eval.py --config config_flow_eval.yaml \
        --checkpoint "$CK" --phase cell --k 4 --max_windows 1000 \
        --mode "$M" --flow_steps 1 \
        --output "predictions/task18/rollout_${M}.json" \
        > "logs/${SLURM_JOB_ID}_roll_${M}.log" 2>&1 || echo "[roll] FEHL $M" >&2
done

echo "[done] Rollout det vs sample (1000 Fenster):"
python - <<'PY'
import json
for m in ["det", "sample"]:
    d = json.load(open(f"predictions/task18/rollout_{m}.json"))
    c = d["curve"]
    row = "  ".join(f"k{k}: mIoU={c[str(k)]['mIoU']:.4f} std={c[str(k)]['std_ratio']:.3f}" for k in range(1, 5))
    print(f"  {m:7s} {row}")
PY
