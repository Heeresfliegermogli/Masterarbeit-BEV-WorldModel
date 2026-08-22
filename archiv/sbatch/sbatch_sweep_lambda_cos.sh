#!/bin/bash
# =============================================================================
# sbatch_sweep_parallel.sh  —  Task 16b.2 (Cluster, 1 Job = 1 Regler)
# =============================================================================
# EIN Sweep = EIN Mehr-GPU-Job. GPU-Zahl = Anzahl der Sweep-Werte. EIN
# geteiltes /dev/shm-Staging (job_shared) fuer alle Trainings des Jobs.
# Abgeleitet aus sbatch_parallel_test.sh (Job 122812): gleiche conda-/Guard-/
# trap-Konventionen; Pre-Staging jetzt ueber den sauberen shm_staging --stage
# statt Inline-Heredoc.
#
# ABLAUF:
#   0. Per-Lambda-Configs VORAB erzeugen (mk_sweep_cfgs.py, 2-Zeilen-Diff/Config)
#   1. EINMALIGES sequenzielles Pre-Staging nach /dev/shm  (shm_staging --stage)
#   2. Trainings in WELLEN: je Welle bis zu NGPU parallel, je eine GPU
#      (CUDA_VISIBLE_DEVICES). NGPU == |VALUES| -> genau 1 Welle (wie bisher).
#      Staging bleibt ZWISCHEN den Wellen liegen (gleiche SLURM_JOB_ID).
#   3. wait JE WELLE -> Fehlschlaege ueber alle Wellen sammeln
#   4. Inference je Wert (16b.3), seriell, --no-save (kein npy-Dump)
#   5. CSV-Harvest (harvest_sweep.py), nicht-fatal
#   trap: /dev/shm am JOB-Ende raeumen (auch bei scancel/Timeout/OOM)
#
# PRO SWEEP ZU EDITIEREN: nur die vier  <<< EDIT  Zeilen (job-name, --gres,
#   REGLER, VALUES). REGEL (Wellen-Modus): 1 <= GPU-Zahl <= Laenge VALUES
#   (Guard prueft). Weniger GPUs als Werte -> ceil(|VALUES|/NGPU) WELLEN,
#   EIN Job, EIN Staging. --cpus/--mem/--time mitskalieren (Formeln an den
#   Zeilen). VALUES-Reihenfolge = Wellen-Reihenfolge: Nullpunkt (Wert 0)
#   IMMER ZUERST -> landet in Welle 1, GO/NO-GO-Band frueh lesbar.
#   16b.4 lambda_grad : gpu:a100:5 (1 Welle) ODER :3 (2 Wellen)
#                       VALUES=(0 0.05 0.1 0.3 1.0)
#   16b.5 lambda_ssim : --gres=gpu:a100:3   VALUES=(0 0.1 0.3)
#   16b.6 lambda_mean : --gres=gpu:a100:3   VALUES=(0 0.1 0.3)
#   16b.6 lambda_cos  : --gres=gpu:a100:3   VALUES=(0 0.1 0.5)
# !!! IMMER PLAIN absetzen (sbatch sbatch_sweep_parallel.sh). NIE CLI --export
#     -- das zerstoert das /dev/shm-Staging (16b.6-Lektion, s. REGLER-Block).
#     Parallele Regler = Per-Regler-KOPIE dieser Datei, jede plain.
# =============================================================================

#SBATCH --job-name=sweep_lambda_cos            # <<< EDIT (pro Sweep, IN DER DATEI -- nicht via CLI!)
#SBATCH --gres=gpu:a100:3                      # <<< EDIT (16b.5: 3 Werte -> 1 Welle)
#SBATCH --cpus-per-task=66                     # ~ NGPU x 22
#SBATCH --mem=500G                             # 286G shm-Pack (EINMAL) + 3 x ~66G Worker/pinned
#SBATCH --time=13:00:00                        # 16b.5-Budget aus GEMESSENEN Zeiten (Lektion 16b.4):
                                               #   Epochenzeit 16b.4/Welle 1 = 6h21min / 23 Ep. ~= 995 s/Ep.
                                               #   (4-Wege-Konkurrenz; die 650 s aus der Loader-Diagnose
                                               #    gelten nur fuer Einzellaeufe -> nicht verwenden).
                                               #   35 Ep. (Obergrenze; 16b.4 stoppte bei 17-29) x 1000 s
                                               #   = 9h43 + Staging ~20min + 3x Inference/Harvest ~45min
                                               #   = 10h48, +20% Sicherheit -> 13h.
#SBATCH --output=logs/sweep_%x_%j.out
# Keine Partition/Account noetig (wie sbatch_parallel_test.sh, Job 122812).
# Falls eure Skripte doch eine Partition setzen: hier ergaenzen.

cd ~/Code_final
mkdir -p logs


# ==========================================================================
# SWEEP-DEFINITION (pro Sweep editieren)
# ==========================================================================
# ==========================================================================
# !!! NIE mit CLI --export absetzen !!!  (16b.6-Lektion, ~15 Debug-Jobs)
# Ein sbatch-CLI --export (ALL ODER Einzelvar) ZERSTOERT das /dev/shm-Staging:
# shutil.copy2 -> utime-ENOENT, Zielverz. verschwindet ohne unlink-Syscall
# (job_container/tmpfs-Teardown, weil SLURM mit explizitem --export Env/Session
# anders aufsetzt). Bewiesen per Bisektion (Job 123171: plain=OK, +--export=FAIL).
# REGEL: REGLER/VALUES HIER in der Datei setzen, dann PLAIN absetzen:
#     ssh head 'bash -lc "cd ~/Code_final && sbatch sbatch_sweep_parallel.sh"'
# Parallele Regler -> Per-Regler-KOPIE dieser Datei (je REGLER/VALUES/job-name),
# jede PLAIN absetzen. Die frueheren SWEEP_REGLER/--export-Beispiele sind TOT.
# HINWEIS: Basis-Config traegt lambda_mean=0.1 UND lambda_cos=0.1 -> der jeweilige
# 0.1-Sweep-Wert = 16b.4/16b.5-Anker (Seed 42, Same-Seed-Replikat), kostenlos.
# ==========================================================================
REGLER="lambda_cos"                            # <<< EDIT pro Sweep (IN DER DATEI)
VALUES=(0 0.1 0.5)                             # <<< EDIT pro Sweep (Nullpunkt zuerst!)

BASE_CFG="config_sweep_base_cluster_fp16.yaml" # 16b.1, patch_line-bereit
PHASE="cell"
N_INFER=300                                    # 15.3-Methodik (std-Ratio)

# shm_staging.py ist ein Modul und liegt in Code/ (nicht top-level). Standalone
# lauffaehig, da es nur yaml + stdlib importiert (keine Geschwister-Module).
SHM_STAGING="Code/shm_staging.py"

SWEEP_ROOT="$HOME/Code_final/checkpoints/task16b/${REGLER}_sweep"
CFG_DIR="$HOME/Code_final/configs/task16b/${REGLER}_sweep"
PRED_ROOT="$HOME/Code_final/predictions/task16b/${REGLER}_sweep"


# --- conda VOR set -u (bewaehrter Block aus sbatch_parallel_test.sh) --------
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /usr/local/anaconda3; do
    [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
done
conda activate bevwm 2>/dev/null || source activate bevwm 2>/dev/null || true
python -c "import torch" 2>/dev/null || {
    echo "FEHLER: env 'bevwm' nicht aktiv (import torch schlaegt fehl)."; exit 1; }
echo "[env] python: $(command -v python)"
export WANDB_MODE=offline

set -uo pipefail


# ==========================================================================
# DEPLOY-GUARDS: richtiger shm_staging-Stand (job_shared + --stage-Patch)?
# ==========================================================================
grep -q "job_shared" "$SHM_STAGING" || {
    echo "FEHLER: $SHM_STAGING ohne job_shared-Strategie -- neuen Stand einspielen."; exit 1; }
grep -q -- "--stage" "$SHM_STAGING" || {
    echo "FEHLER: $SHM_STAGING ohne --stage-Modus (16b.2-Patch) -- neuen Stand einspielen."; exit 1; }


# ==========================================================================
# GUARD (Wellen-Modus): 1 <= sichtbare GPU-Zahl <= Anzahl VALUES.
# NGPU < |VALUES| -> Werte laufen in ceil(|VALUES|/NGPU) Wellen (EIN Job,
# EIN Staging). NGPU > |VALUES| = GPU-Verschwendung -> --gres verkleinern.
# ==========================================================================
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ] || [ "$NGPU" -gt "${#VALUES[@]}" ]; then
    echo "[ABBRUCH] $NGPU GPUs sichtbar bei ${#VALUES[@]} Werten." >&2
    echo "          Erlaubt: 1 <= --gres <= ${#VALUES[@]} (Wellen-Modus)." >&2
    exit 1
fi
N_WAVES=$(( ( ${#VALUES[@]} + NGPU - 1 ) / NGPU ))
echo "[plan] ${#VALUES[@]} Werte auf ${NGPU} GPU(s) -> ${N_WAVES} Welle(n)."

mkdir -p "$CFG_DIR" "$PRED_ROOT"


# ==========================================================================
# CLEANUP-TRAP: geteilte /dev/shm-Kopie am JOB-Ende (job_shared -> train_linux
# raeumt NICHT selbst). --job-dir ist robust; Fallback direktes rm.
# ==========================================================================
SHM_JOB_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
trap 'echo "[trap] cleanup ${SHM_JOB_DIR}"; \
      python -u "$SHM_STAGING" --cleanup --job-dir "${SHM_JOB_DIR}" 2>/dev/null \
      || rm -rf "${SHM_JOB_DIR}"' EXIT


# ==========================================================================
# 0. Per-Lambda-Configs VORAB erzeugen (diff-verifizierbar, 2 Zeilen/Config)
# ==========================================================================
echo "[cfg] erzeuge ${#VALUES[@]} Per-Lambda-Configs ..."
python -u mk_sweep_cfgs.py --regler "$REGLER" --values "${VALUES[@]}" \
    --base "$BASE_CFG" --sweep-root "$SWEEP_ROOT" --out-dir "$CFG_DIR" || {
    echo "FEHLER: Config-Generierung fehlgeschlagen."; exit 1; }
for L in "${VALUES[@]}"; do
    [ -f "$CFG_DIR/config_${REGLER}_${L}.yaml" ] || {
        echo "FEHLER: config_${REGLER}_${L}.yaml fehlt nach Generierung."; exit 1; }
done


# ==========================================================================
# 1. EINMALIGES sequenzielles Pre-Staging nach /dev/shm/<user>_<jobid>/
#    Danach treffen ALLE Trainings den Skip-Copy (race-frei).
# ==========================================================================
echo "[stage] Pre-Staging nach ${SHM_JOB_DIR} ..."
python -u "$SHM_STAGING" --stage --config "$BASE_CFG" || {
    echo "FEHLER: Pre-Staging fehlgeschlagen."; exit 1; }


# ==========================================================================
# 2.+3. Trainings in WELLEN -- je Welle bis zu NGPU parallel (eine GPU pro
# Prozess, CUDA_VISIBLE_DEVICES=0..NGPU-1), wait je Welle, Fehlschlaege ueber
# alle Wellen sammeln. NGPU == |VALUES| -> genau 1 Welle == altes Verhalten.
# /dev/shm-Staging (job_shared, SLURM_JOB_ID-keyed) bleibt zwischen den
# Wellen liegen -> Welle-2+-Prozesse treffen den Skip-Copy. trap raeumt
# EINMAL am Job-Ende.
# ==========================================================================
echo "[train] Start: $(date +%T)"
FAIL=0
for (( OFF=0; OFF<${#VALUES[@]}; OFF+=NGPU )); do
    WAVE=( "${VALUES[@]:OFF:NGPU}" )
    echo "[train] Welle $(( OFF/NGPU + 1 ))/${N_WAVES}: ${WAVE[*]}  ($(date +%T))"
    PIDS=()
    for g in "${!WAVE[@]}"; do
        L="${WAVE[$g]}"
        CFG="$CFG_DIR/config_${REGLER}_${L}.yaml"
        LOG="logs/${SLURM_JOB_ID}_${REGLER}_${L}.log"
        echo "[train] GPU $g  ${REGLER}=${L}  -> $LOG"
        CUDA_VISIBLE_DEVICES="$g" WANDB_NAME="t16b_${REGLER}_${L}" \
            python -u train_linux.py --config "$CFG" --phase "$PHASE" > "$LOG" 2>&1 &
        PIDS[$g]=$!
    done
    for g in "${!PIDS[@]}"; do
        wait "${PIDS[$g]}" || {
            echo "[train] FEHLGESCHLAGEN: ${REGLER}=${WAVE[$g]} -> logs/${SLURM_JOB_ID}_${REGLER}_${WAVE[$g]}.log" >&2
            FAIL=1; }
    done
done
echo "[train] Ende:  $(date +%T)  (fail=${FAIL})"
# Staging-Kontrolle: bei vorgezogenem --stage muessen ALLE Laeufe skippen.
echo "[stage] Skip-Kontrolle (erwartet: nur 'Skip-Copy'/'bereits gestaged'):"
grep -H "Skip-Copy\|bereits gestaged\|kopiert (" logs/${SLURM_JOB_ID}_${REGLER}_*.log || true


# ==========================================================================
# 4. Inference je Wert (16b.3), seriell auf GPU 0. 300 Val-Samples, --no-save.
# ==========================================================================
for L in "${VALUES[@]}"; do
    CKPT="$SWEEP_ROOT/${REGLER}_${L}/phase2/best_miou.pt"
    OUT="$PRED_ROOT/${REGLER}_${L}"
    CFG="$CFG_DIR/config_${REGLER}_${L}.yaml"
    if [ -f "$CKPT" ]; then
        echo "[infer] ${REGLER}=${L}"
        python -u inference.py --config "$CFG" --phase "$PHASE" \
            --checkpoint "$CKPT" --output "$OUT" \
            --max_samples "$N_INFER" --no-save --device cuda:0 || \
            echo "[warn] Inference ${REGLER}=${L} fehlgeschlagen." >&2
    else
        echo "[infer] SKIP ${REGLER}=${L}: $CKPT fehlt (Training gescheitert?)." >&2
    fi
done


# ==========================================================================
# 5. CSV-Harvest (16b.3c) -- reine JSON-Ernte, nicht-fatal.
# ==========================================================================
CSV_OUT="$PRED_ROOT/../sweep_${REGLER}.csv"
python -u harvest_sweep.py --regler "$REGLER" --values "${VALUES[@]}" \
    --sweep-root "$SWEEP_ROOT" --pred-root "$PRED_ROOT" --out "$CSV_OUT" || \
    echo "[warn] Harvest fehlgeschlagen -- CSV spaeter lokal nachziehen." >&2

echo "[done] Sweep ${REGLER} fertig (fail=${FAIL}).  CSV: $CSV_OUT"
exit "$FAIL"
