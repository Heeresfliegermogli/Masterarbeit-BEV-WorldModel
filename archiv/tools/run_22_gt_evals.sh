#!/bin/bash
# =============================================================================
# run_22_gt_evals.sh — Task 22: lokale GT-Eval-Kette (laeuft unbeaufsichtigt)
# =============================================================================
# 0) wartet auf die 3 BeeGFS-Dumps (Job 125773), 1) rsynct sie, 2) Full-Val-
# Injection je Variante, 3) Dynamik-Quartil-Evals fuer samp_dyn/rare_w2/
# minimal (Task-21-Dump als Basis-Vergleich). Alles sequenziell (1 GPU).
# Logs: /home/vima/det_latents_out/eval22_*.log ; Ende-Marker: eval22_DONE
# =============================================================================
set -uo pipefail
OUT=/home/vima/det_latents_out
BG=/mnt/beegfs/ssd/lrt81-students/lrt81-vima/dumps_task22
VARIANTS=(rare_w2 rare_w8 samp_dyn)
SEGCFG=configs/nuscenes/seg/fusion-bev256d2-lss.yaml
CKPT=pretrained/bevfusion-seg.pth
rm -f ${OUT}/eval22_DONE

log() { echo "[$(date +%H:%M:%S)] $*" >> ${OUT}/eval22_orchestrator.log; }

# --- 0) auf Dumps warten ----------------------------------------------------
log "warte auf BeeGFS-Dumps..."
while true; do
    OK=1
    for L in "${VARIANTS[@]}"; do
        N=$(ssh head "ls ${BG}/dump_22_${L} 2>/dev/null | wc -l")
        [ "$N" = "6019" ] || OK=0
    done
    [ "$OK" = "1" ] && break
    sleep 180
done
log "Dumps komplett."

# --- 1) rsync ---------------------------------------------------------------
for L in "${VARIANTS[@]}"; do
    rsync -a head:${BG}/dump_22_${L} ${OUT}/ >> ${OUT}/eval22_orchestrator.log 2>&1
    log "rsync ${L}: $(ls ${OUT}/dump_22_${L} | wc -l) Dateien"
done

wait_gpu() { until ! docker exec seg_gate pgrep -f tools/test.py >/dev/null 2>&1; do sleep 60; done; }

# --- 2) Full-Val-Injection je Variante --------------------------------------
for L in "${VARIANTS[@]}"; do
    wait_gpu
    log "Full-Val-Eval ${L} startet"
    docker exec seg_gate bash -c "cd /bevfusion && \
      LOAD_BEV_LATENTS=1 LATENT_LOAD_DIR=/output/dump_22_${L} \
      MASTER_HOST=127.0.0.1 MASTER_PORT=29501 \
      torchpack dist-run -np 1 python tools/test.py ${SEGCFG} ${CKPT} \
      --eval map > /output/eval22_full_${L}.log 2>&1"
    log "Full-Val-Eval ${L} fertig"
done

# --- 3) Dynamik-Quartil-Evals ------------------------------------------------
# minimal = Task-21-Dump (lokal vorhanden) als ungewichtete Basis.
for PAIR in "samp_dyn:dump_22_samp_dyn" "rare_w2:dump_22_rare_w2" "minimal:dump_seg_minimal"; do
    NAME=${PAIR%%:*}; DIR=${PAIR##*:}
    for Q in 1 2 3 4; do
        wait_gpu
        log "Quartil-Eval ${NAME} q${Q} startet"
        docker exec seg_gate bash -c "cd /bevfusion && \
          LOAD_BEV_LATENTS=1 LATENT_LOAD_DIR=/output/${DIR} \
          MASTER_HOST=127.0.0.1 MASTER_PORT=29501 \
          torchpack dist-run -np 1 python tools/test.py ${SEGCFG} ${CKPT} \
          --eval map --cfg-options data.test.ann_file=/output/quartile_pkls/dyn_q${Q}.pkl \
          > /output/eval22_${NAME}_q${Q}.log 2>&1"
        log "Quartil-Eval ${NAME} q${Q} fertig"
    done
done

touch ${OUT}/eval22_DONE
log "ALLE EVALS FERTIG"
