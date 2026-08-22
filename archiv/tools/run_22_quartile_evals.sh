#!/bin/bash
# Task 22 Phase 3 RERUN: Dynamik-Quartil-Evals (nach PKL-Pfad-Fix).
set -uo pipefail
OUT=/home/vima/det_latents_out
SEGCFG=configs/nuscenes/seg/fusion-bev256d2-lss.yaml
CKPT=pretrained/bevfusion-seg.pth
log() { echo "[$(date +%H:%M:%S)] $*" >> ${OUT}/eval22_orchestrator.log; }
wait_gpu() { until ! docker exec seg_gate pgrep -f tools/test.py >/dev/null 2>&1; do sleep 60; done; }
for PAIR in "samp_dyn:dump_22_samp_dyn" "rare_w2:dump_22_rare_w2" "minimal:dump_seg_minimal"; do
    NAME=${PAIR%%:*}; DIR=${PAIR##*:}
    for Q in 1 2 3 4; do
        wait_gpu
        log "RERUN Quartil-Eval ${NAME} q${Q} startet"
        docker exec seg_gate bash -c "cd /bevfusion && \
          LOAD_BEV_LATENTS=1 LATENT_LOAD_DIR=/output/${DIR} \
          MASTER_HOST=127.0.0.1 MASTER_PORT=29501 \
          torchpack dist-run -np 1 python tools/test.py ${SEGCFG} ${CKPT} \
          --eval map --cfg-options data.test.ann_file=/output/quartile_pkls/dyn_q${Q}.pkl \
          > /output/eval22_${NAME}_q${Q}.log 2>&1"
        grep -qa "iou@max" ${OUT}/eval22_${NAME}_q${Q}.log && log "RERUN ${NAME} q${Q} OK" || log "RERUN ${NAME} q${Q} OHNE ERGEBNIS (pruefen!)"
    done
done
touch ${OUT}/eval22_DONE
log "RERUN ALLE QUARTIL-EVALS FERTIG"
