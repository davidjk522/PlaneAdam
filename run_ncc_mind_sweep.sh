#!/bin/bash
# Sequential overnight sweep: only-NCC, only-MIND-SSC, both — on GPU 1.
# Runs one after another (not in parallel) so they share the GPU cleanly.
set -e  # stop the whole sweep if one run errors out, rather than silently continuing on garbage

GPU=1
CONFIG=OASIS_config.json
SCRIPT=PlaneAdam/convex_optimization/convex_run_paired_dino_slice2d_512.py
PCA=256
ADAM=80
LOGDIR=logs/ncc_mind_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

echo "=== [1/3] NCC only ==="
python "$SCRIPT" "$GPU" "$CONFIG" --pca "$PCA" --adam "$ADAM" \
    --lambda-ncc 1.0 \
    2>&1 | tee "$LOGDIR/ncc_only.log"

echo "=== [2/3] MIND-SSC only ==="
python "$SCRIPT" "$GPU" "$CONFIG" --pca "$PCA" --adam "$ADAM" \
    --lambda-mind 1.0 \
    2>&1 | tee "$LOGDIR/mind_only.log"

echo "=== [3/3] NCC + MIND-SSC ==="
python "$SCRIPT" "$GPU" "$CONFIG" --pca "$PCA" --adam "$ADAM" \
    --lambda-ncc 1.0 --lambda-mind 1.0 \
    2>&1 | tee "$LOGDIR/ncc_and_mind.log"

echo "=== Sweep complete. Logs in $LOGDIR ==="
