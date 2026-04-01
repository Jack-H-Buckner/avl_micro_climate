#!/usr/bin/env bash
# Queue: wait for SVF + HLS + existing model runs to finish, then run
# the full downstream pipeline with climatology-filtered ECOSTRESS data
# and new covariates (NLCD, SVF, NDVI).
#
# Usage:  bash scripts/run_after_covariates.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

SVF_PID=31843
HLS_PID=31846
# Prior-session processes that may still be running
OLD_PREPARE_PID=30719
OLD_MODEL_PID=30187

LOG="scripts/run_after_covariates.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

wait_for() {
    local name=$1 pid=$2
    if kill -0 "$pid" 2>/dev/null; then
        echo "$(date)  $name (PID $pid) running — waiting..."
        while kill -0 "$pid" 2>/dev/null; do sleep 10; done
    fi
    echo "$(date)  $name done."
}

echo "$(date)  Waiting for background tasks..."

# ── Wait for new covariate tasks ──────────────────────────────────────
wait_for "SVF computation" "$SVF_PID"
wait_for "HLS NDVI download" "$HLS_PID"

# ── Wait for prior-session tasks to avoid file conflicts ──────────────
wait_for "Prior prepare_training" "$OLD_PREPARE_PID"
wait_for "Prior random_forest" "$OLD_MODEL_PID"

echo ""
echo "========================================"
echo "$(date)  All prerequisite tasks complete."
echo "========================================"

# ── Step 1: NDVI compositing ─────────────────────────────────────────
echo ""
echo "$(date)  [1/5] Running NDVI compositing..."
python src/preprocessing/ndvi_composites.py
echo "$(date)  NDVI compositing done."

# ── Step 2: Re-run ECOSTRESS climatology + Tukey filter ──────────────
echo ""
echo "$(date)  [2/5] Re-running ECOSTRESS climatology filter..."
rm -f data/processed/satellite/ecostress_filtered/*.tif
rm -f data/processed/satellite/ecostress_filtered/tukey_filter_diagnostics.parquet
python -m src.data.filter_ecostress
echo "$(date)  ECOSTRESS filtering done."

# ── Step 3: Re-align grids (filtered scenes → 100 m) ────────────────
echo ""
echo "$(date)  [3/5] Re-aligning ECOSTRESS scenes to 100 m grid..."
rm -f data/processed/satellite/ecostress_aligned/*.tif
python -m src.preprocessing.align_grids
echo "$(date)  Grid alignment done."

# ── Step 4: Build training data with new covariates ──────────────────
echo ""
echo "$(date)  [4/5] Building training dataset (with NLCD + SVF + NDVI)..."
rm -rf data/processed/training/splits
rm -f data/processed/training/ecostress_training_samples.parquet
python -m src.preprocessing.prepare_training
echo "$(date)  Training data built."

# ── Step 5: Fit model with expanded features ─────────────────────────
echo ""
echo "$(date)  [5/5] Running random forest model (predawn only, ≤4 hrs to sunrise)..."
python -m src.model.random_forest --max-hours 4.0
echo "$(date)  Modeling complete."

echo ""
echo "========================================"
echo "$(date)  Full pipeline finished successfully."
echo "========================================"
