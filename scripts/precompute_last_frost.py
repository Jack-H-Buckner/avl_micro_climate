"""Pre-compute last-frost date grids at sub-weekly resolution using mu/sigma GBMs.

Uses the same approach as plot_spring_frost_free.py but saves results for
many threshold levels so the interactive app can load any threshold instantly.

This takes ~30-60 min on first run. Results are cached to disk.

Usage
-----
    python scripts/precompute_last_frost.py
    python scripts/precompute_last_frost.py --steps-per-week 3  # faster, coarser
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_DIR, COVARIATES_DIR, PROCESSED_DIR, ELEV_VALID_MIN, ELEV_VALID_MAX
from src.postprocessing.frost_climatology import (
    FROST_CLIM_DIR,
    STATIC_COV_NAMES,
    _load_covariates,
    define_frost_weeks,
)


def load_mu_sigma_models(frost_dir=None):
    if frost_dir is None:
        frost_dir = FROST_CLIM_DIR
    with open(Path(frost_dir) / "spatial_model_mu_sigma.pkl", "rb") as f:
        d = pickle.load(f)
    return d["gbm_mu"], d["gbm_sigma"], d["feature_names"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-week", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    steps_per_week = args.steps_per_week

    print("Loading mu/sigma GBMs ...")
    gbm_mu, gbm_sigma, _ = load_mu_sigma_models()

    print("Loading covariates ...")
    covariates = _load_covariates()
    grid_h, grid_w = next(iter(covariates.values())).shape

    valid_mask = np.ones((grid_h, grid_w), dtype=bool)
    for name in STATIC_COV_NAMES:
        if name in covariates:
            valid_mask &= np.isfinite(covariates[name])

    # Elevation validity mask — flag pixels outside the GBM training range
    elev = covariates.get("elevation")
    if elev is not None:
        elev_valid = (elev >= ELEV_VALID_MIN) & (elev <= ELEV_VALID_MAX) & np.isfinite(elev)
    else:
        elev_valid = np.ones((grid_h, grid_w), dtype=bool)

    # Combine: predict only where covariates are finite AND elevation is in range
    predict_mask = valid_mask & elev_valid

    valid_rows, valid_cols = np.where(predict_mask)
    flat_idx = np.ravel_multi_index((valid_rows, valid_cols), (grid_h, grid_w))
    n_valid = len(flat_idx)
    n_elev_excluded = int(valid_mask.sum() - predict_mask.sum())
    print(f"  Valid pixels: {n_valid}  (elevation-excluded: {n_elev_excluded})")

    static_features = np.column_stack([
        covariates[name].ravel()[flat_idx] for name in STATIC_COV_NAMES
    ])

    omega = 2 * np.pi / 52
    fractional_weeks = np.arange(1, 39 + 1.0 / steps_per_week, 1.0 / steps_per_week)
    n_steps = len(fractional_weeks)
    print(f"  Time steps: {n_steps} ({steps_per_week}/week)")

    # ── Pass 1: predict frost prob at each step ──
    print("Predicting frost probability ...")
    all_step_probs = np.zeros((n_steps, n_valid), dtype=np.float32)
    for i, fw in enumerate(fractional_weeks):
        if i % 20 == 0:
            print(f"  Step {i}/{n_steps} ({time.time() - t0:.0f}s)")
        wc = np.full(n_valid, np.cos(omega * fw), dtype=np.float32)
        ws = np.full(n_valid, np.sin(omega * fw), dtype=np.float32)
        X = np.column_stack([static_features, wc, ws])
        mu = gbm_mu.predict(X)
        sigma = np.exp(gbm_sigma.predict(X)).clip(0.01)
        p_week = norm.cdf((0.0 - mu) / sigma).astype(np.float32)
        p_step = 1.0 - np.power(np.clip(1.0 - p_week, 0, 1), 1.0 / steps_per_week)
        all_step_probs[i] = p_step

    peak_idx = np.argmax(all_step_probs, axis=0)

    # ── Pass 2: backward survival ──
    print(f"Backward survival pass ... ({time.time() - t0:.0f}s)")
    survival = np.ones((n_steps, n_valid), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        survival[t] = survival[t + 1] * (1.0 - all_step_probs[t + 1])

    # ── Pass 3: forward cumsum → threshold crossing for many thresholds ──
    thresholds = np.arange(0.01, 1.001, 0.01)  # 1% to 100%
    print(f"Forward cumsum for {len(thresholds)} thresholds ... ({time.time() - t0:.0f}s)")

    # Store result as fractional week for each threshold
    results = {th: np.full(n_valid, np.nan, dtype=np.float32) for th in thresholds}
    cumsum = np.zeros(n_valid, dtype=np.float64)

    for t in range(n_steps):
        after_peak = t >= peak_idx
        lf_prob = all_step_probs[t] * survival[t]
        lf_prob[~after_peak] = 0.0
        cumsum += lf_prob

        for th in thresholds:
            crosses = (cumsum >= th) & np.isnan(results[th])
            results[th][crosses] = fractional_weeks[t]

    # ── Reshape to grids and save ──
    print(f"Reshaping and saving ... ({time.time() - t0:.0f}s)")

    # Stack all threshold results into (n_thresholds, H, W)
    n_th = len(thresholds)
    last_frost_grids = np.full((n_th, grid_h, grid_w), np.nan, dtype=np.float32)
    for i, th in enumerate(thresholds):
        layer = np.full(grid_h * grid_w, np.nan, dtype=np.float32)
        layer[flat_idx] = results[th]
        last_frost_grids[i] = layer.reshape(grid_h, grid_w)

    out_path = FROST_CLIM_DIR / "last_frost_dates.npz"
    np.savez_compressed(
        out_path,
        last_frost_grids=last_frost_grids,
        thresholds=thresholds.astype(np.float32),
        valid_mask=predict_mask,
        elev_valid_mask=elev_valid,
    )
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
