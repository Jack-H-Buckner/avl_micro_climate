"""Plot the last frost date at each pixel using continuous sub-weekly resolution.

Loads the fitted mu/sigma GBMs to predict frost probability at fine temporal
steps (0.1-week increments), then computes:
  P(last frost in step t) = p_t * prod(1 - p_k for k > t)
and finds the date by which cumulative last-frost probability reaches
each confidence level.

Usage
-----
    python scripts/plot_spring_frost_free.py
    python scripts/plot_spring_frost_free.py --thresholds 0.70 0.80 0.90 0.95
"""

import argparse
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FIGURES_DIR, OUTPUT_DIR
from src.postprocessing.frost_climatology import (
    FROST_CLIM_DIR,
    STATIC_COV_NAMES,
    define_frost_weeks,
    _load_covariates,
    _get_reference_profile,
)


def load_mu_sigma_models(frost_dir=None):
    """Load the fitted mu/sigma GBM models."""
    if frost_dir is None:
        frost_dir = FROST_CLIM_DIR
    with open(Path(frost_dir) / "spatial_model_mu_sigma.pkl", "rb") as f:
        d = pickle.load(f)
    return d["gbm_mu"], d["gbm_sigma"], d["feature_names"]


def compute_last_frost_dates(
    gbm_mu, gbm_sigma, static_features, valid_mask, thresholds,
    steps_per_week=5,
):
    """Compute last-frost dates at sub-weekly resolution.

    Memory-efficient: predicts frost probability one step at a time,
    does two passes (backward for survival, forward for cumulative
    last-frost), storing only per-step probabilities as 1D arrays
    over valid pixels.

    Parameters
    ----------
    gbm_mu, gbm_sigma : fitted models.
    static_features : (n_valid, 16) covariate values at valid pixels.
    valid_mask : (H, W) boolean.
    thresholds : list of cumulative confidence levels.
    steps_per_week : temporal resolution.

    Returns
    -------
    dict mapping threshold -> (H,W) array of fractional week numbers.
    fractional_weeks : 1D array of time steps used.
    """
    n_valid = static_features.shape[0]
    omega = 2 * np.pi / 52

    fractional_weeks = np.arange(1, 39 + 1.0 / steps_per_week, 1.0 / steps_per_week)
    n_steps = len(fractional_weeks)

    print(f"  Predicting frost prob at {n_steps} time steps ...")

    # Pass 1: predict all step probabilities (store as n_steps x n_valid)
    # Convert weekly prob to per-step prob for discrete time steps
    all_step_probs = np.zeros((n_steps, n_valid), dtype=np.float32)
    for i, fw in enumerate(fractional_weeks):
        wc = np.full(n_valid, np.cos(omega * fw), dtype=np.float32)
        ws = np.full(n_valid, np.sin(omega * fw), dtype=np.float32)
        X = np.column_stack([static_features, wc, ws])

        mu = gbm_mu.predict(X)
        sigma = np.exp(gbm_sigma.predict(X)).clip(0.01)
        p_week = norm.cdf((0.0 - mu) / sigma).astype(np.float32)

        # Convert weekly prob to per-step prob
        p_step = 1.0 - np.power(np.clip(1.0 - p_week, 0, 1), 1.0 / steps_per_week)
        all_step_probs[i] = p_step

    # Find per-pixel peak step
    peak_idx = np.argmax(all_step_probs, axis=0)

    # Pass 2: backward survival
    print("  Computing survival probabilities (backward pass) ...")
    survival = np.ones((n_steps, n_valid), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        survival[t] = survival[t + 1] * (1.0 - all_step_probs[t + 1])

    # Pass 3: forward cumulative last-frost
    print("  Computing last-frost dates (forward pass) ...")
    grid_h, grid_w = valid_mask.shape
    flat_idx = np.where(valid_mask.ravel())[0]

    results = {}
    for thresh in thresholds:
        results[thresh] = np.full(n_valid, np.nan, dtype=np.float32)

    cumsum = np.zeros(n_valid, dtype=np.float64)
    step_indices = np.arange(n_steps)

    for t in range(n_steps):
        after_peak = t >= peak_idx
        lf_prob = all_step_probs[t] * survival[t]
        lf_prob[~after_peak] = 0.0
        cumsum += lf_prob

        for thresh in thresholds:
            crosses = (cumsum >= thresh) & np.isnan(results[thresh])
            results[thresh][crosses] = fractional_weeks[t]

    # Reshape to grid
    grid_results = {}
    for thresh in thresholds:
        grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        grid.ravel()[flat_idx] = results[thresh]
        grid_results[thresh] = grid

    return grid_results, fractional_weeks


def fractional_week_to_date_label(fw):
    """Convert a fractional frost-season week to a date string."""
    weeks = define_frost_weeks()
    # Interpolate DOY from week number
    wn_floor = int(np.floor(fw))
    frac = fw - wn_floor
    wn_floor = max(1, min(wn_floor, len(weeks)))
    wn_ceil = min(wn_floor + 1, len(weeks))

    doy_floor = weeks[wn_floor - 1]["center_doy"]
    doy_ceil = weeks[wn_ceil - 1]["center_doy"]
    # Handle year wrap
    if doy_ceil < doy_floor:
        doy_ceil += 365
    doy = doy_floor + frac * (doy_ceil - doy_floor)
    if doy > 365:
        doy -= 365

    approx = date(2024, 1, 1) + timedelta(days=int(doy) - 1)
    return approx.strftime("%b %d")


def week_num_to_date_label(week_num):
    """Convert frost-season week number to an approximate date string."""
    weeks = define_frost_weeks()
    wk = next(w for w in weeks if w["week_num"] == week_num)
    doy = wk["center_doy"]
    approx = date(2024, 1, 1) + timedelta(days=doy - 1)
    return approx.strftime("%b %d")


def plot_frost_free_panels(grid_results, valid_mask, out_path=None):
    """Create a multi-panel figure showing last frost date at each pixel.

    Parameters
    ----------
    grid_results : dict mapping threshold -> (H,W) array of fractional weeks.
    valid_mask : (H, W) boolean.
    out_path : output path.
    """
    thresholds = sorted(grid_results.keys())
    n = len(thresholds)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    profile = _get_reference_profile()
    transform = profile["transform"]
    h, w = valid_mask.shape
    west = transform.c
    north = transform.f
    east = west + w * transform.a
    south = north + h * transform.e
    extent_km = [west / 1000, east / 1000, south / 1000, north / 1000]

    cmap = plt.cm.viridis

    for i, thresh in enumerate(thresholds):
        ax = axes[i]
        frost_free = grid_results[thresh]

        # Per-panel color scale
        valid = frost_free[np.isfinite(frost_free)]
        if len(valid) > 0:
            panel_vmin = float(np.percentile(valid, 2))
            panel_vmax = float(np.percentile(valid, 98))
            if panel_vmax - panel_vmin < 0.5:
                panel_vmin -= 0.5
                panel_vmax += 0.5
        else:
            panel_vmin, panel_vmax = 31, 39

        panel_norm = mcolors.Normalize(vmin=panel_vmin, vmax=panel_vmax)

        im = ax.imshow(
            frost_free, cmap=cmap, norm=panel_norm,
            extent=extent_km, interpolation="nearest",
        )
        ax.set_title(
            f"{thresh:.0%} confidence last frost has passed",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Easting (km)", fontsize=10)
        ax.set_ylabel("Northing (km)", fontsize=10)

        # Per-panel colorbar with date labels
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        n_ticks = 7
        tick_vals = np.linspace(panel_vmin, panel_vmax, n_ticks)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([fractional_week_to_date_label(fw) for fw in tick_vals])
        cbar.ax.tick_params(labelsize=8)

        # Stats
        if len(valid) > 0:
            median_fw = float(np.nanmedian(valid))
            median_label = fractional_week_to_date_label(median_fw)
            never_pct = 100 * np.sum(np.isnan(frost_free) & valid_mask) / valid_mask.sum()
            ax.text(
                0.02, 0.02,
                f"Median: {median_label}\nNot reached: {never_pct:.0f}% of pixels",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Last frost date — Asheville region (100m)",
        fontsize=15, fontweight="bold",
    )

    if out_path is None:
        out_path = FIGURES_DIR / "spring_frost_free_panels.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot last frost date maps using continuous sub-weekly resolution",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.70, 0.80, 0.90, 0.95],
        help="Cumulative last-frost confidence levels (default: 0.70 0.80 0.90 0.95)",
    )
    parser.add_argument(
        "--frost-dir", type=Path, default=None,
        help="Directory with spatial_model_mu_sigma.pkl",
    )
    parser.add_argument(
        "--steps-per-week", type=int, default=5,
        help="Sub-weekly resolution (default: 5 = 0.2-week steps)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output figure path",
    )
    args = parser.parse_args()

    frost_dir = args.frost_dir or FROST_CLIM_DIR

    # Load models
    print("Loading mu/sigma GBMs ...")
    gbm_mu, gbm_sigma, feature_names = load_mu_sigma_models(frost_dir)

    # Load covariates and build valid pixel features
    print("Loading covariates ...")
    covariates = _load_covariates()
    grid_h, grid_w = next(iter(covariates.values())).shape

    valid_mask = np.ones((grid_h, grid_w), dtype=bool)
    for name in STATIC_COV_NAMES:
        if name in covariates:
            valid_mask &= np.isfinite(covariates[name])

    valid_rows, valid_cols = np.where(valid_mask)
    flat_idx = np.ravel_multi_index((valid_rows, valid_cols), (grid_h, grid_w))
    static_features = np.column_stack([
        covariates[name].ravel()[flat_idx] for name in STATIC_COV_NAMES
    ])
    print(f"  Valid pixels: {len(valid_rows)}")

    # Compute last-frost dates at sub-weekly resolution
    print(f"Computing last-frost dates ({args.steps_per_week} steps/week) ...")
    grid_results, fractional_weeks = compute_last_frost_dates(
        gbm_mu, gbm_sigma, static_features, valid_mask,
        args.thresholds, steps_per_week=args.steps_per_week,
    )
    print(f"  {len(fractional_weeks)} time steps")

    # Plot
    print("Generating panels ...")
    plot_frost_free_panels(grid_results, valid_mask, args.output)


if __name__ == "__main__":
    main()
