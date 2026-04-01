"""Generate partial dependence plots (PDP) for the trained Random Forest model.

Produces:
  1. A grid of 1-D PDPs for all features, ranked by importance.
  2. A 2-D PDP for wind_speed × tpi_300m (cold-pooling interaction).

Uses a 10k subsample of the training data for efficient PDP computation.

Usage
-----
    python scripts/plot_pdp.py
    python scripts/plot_pdp.py --model data/output/models/rf_benchmark_hrs4.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FIGURES_DIR, PROCESSED_DIR, OUTPUT_DIR
from src.model.cross_validation import FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SPLIT_DIR = PROCESSED_DIR / "training" / "splits"
DEFAULT_MODEL = OUTPUT_DIR / "models" / "rf_benchmark_hrs4.pkl"

LABELS = {
    "elevation": "Elevation (m)",
    "slope": "Slope (°)",
    "aspect_sin": "Aspect sin (N–S)",
    "aspect_cos": "Aspect cos (E–W)",
    "tpi_300m": "TPI 300 m",
    "tpi_1000m": "TPI 1000 m",
    "curvature": "Curvature",
    "twi": "TWI",
    "impervious_pct": "Impervious (%)",
    "tree_canopy_pct": "Tree canopy (%)",
    "sky_view_factor": "Sky view factor",
    "dist_to_water_m": "Dist. to water (m)",
    "is_forest": "Forest",
    "is_developed": "Developed",
    "is_agriculture": "Agriculture",
    "is_water": "Water",
    "ndvi": "NDVI",
    "ecostress_nighttime_median": "ECOSTRESS median (°C)",
    "ecostress_nighttime_q15": "ECOSTRESS Q15 (°C)",
    "ecostress_nighttime_q85": "ECOSTRESS Q85 (°C)",
    "ecostress_nighttime_iqr": "ECOSTRESS IQR (°C)",
    "tmmx": "Tmax (°C)",
    "tmmx_prev": "Tmax prev day (°C)",
    "diurnal_range": "Diurnal range (°C)",
    "vs": "Wind speed (m/s)",
    "sph": "Specific humidity (kg/kg)",
    "vpd": "VPD (kPa)",
    "srad": "Solar rad (W/m²)",
    "srad_prev": "Solar rad prev (W/m²)",
    "pr": "Precipitation (mm)",
    "rmin": "RH min (%)",
    "hours_until_sunrise": "Hours until sunrise",
    "dist_to_removed_m": "Dist. to removed pixel (m)",
    "fraction_scene_removed": "Fraction scene removed",
}


def load_model_and_data(model_path: Path, n_subsample: int = 10_000):
    """Load the saved RF model and a subsample of training data."""
    log.info("Loading model from %s ...", model_path)
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    # Find cached training split
    train_files = sorted(SPLIT_DIR.glob("train_*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No cached training splits in {SPLIT_DIR}")
    train_path = train_files[-1]
    log.info("Loading training data from %s ...", train_path)
    train_df = pd.read_parquet(train_path)

    # Use only features present in both model and data
    features = [c for c in FEATURE_COLS if c in train_df.columns]
    X = train_df[features]

    # Subsample for PDP speed
    if n_subsample < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n_subsample, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
    log.info("PDP subsample: %d rows, %d features", len(X), len(features))

    return rf, X, features


def plot_1d_pdps(rf, X, features, out_path: Path):
    """Plot 1-D PDPs for all features, ordered by importance."""
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]  # descending

    n_features = len(features)
    n_cols = 5
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = axes.ravel()

    for plot_idx, feat_idx in enumerate(order):
        ax = axes[plot_idx]
        feat_name = features[feat_idx]

        # Use the full data range (P0–P100) instead of sklearn's default
        # percentile clipping, so PDP x-axes reflect the actual extent.
        feat_vals = X.iloc[:, feat_idx].dropna().values
        grid = np.linspace(feat_vals.min(), feat_vals.max(), 50)

        pdp_result = partial_dependence(
            rf, X, features=[feat_idx],
            kind="average", percentiles=(0, 1), grid_resolution=50,
        )
        grid_vals = pdp_result["grid_values"][0]
        avg_pred = pdp_result["average"][0]

        pd_range = float(avg_pred.max() - avg_pred.min())

        ax.plot(grid_vals, avg_pred, color="steelblue", linewidth=2)
        # Set y-axis to the individual variable's PD range with small padding
        y_pad = pd_range * 0.15 if pd_range > 0 else 0.01
        ax.set_ylim(avg_pred.min() - y_pad, avg_pred.max() + y_pad)

        ax.set_xlabel(LABELS.get(feat_name, feat_name), fontsize=9)
        if plot_idx % n_cols == 0:
            ax.set_ylabel("Partial dependence\n(LST − Tmin, °C)", fontsize=9)
        ax.set_title(
            f"{feat_name}\n(imp={importances[feat_idx]:.4f})",
            fontsize=9, fontweight="bold",
        )
        ax.tick_params(labelsize=8)

        # Annotate PD range
        ax.text(0.97, 0.95, f"PD range: {pd_range:.3f} °C",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="firebrick", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

        # Add rug plot
        x_vals = X.iloc[:, feat_idx].values
        ax.plot(
            x_vals[::max(1, len(x_vals) // 100)],
            np.full(min(100, len(x_vals)), ax.get_ylim()[0]),
            "|", color="gray", alpha=0.3, markersize=4,
        )

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Partial Dependence Plots — Random Forest (predawn, 250K samples)\n"
        "Features ordered by importance (descending)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 1-D PDPs → %s", out_path)


def plot_2d_pdp(rf, X, features, feat_pair, out_path: Path):
    """Plot a 2-D PDP for a pair of features."""
    idx_a = features.index(feat_pair[0])
    idx_b = features.index(feat_pair[1])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    display = PartialDependenceDisplay.from_estimator(
        rf, X, features=[(idx_a, idx_b)],
        kind="average", grid_resolution=30,
        ax=ax,
    )

    ax.set_xlabel(LABELS.get(feat_pair[0], feat_pair[0]), fontsize=11)
    ax.set_ylabel(LABELS.get(feat_pair[1], feat_pair[1]), fontsize=11)
    ax.set_title(
        f"2-D Partial Dependence: {feat_pair[0]} × {feat_pair[1]}\n"
        "(cold-pooling interaction under calm winds in valleys)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 2-D PDP → %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate PDP plots")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to saved RF model pickle",
    )
    parser.add_argument(
        "--n-subsample", type=int, default=10_000,
        help="Number of training rows for PDP computation (default: 10000)",
    )
    args = parser.parse_args()

    rf, X, features = load_model_and_data(args.model, n_subsample=args.n_subsample)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1-D PDPs for all features
    plot_1d_pdps(rf, X, features, FIGURES_DIR / "pdp_all_features.png")

    # 2-D PDP: wind speed × TPI (cold-pooling interaction)
    if "vs" in features and "tpi_300m" in features:
        plot_2d_pdp(rf, X, features, ("vs", "tpi_300m"),
                    FIGURES_DIR / "pdp_2d_wind_tpi.png")

    # 2-D PDP: elevation × hours_until_sunrise (inversion + cooling curve)
    if "elevation" in features and "hours_until_sunrise" in features:
        plot_2d_pdp(rf, X, features, ("elevation", "hours_until_sunrise"),
                    FIGURES_DIR / "pdp_2d_elevation_hours.png")

    log.info("Done.")


if __name__ == "__main__":
    main()
