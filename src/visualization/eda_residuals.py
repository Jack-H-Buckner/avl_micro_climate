"""EDA: Scatter plots of ECOSTRESS–gridMET residuals vs predictor variables.

Residual = LST_ecostress − tmmn_gridMET

Uses a random 250k subsample for visualisation. Plots are arranged in a grid
with one panel per predictor, each showing a scatter (with transparency) plus
a LOWESS-like binned mean trend line.

Usage
-----
    python -m src.visualization.eda_residuals
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRAINING_PATH = PROCESSED_DIR / "training" / "ecostress_training_samples.parquet"
N_SAMPLE = 250_000

# Predictors to plot (excludes tmmn since it's the baseline)
PREDICTORS = [
    # Terrain
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi_300m", "tpi_1000m", "curvature", "twi",
    # Meteorological
    "tmmx", "tmmx_prev", "diurnal_range",
    "vs", "sph", "vpd", "srad", "srad_prev", "pr", "rmin",
    # Temporal
    "hours_until_sunrise",
]

LABELS = {
    "elevation": "Elevation (m)",
    "slope": "Slope (°)",
    "aspect_sin": "Aspect sin (N–S)",
    "aspect_cos": "Aspect cos (E–W)",
    "tpi_300m": "TPI 300 m",
    "tpi_1000m": "TPI 1000 m",
    "curvature": "Curvature",
    "twi": "TWI",
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
}


def _binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 40):
    """Compute binned means for a trend line."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins + 1)
    idx = np.digitize(x, bins)
    bin_centers = []
    bin_means = []
    for i in range(1, len(bins)):
        sel = y[idx == i]
        if len(sel) >= 20:
            bin_centers.append((bins[i - 1] + bins[i]) / 2)
            bin_means.append(np.mean(sel))
    return np.array(bin_centers), np.array(bin_means)


def run():
    log.info("Loading training data ...")
    df = pd.read_parquet(TRAINING_PATH)

    # Compute residual
    df["residual"] = df["lst"] - df["tmmn"]

    # Random subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
    sub = df.iloc[idx].copy()
    log.info("Subsampled %d rows (residual mean=%.2f, std=%.2f)",
             len(sub), sub["residual"].mean(), sub["residual"].std())

    # ── Plot ─────────────────────────────────────────────────────────
    n_vars = len(PREDICTORS)
    n_cols = 4
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel()

    for i, var in enumerate(PREDICTORS):
        ax = axes[i]
        x = sub[var].values
        y = sub["residual"].values

        ax.scatter(x, y, s=1, alpha=0.05, color="steelblue", rasterized=True)

        # Binned mean trend
        bx, by = _binned_mean(x, y)
        if len(bx) > 2:
            ax.plot(bx, by, color="firebrick", linewidth=2, label="Binned mean")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel(LABELS.get(var, var), fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel("Residual (LST − Tmin) °C", fontsize=10)
        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)

        # Annotate correlation
        mask = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.text(0.03, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, va="top", color="firebrick")

    # Hide unused axes
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "ECOSTRESS LST − gridMET Tmin residual vs predictors\n"
        f"(n = {len(sub):,} random samples)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "eda_residual_vs_predictors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)

    return out


if __name__ == "__main__":
    run()
