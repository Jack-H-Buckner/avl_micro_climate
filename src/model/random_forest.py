"""Random Forest benchmark model for ECOSTRESS LST downscaling.

Trains a Random Forest regressor using grouped cross-validation (by scene)
for hyperparameter tuning, then evaluates on the held-out 30 % test scenes.

The model predicts ECOSTRESS LST from terrain covariates, gridMET
meteorological variables, and hours_until_sunrise.

Usage
-----
    python -m src.model.random_forest
    python -m src.model.random_forest --subsample 2000000
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import FIGURES_DIR, PROCESSED_DIR, OUTPUT_DIR
from src.model.cross_validation import (
    FEATURE_COLS,
    TARGET_COL,
    GROUP_COL,
    load_and_split,
    grouped_kfold_indices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = OUTPUT_DIR / "models"


# ── Metrics helper ─────────────────────────────────────────────────────────

def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def _log_metrics(label: str, m: dict):
    log.info("  %s — R²=%.4f  RMSE=%.3f °C  MAE=%.3f °C  Bias=%.3f °C",
             label, m["r2"], m["rmse"], m["mae"], m["bias"])


# ── Cross-validation tuning ───────────────────────────────────────────────

def cross_validate_rf(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    rf_params: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Run grouped K-fold CV and return aggregated + per-fold metrics.

    Parameters
    ----------
    train_df : DataFrame
        Training split (70 % of scenes).
    n_splits : int
        Number of CV folds.
    rf_params : dict
        RandomForestRegressor keyword arguments.

    Returns
    -------
    (mean_metrics, fold_metrics_list)
    """
    if rf_params is None:
        rf_params = {}

    folds = grouped_kfold_indices(train_df, n_splits=n_splits)
    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET_COL].values

    fold_metrics = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X[tr_idx], y[tr_idx])
        y_pred = rf.predict(X[va_idx])
        m = _eval_metrics(y[va_idx], y_pred)
        fold_metrics.append(m)
        _log_metrics(f"Fold {i+1}", m)
        log.info("    (%.1f s)", time.time() - t0)

    # Aggregate
    mean_metrics = {
        k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]
    }
    std_metrics = {
        k: float(np.std([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]
    }
    log.info("  CV mean — R²=%.4f±%.4f  RMSE=%.3f±%.3f  MAE=%.3f±%.3f",
             mean_metrics["r2"], std_metrics["r2"],
             mean_metrics["rmse"], std_metrics["rmse"],
             mean_metrics["mae"], std_metrics["mae"])

    return mean_metrics, fold_metrics


# ── Final model training + test evaluation ─────────────────────────────────

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rf_params: dict | None = None,
) -> tuple[RandomForestRegressor, dict]:
    """Train RF on full training set, evaluate on held-out test scenes.

    Returns
    -------
    (fitted_model, test_metrics)
    """
    if rf_params is None:
        rf_params = {}

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    log.info("Training final RF on %d samples ...", len(X_train))
    t0 = time.time()
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    log.info("  Training done (%.1f s)", time.time() - t0)

    y_pred = rf.predict(X_test)
    test_metrics = _eval_metrics(y_test, y_pred)
    _log_metrics("Test set", test_metrics)

    return rf, test_metrics, y_test, y_pred


# ── Diagnostic plots ──────────────────────────────────────────────────────

def plot_diagnostics(
    rf: RandomForestRegressor,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_metrics: dict,
    test_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Generate diagnostic figure: feature importance, pred-vs-obs, residual histogram, per-scene RMSE."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── 1. Feature importance (color-coded by category) ────────────
    ax = axes[0, 0]
    importances = rf.feature_importances_
    order = np.argsort(importances)

    _TERRAIN = {"elevation", "slope", "aspect_sin", "aspect_cos",
                "tpi_300m", "tpi_1000m", "curvature", "twi"}
    _LAND_SURFACE = {"impervious_pct", "tree_canopy_pct", "sky_view_factor",
                     "dist_to_water_m", "is_forest", "is_developed",
                     "is_agriculture", "is_water", "ndvi"}
    _METEOROLOGICAL = {"tmmx", "tmmx_prev", "diurnal_range", "vs", "sph",
                       "vpd", "srad", "srad_prev", "pr", "rmin"}
    _ECOSTRESS = {"ecostress_nighttime_median", "ecostress_nighttime_q15",
                  "ecostress_nighttime_q85", "ecostress_nighttime_iqr"}
    _TEMPORAL = {"hours_until_sunrise"}
    _CLOUD_FILTER = {"dist_to_removed_m", "fraction_scene_removed"}

    _CAT_COLORS = {
        "Terrain": "#2166ac",
        "Land surface": "#4dac26",
        "Meteorological": "#d6604d",
        "ECOSTRESS composites": "#b2abd2",
        "Temporal": "#fdb863",
        "Cloud filter": "#8c510a",
    }

    def _feat_category(name):
        if name in _TERRAIN: return "Terrain"
        if name in _LAND_SURFACE: return "Land surface"
        if name in _METEOROLOGICAL: return "Meteorological"
        if name in _ECOSTRESS: return "ECOSTRESS composites"
        if name in _TEMPORAL: return "Temporal"
        if name in _CLOUD_FILTER: return "Cloud filter"
        return "Other"

    bar_colors = [_CAT_COLORS.get(_feat_category(FEATURE_COLS[i]), "gray") for i in order]
    ax.barh(np.array(FEATURE_COLS)[order], importances[order], color=bar_colors)
    ax.set_xlabel("Importance (impurity decrease)")
    ax.set_title("Feature importance", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=cat) for cat, c in _CAT_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    # ── 2. Predicted vs observed ─────────────────────────────────────
    ax = axes[0, 1]
    # Subsample for plotting
    rng = np.random.default_rng(42)
    n_plot = min(100_000, len(y_test))
    idx = rng.choice(len(y_test), size=n_plot, replace=False)
    ax.scatter(y_test[idx], y_pred[idx], s=1, alpha=0.05, color="steelblue", rasterized=True)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Observed LST residual (LST − Tmin, °C)")
    ax.set_ylabel("Predicted LST residual (°C)")
    ax.set_title(
        f"Predicted vs observed (test)\n"
        f"R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.2f} °C",
        fontweight="bold",
    )

    # ── 3. Residual histogram ────────────────────────────────────────
    ax = axes[1, 0]
    residuals = y_pred - y_test
    ax.hist(residuals, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Residual (pred − obs) °C")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residual distribution\nBias={test_metrics['bias']:.3f} °C  MAE={test_metrics['mae']:.2f} °C",
        fontweight="bold",
    )

    # ── 4. Per-scene RMSE ────────────────────────────────────────────
    ax = axes[1, 1]
    test_df = test_df.copy()
    test_df["residual"] = y_pred - y_test
    scene_rmse = test_df.groupby(GROUP_COL)["residual"].apply(
        lambda r: np.sqrt((r ** 2).mean())
    ).sort_values()
    ax.barh(range(len(scene_rmse)), scene_rmse.values, color="steelblue", height=0.8)
    ax.set_xlabel("RMSE (°C)")
    ax.set_ylabel("Test scene index")
    ax.set_title(
        f"Per-scene RMSE (n={len(scene_rmse)} test scenes)\n"
        f"Median={scene_rmse.median():.2f} °C",
        fontweight="bold",
    )

    fig.suptitle("Random Forest benchmark — test set diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or (FIGURES_DIR / "rf_benchmark_diagnostics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved diagnostics → %s", out)
    return out


# ── Main pipeline ──────────────────────────────────────────────────────────

def run(subsample: int = 2_000_000, n_cv_folds: int = 5, max_hours: float | None = None):
    """Full RF benchmark pipeline: split → CV → train → evaluate → plot.

    Parameters
    ----------
    subsample : int
        Number of training rows to use (subsampled after grouped split).
        Keeps RF fitting tractable on a laptop.
    n_cv_folds : int
        Number of grouped CV folds for tuning diagnostics.
    max_hours : float or None
        If set, restrict to scenes with hours_until_sunrise ≤ this value.
    """
    # ── 1. Grouped 70/30 split ───────────────────────────────────────
    log.info("══ Step 1: Grouped train/test split ══")
    train_df, test_df = load_and_split(
        test_size=0.30, subsample=subsample, max_hours_until_sunrise=max_hours,
    )

    # ── 2. RF hyperparameters ────────────────────────────────────────
    rf_params = dict(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=20,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )
    log.info("RF params: %s", rf_params)

    # ── 3. Grouped K-fold CV on training split ───────────────────────
    log.info("══ Step 2: %d-fold grouped CV ══", n_cv_folds)
    cv_mean, cv_folds = cross_validate_rf(train_df, n_splits=n_cv_folds, rf_params=rf_params)

    # ── 4. Train final model on full training split, evaluate on test ─
    log.info("══ Step 3: Final model on full training split ══")
    rf, test_metrics, y_test, y_pred = train_and_evaluate(train_df, test_df, rf_params=rf_params)

    # ── 5. Save model ────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    hrs_tag = f"_hrs{max_hours:.0f}" if max_hours is not None else ""
    model_path = MODEL_DIR / f"rf_benchmark{hrs_tag}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    log.info("Saved model → %s", model_path)

    # ── 6. Diagnostic plots ──────────────────────────────────────────
    log.info("══ Step 4: Diagnostic plots ══")
    fig_path = FIGURES_DIR / f"rf_benchmark_diagnostics{hrs_tag}.png"
    plot_diagnostics(rf, y_test, y_pred, test_metrics, test_df, out_path=fig_path)

    # ── 7. Summary ───────────────────────────────────────────────────
    log.info("══ Summary ══")
    log.info("  CV R²:   %.4f ± %.4f", cv_mean["r2"],
             float(np.std([f["r2"] for f in cv_folds])))
    log.info("  CV RMSE: %.3f ± %.3f °C", cv_mean["rmse"],
             float(np.std([f["rmse"] for f in cv_folds])))
    log.info("  Test R²:   %.4f", test_metrics["r2"])
    log.info("  Test RMSE: %.3f °C", test_metrics["rmse"])
    log.info("  Test MAE:  %.3f °C", test_metrics["mae"])
    log.info("  Test Bias: %.3f °C", test_metrics["bias"])

    return rf, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Random Forest benchmark model")
    parser.add_argument(
        "--subsample", type=int, default=2_000_000,
        help="Training subsample size (default: 2,000,000)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of grouped CV folds (default: 5)",
    )
    parser.add_argument(
        "--max-hours", type=float, default=None,
        help="Max hours_until_sunrise filter (e.g. 4.0 for predawn only)",
    )
    args = parser.parse_args()
    run(subsample=args.subsample, n_cv_folds=args.cv_folds, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
