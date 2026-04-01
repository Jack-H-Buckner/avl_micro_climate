"""Plot variable importance for the GBM spatial model (mu and sigma).

Uses permutation importance on a held-out subsample, scoring by
neg_mean_squared_error to avoid misleading R² values at very high
baseline accuracy.
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FROST_CLIM_DIR = Path("data/output/frost_climatology")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def reconstruct_training_data(features):
    """Rebuild X, y_mu from saved parquet files."""
    sample_df = pd.read_parquet(FROST_CLIM_DIR / "sample_points.parquet")
    frost_probs = pd.read_parquet(FROST_CLIM_DIR / "weekly_frost_probs.parquet")

    merged = frost_probs.merge(sample_df, on="sample_idx", how="left")

    omega = 2 * np.pi / 52
    merged["week_cos"] = np.cos(omega * merged["week_num"])
    merged["week_sin"] = np.sin(omega * merged["week_num"])

    X = merged[features].values

    # mu target: logit(frost_prob)
    probs = merged["frost_prob"].values.clip(0.001, 0.999)
    y_mu = np.log(probs / (1 - probs))

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y_mu)
    return X[valid], y_mu[valid]


def main():
    # Load model
    with open(FROST_CLIM_DIR / "spatial_model_mu_sigma.pkl", "rb") as f:
        obj = pickle.load(f)

    features = obj["feature_names"]
    gbm_mu = obj["gbm_mu"]

    print(f"Features ({len(features)}): {features}")

    # Reconstruct training data
    print("Reconstructing training data ...")
    X, y_mu = reconstruct_training_data(features)
    print(f"  Total rows: {len(y_mu)}")

    # Subsample — 50k gives stable estimates
    rng = np.random.default_rng(42)
    n_sub = min(50_000, len(y_mu))
    idx = rng.choice(len(y_mu), size=n_sub, replace=False)
    X_sub = X[idx]
    y_sub = y_mu[idx]

    # Baseline MSE
    y_pred = gbm_mu.predict(X_sub)
    baseline_mse = mean_squared_error(y_sub, y_pred)
    print(f"  Baseline MSE: {baseline_mse:.4f}")

    # Permutation importance scored by neg_mean_squared_error
    print("Computing permutation importance (neg_MSE, 50k samples, 10 repeats) ...")
    perm = permutation_importance(
        gbm_mu, X_sub, y_sub,
        scoring="neg_mean_squared_error",
        n_repeats=10, random_state=42, n_jobs=-1,
    )

    # importances_mean is the decrease in neg_MSE (i.e., increase in MSE)
    # A positive value means shuffling the feature increases MSE = feature is important
    imp_mean = -perm.importances_mean  # flip sign: positive = MSE increase
    imp_std = perm.importances_std

    # Sort descending by importance
    sorted_idx = np.argsort(imp_mean)[::-1]

    # Pretty feature names
    pretty = {
        "elevation": "Elevation",
        "slope": "Slope",
        "aspect_sin": "Aspect (sin)",
        "aspect_cos": "Aspect (cos)",
        "tpi_300m": "TPI 300m",
        "tpi_1000m": "TPI 1000m",
        "curvature": "Curvature",
        "twi": "TWI",
        "impervious_pct": "Impervious %",
        "tree_canopy_pct": "Tree canopy %",
        "sky_view_factor": "Sky view factor",
        "dist_to_water_m": "Dist to water",
        "is_forest": "Forest",
        "is_developed": "Developed",
        "is_agriculture": "Agriculture",
        "is_water": "Water",
        "week_cos": "Week (cos)",
        "week_sin": "Week (sin)",
        "seasonal_anomaly_C": "Seasonal anomaly",
    }

    names = [pretty.get(features[i], features[i]) for i in sorted_idx]
    means = imp_mean[sorted_idx]
    stds = imp_std[sorted_idx]

    # Print table
    print(f"\nPermutation importance (MSE increase when feature shuffled):")
    print(f"{'Feature':25s} {'Mean MSE Δ':>12s} {'Std':>10s}")
    print("-" * 49)
    for n, m, s in zip(names, means, stds):
        print(f"{n:25s} {m:12.4f} {s:10.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))

    # Color by category
    terrain_pretty = {pretty.get(n, n) for n in [
        "elevation", "slope", "aspect_sin", "aspect_cos",
        "tpi_300m", "tpi_1000m", "curvature", "twi"
    ]}
    land_pretty = {pretty.get(n, n) for n in [
        "impervious_pct", "tree_canopy_pct", "sky_view_factor",
        "dist_to_water_m", "is_forest", "is_developed", "is_agriculture", "is_water"
    ]}
    colors = []
    for n in names:
        if n in terrain_pretty:
            colors.append("#2196F3")
        elif n in land_pretty:
            colors.append("#4CAF50")
        else:
            colors.append("#FF9800")

    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="white",
            height=0.7, capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("MSE increase when feature shuffled", fontsize=11)
    ax.set_title("GBM Variable Importance — Frost Probability (mu)", fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Terrain"),
        Patch(facecolor="#4CAF50", label="Land surface"),
        Patch(facecolor="#FF9800", label="Temporal"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    out_path = FIGURES_DIR / "gbm_variable_importance.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")

    # ── Spatial-only panel (drop week_cos / week_sin) ────────────────────
    spatial_mask = np.array([f not in ("week_cos", "week_sin") for f in features])
    spatial_idx_all = np.where(spatial_mask)[0]
    spatial_imp = imp_mean[spatial_idx_all]
    spatial_std = imp_std[spatial_idx_all]
    spatial_order = np.argsort(spatial_imp)[::-1]

    sp_names = [pretty.get(features[spatial_idx_all[i]], features[spatial_idx_all[i]])
                for i in spatial_order]
    sp_means = spatial_imp[spatial_order]
    sp_stds = spatial_std[spatial_order]

    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    y_pos2 = np.arange(len(sp_names))
    colors2 = []
    for n in sp_names:
        if n in terrain_pretty:
            colors2.append("#2196F3")
        elif n in land_pretty:
            colors2.append("#4CAF50")
        else:
            colors2.append("#FF9800")

    ax2.barh(y_pos2, sp_means, xerr=sp_stds, color=colors2, edgecolor="white",
             height=0.7, capsize=2)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(sp_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("MSE increase when feature shuffled", fontsize=11)
    ax2.set_title("GBM Variable Importance — Spatial Covariates Only", fontsize=13)
    ax2.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    ax2.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    out2 = FIGURES_DIR / "gbm_variable_importance_spatial.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved to {out2}")


if __name__ == "__main__":
    main()
