"""Build weekly frost probability climatology maps.

End-to-end pipeline:
  1. Generate 2,000 sample points via LHS on static covariates
  2. Predict daily Tmin at all sample points across all frost seasons
  3. Fit harmonic regression, compute weekly P(Tmin < 0°C) per point
  4. Train GBM to interpolate frost probabilities across the full grid
  5. Predict and save 30 weekly frost probability maps at 100m

Usage
-----
    python scripts/build_frost_climatology.py
    python scripts/build_frost_climatology.py --n-samples 2000 --skip-stage1

Note: Before running, ensure gridMET Zarr covers the desired date range.
      Run `python -m src.data.download_gridmet` if you need to extend it.
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_DIR, FIGURES_DIR
from src.postprocessing.frost_climatology import (
    FROST_CLIM_DIR,
    compute_weekly_frost_probs,
    compute_weekly_frost_probs_climatology,
    generate_sample_points,
    plot_frost_panel,
    predict_frost_maps,
    predict_tmin_at_samples,
    save_frost_maps,
    save_metadata,
    train_spatial_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build weekly frost probability climatology maps",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2000,
        help="Number of LHS sample points (default: 2000)",
    )
    parser.add_argument(
        "--model", type=Path,
        default=OUTPUT_DIR / "models" / "rf_benchmark_hrs4.pkl",
        help="Path to trained RF model pickle",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: data/output/frost_climatology)",
    )
    parser.add_argument(
        "--skip-stage1", action="store_true",
        help="Skip daily Tmin prediction (reuse cached parquet)",
    )
    parser.add_argument(
        "--seasonal-anomaly", type=float, default=None,
        help="Seasonal Tmin anomaly (°C) for year-specific forecast. "
             "Omit for climatology (anomaly=0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LHS sampling",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or FROST_CLIM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Stage 1a: Sample points ──────────────────────────────────────────
    sample_path = out_dir / "sample_points.parquet"
    if sample_path.exists() and args.skip_stage1:
        log.info("Loading cached sample points from %s", sample_path)
        sample_df = pd.read_parquet(sample_path)
    else:
        log.info("═══ Stage 1a: Generating %d sample points ═══", args.n_samples)
        sample_df = generate_sample_points(n=args.n_samples, seed=args.seed)
        sample_df["sample_idx"] = range(len(sample_df))
        sample_df.to_parquet(sample_path, index=False)
        log.info("Saved sample points → %s", sample_path)

    # ── Stage 1b: Daily Tmin prediction ──────────────────────────────────
    tmin_path = out_dir / "daily_tmin_samples.parquet"
    if tmin_path.exists() and args.skip_stage1:
        log.info("Loading cached daily Tmin from %s", tmin_path)
        tmin_df = pd.read_parquet(tmin_path)
    else:
        log.info("═══ Stage 1b: Predicting daily Tmin at %d points ═══", len(sample_df))
        tmin_df = predict_tmin_at_samples(sample_df, model_path=args.model)
        # Save with compression; only essential columns to reduce file size
        try:
            tmin_df[["sample_idx", "date", "predicted_tmin_C"]].to_parquet(
                tmin_path, index=False, compression="gzip")
            log.info("Saved daily Tmin → %s (%.1f MB)",
                     tmin_path, tmin_path.stat().st_size / 1e6)
        except OSError as e:
            log.warning("Could not save daily Tmin to disk: %s — continuing in memory", e)

    t_stage1 = time.time()
    log.info("Stage 1 complete in %.1f min", (t_stage1 - t0) / 60)

    # ── Stage 1c: Harmonic regression → frost probabilities ──────────────
    frost_probs_path = out_dir / "weekly_frost_probs.parquet"
    log.info("═══ Stage 1c: Computing weekly frost probabilities ═══")
    frost_probs_df = compute_weekly_frost_probs(tmin_df)
    frost_probs_df.to_parquet(frost_probs_path, index=False)
    log.info("Saved frost probs → %s", frost_probs_path)

    # ── Stage 2a: Train spatial model ────────────────────────────────────
    log.info("═══ Stage 2a: Training spatial interpolation model ═══")
    gbm, cv_metrics, feature_names = train_spatial_model(frost_probs_df, sample_df)

    model_path = out_dir / "spatial_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": gbm, "feature_names": feature_names}, f)
    log.info("Saved GBM → %s", model_path)

    # ── Stage 2b: Full-grid prediction ───────────────────────────────────
    anomaly = args.seasonal_anomaly if args.seasonal_anomaly is not None else 0.0
    log.info("═══ Stage 2b: Predicting frost maps (anomaly=%.2f°C) ═══", anomaly)
    maps = predict_frost_maps(gbm, feature_names, seasonal_anomaly=anomaly)

    # ── Save outputs ─────────────────────────────────────────────────────
    log.info("═══ Saving outputs ═══")
    save_frost_maps(maps, output_dir=out_dir)
    save_metadata(cv_metrics, sample_df, output_dir=out_dir)

    # ── Visualization ────────────────────────────────────────────────────
    log.info("═══ Generating visualization ═══")
    plot_frost_panel(maps)

    t_total = time.time()
    log.info("═══ Pipeline complete in %.1f min ═══", (t_total - t0) / 60)
    log.info("  Output directory: %s", out_dir)
    log.info("  CV R²: %.4f ± %.4f", cv_metrics["cv_r2_mean"], cv_metrics["cv_r2_std"])


if __name__ == "__main__":
    main()
