"""Grouped train/test splitting and cross-validation utilities.

All splits are grouped by ECOSTRESS scene — no pixels from the same scene
appear in both train and test.  This prevents spatial autocorrelation within
a scene from leaking into evaluation metrics.

Usage
-----
    from src.model.cross_validation import load_and_split, grouped_kfold
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRAINING_PATH = PROCESSED_DIR / "training" / "ecostress_training_samples.parquet"
SPLIT_DIR = PROCESSED_DIR / "training" / "splits"

# Predictor columns (response = "lst_residual", i.e. LST − gridMET Tmin)
FEATURE_COLS = [
    # Terrain
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi_300m", "tpi_1000m", "curvature", "twi",
    # Land surface covariates (NLCD + SVF, static)
    "impervious_pct", "tree_canopy_pct", "sky_view_factor",
    "dist_to_water_m", "is_forest", "is_developed",
    "is_agriculture", "is_water",
    # Vegetation state (seasonally varying)
    "ndvi",
    # ECOSTRESS composites (static spatial layers)
    "ecostress_nighttime_median", "ecostress_nighttime_q15",
    "ecostress_nighttime_q85", "ecostress_nighttime_iqr",
    # Meteorological (tmmn excluded — used to compute the residual target)
    "tmmx", "tmmx_prev", "diurnal_range",
    "vs", "sph", "vpd", "srad", "srad_prev", "pr", "rmin",
    # Temporal
    "hours_until_sunrise",
    # Cloud filter proximity (per-pixel distance + scene-level fraction)
    "dist_to_removed_m", "fraction_scene_removed",
]

TARGET_COL = "lst_residual"
GROUP_COL = "scene_id"


def available_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of FEATURE_COLS present in df, warning about missing ones."""
    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (will be skipped): %s", missing)
    return present


def _split_cache_paths(subsample: int | None, max_hrs: float | None = None) -> tuple[Path, Path]:
    """Return (train_path, test_path) for a given subsample size and hour filter."""
    sub_tag = f"_{subsample}" if subsample else "_full"
    hrs_tag = f"_hrs{max_hrs:.0f}" if max_hrs is not None else ""
    return (
        SPLIT_DIR / f"train{sub_tag}{hrs_tag}.parquet",
        SPLIT_DIR / f"test{hrs_tag}.parquet",
    )


def load_and_split(
    test_size: float = 0.30,
    random_state: int = 42,
    subsample: int | None = None,
    max_hours_until_sunrise: float | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data and perform a grouped 70/30 train/test split.

    Caches the split to ``data/processed/training/splits/`` so subsequent
    runs skip the expensive 45M-row parquet load.

    Parameters
    ----------
    test_size : float
        Fraction of *scenes* held out for final validation.
    random_state : int
        Random seed for reproducibility.
    subsample : int or None
        If set, randomly subsample this many rows from the training split
        (after the grouped split) to reduce memory/compute.  The test split
        is never subsampled.
    max_hours_until_sunrise : float or None
        If set, keep only scenes where hours_until_sunrise ≤ this value
        (i.e. close to sunrise / predawn).  Applied before the grouped split
        so both train and test contain only predawn scenes.
    use_cache : bool
        If True (default), load from cached split files when available.

    Returns
    -------
    (train_df, test_df) — full DataFrames with all columns preserved.
    """
    train_path, test_path = _split_cache_paths(subsample, max_hours_until_sunrise)

    # ── Try cache first ──────────────────────────────────────────────
    if use_cache and train_path.exists() and test_path.exists():
        log.info("Loading cached split from %s", SPLIT_DIR)
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        log.info("  Train: %d samples, %d scenes (cached)",
                 len(train_df), train_df[GROUP_COL].nunique())
        log.info("  Test:  %d samples, %d scenes (cached)",
                 len(test_df), test_df[GROUP_COL].nunique())
        return train_df, test_df

    # ── Build split from scratch ─────────────────────────────────────
    log.info("Loading training data from %s ...", TRAINING_PATH)
    # Only load columns needed for modelling to reduce memory on large files
    needed_cols = list(set(
        FEATURE_COLS + [TARGET_COL, GROUP_COL, "hours_until_sunrise", "lst", "date",
                        "pixel_row", "pixel_col"]
    ))
    df = pd.read_parquet(TRAINING_PATH, columns=needed_cols)
    log.info("  %d samples, %d scenes", len(df), df[GROUP_COL].nunique())

    # ── Drop rows with NaN target ──────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log.info("  Dropped %d rows with NaN %s (%d → %d)", n_dropped, TARGET_COL, n_before, len(df))

    # ── Filter by hours_until_sunrise ────────────────────────────────
    if max_hours_until_sunrise is not None:
        before = len(df)
        df = df[df["hours_until_sunrise"] <= max_hours_until_sunrise].reset_index(drop=True)
        log.info("  Filtered to hours_until_sunrise ≤ %.1f: %d → %d samples (%d scenes)",
                 max_hours_until_sunrise, before, len(df), df[GROUP_COL].nunique())

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[GROUP_COL]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    log.info("  Train: %d samples, %d scenes", len(train_df), train_df[GROUP_COL].nunique())
    log.info("  Test:  %d samples, %d scenes", len(test_df), test_df[GROUP_COL].nunique())

    if subsample is not None and subsample < len(train_df):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(train_df), size=subsample, replace=False)
        train_df = train_df.iloc[idx].reset_index(drop=True)
        log.info("  Train subsampled to %d rows", len(train_df))

    # ── Cache to disk ────────────────────────────────────────────────
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    log.info("  Cached splits → %s (%.1f MB train, %.1f MB test)",
             SPLIT_DIR,
             train_path.stat().st_size / 1e6,
             test_path.stat().st_size / 1e6)

    return train_df, test_df


def grouped_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return grouped K-fold split indices (by scene).

    Parameters
    ----------
    df : DataFrame
        Must contain GROUP_COL column.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    List of (train_indices, val_indices) tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[GROUP_COL]
    folds = list(gkf.split(df, groups=groups))
    for i, (tr, va) in enumerate(folds):
        n_tr_scenes = df.iloc[tr][GROUP_COL].nunique()
        n_va_scenes = df.iloc[va][GROUP_COL].nunique()
        log.info("  Fold %d: train=%d samples (%d scenes), val=%d samples (%d scenes)",
                 i + 1, len(tr), n_tr_scenes, len(va), n_va_scenes)
    return folds
