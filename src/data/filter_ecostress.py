"""Apply climatology + Tukey fence filters to remove cold-biased ECOSTRESS pixels.

For each scene the filter applies two passes:

**Pass 1 — Climatology filter (absolute plausibility check):**
  1. Loads precomputed monthly mean/std of gridMET Tmin (1991-2025).
  2. For the scene's calendar month, reprojects the climatological mean
     and std onto the scene's UTM grid.
  3. Masks pixels where LST < (monthly_mean − z × monthly_std).

**Pass 2 — Tukey fence filter (relative statistical check):**
  1. Reprojects gridMET daily Tmin (WGS84, ~4 km) onto the scene's
     UTM grid using bilinear interpolation.
  2. Computes residuals:  r = LST_ecostress − Tmin_gridmet
  3. Applies a left-tail Tukey fence on the residuals:
         lower_fence = Q1 − k * IQR
     where Q1 and Q3 are the 25th/75th percentiles of r, IQR = Q3 − Q1,
     and k is a tunable multiplier (default 1.25).
  4. Masks pixels whose residual falls below the lower fence.

The climatology filter runs first so that severely cold cloud pixels
do not distort the Tukey fence calculation.

Usage
-----
    python -m src.data.filter_ecostress                    # defaults
    python -m src.data.filter_ecostress --k 2.0 --z 2.5   # tune both
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import CRS_GEO, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ECOSTRESS_SCENES_DIR = PROCESSED_DIR / "satellite" / "ecostress_native"
SCENE_INVENTORY_PATH = PROCESSED_DIR / "satellite" / "ecostress_scenes.parquet"
GRIDMET_TMIN_ZARR = PROCESSED_DIR / "gridded_tmin" / "gridmet_tmin_daily.zarr"
GRIDMET_CLIM_ZARR = PROCESSED_DIR / "gridded_tmin" / "gridmet_tmin_climatology.zarr"
FILTERED_DIR = PROCESSED_DIR / "satellite" / "ecostress_filtered"

MIN_VALID_PIXELS = 10  # skip scenes with fewer valid pixels after filtering
LST_FLOOR_C = -30.0    # hard floor: mask pixels colder than this (°C)


def _load_gridmet_tmin() -> xr.Dataset:
    """Open the gridMET daily Tmin Zarr store."""
    return xr.open_zarr(GRIDMET_TMIN_ZARR)


def _load_climatology() -> xr.Dataset:
    """Open the precomputed monthly Tmin climatology Zarr store."""
    if not GRIDMET_CLIM_ZARR.exists():
        raise FileNotFoundError(
            f"Climatology Zarr not found at {GRIDMET_CLIM_ZARR}. "
            "Run 'python -m src.preprocessing.climatology' first."
        )
    return xr.open_zarr(GRIDMET_CLIM_ZARR)


def _climatology_for_month(
    clim_ds: xr.Dataset,
    month: int,
) -> tuple[tuple[np.ndarray, dict], tuple[np.ndarray, dict]]:
    """Extract climatological mean and std for a calendar month.

    Returns ((mean_arr, meta), (std_arr, meta)) with rasterio-style
    metadata for reprojection.
    """
    mean_2d = clim_ds["tmin_mean"].sel(month=month).values.astype(np.float32)
    std_2d = clim_ds["tmin_std"].sel(month=month).values.astype(np.float32)

    lats = clim_ds.lat.values
    lons = clim_ds.lon.values

    dy = float(lats[1] - lats[0]) if len(lats) > 1 else -1 / 24
    dx = float(lons[1] - lons[0]) if len(lons) > 1 else 1 / 24
    from rasterio.transform import from_origin

    transform = from_origin(
        west=float(lons[0]) - dx / 2,
        north=float(lats[0]) - dy / 2,
        xsize=abs(dx),
        ysize=abs(dy),
    )

    meta = {
        "crs": CRS_GEO,
        "transform": transform,
        "height": mean_2d.shape[0],
        "width": mean_2d.shape[1],
    }
    return (mean_2d, meta), (std_2d, meta)


def _gridmet_tmin_for_date(
    ds: xr.Dataset,
    date: np.datetime64,
) -> tuple[np.ndarray, dict] | None:
    """Extract gridMET Tmin for a single date.

    Returns the 2-D array (lat, lon) and a dict with the rasterio-style
    metadata needed for reprojection (crs, transform, shape).
    """
    try:
        tmin = ds["tmin"].sel(time=date, method="nearest")
    except KeyError:
        return None

    vals = tmin.values.astype(np.float32)
    lats = tmin.lat.values
    lons = tmin.lon.values

    # Build an affine transform from the regular lat/lon grid.
    # gridMET lats are *descending*, so dy is negative.
    dy = float(lats[1] - lats[0]) if len(lats) > 1 else -1 / 24
    dx = float(lons[1] - lons[0]) if len(lons) > 1 else 1 / 24
    from rasterio.transform import from_origin

    # upper-left corner
    transform = from_origin(
        west=float(lons[0]) - dx / 2,
        north=float(lats[0]) - dy / 2,  # dy < 0 so subtracting gives north edge
        xsize=abs(dx),
        ysize=abs(dy),
    )

    meta = {
        "crs": CRS_GEO,
        "transform": transform,
        "height": vals.shape[0],
        "width": vals.shape[1],
    }
    return vals, meta


def _reproject_tmin_to_scene(
    tmin_arr: np.ndarray,
    tmin_meta: dict,
    scene_profile: dict,
) -> np.ndarray:
    """Bilinear-reproject gridMET Tmin onto the ECOSTRESS scene grid."""
    dst = np.full(
        (scene_profile["height"], scene_profile["width"]),
        np.nan,
        dtype=np.float32,
    )
    reproject(
        source=tmin_arr,
        destination=dst,
        src_transform=tmin_meta["transform"],
        src_crs=tmin_meta["crs"],
        dst_transform=scene_profile["transform"],
        dst_crs=scene_profile["crs"],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def filter_scene(
    scene_path: Path,
    tmin_reprojected: np.ndarray,
    k: float,
    clim_mean_reprojected: np.ndarray | None = None,
    clim_std_reprojected: np.ndarray | None = None,
    z: float = 3.0,
) -> tuple[np.ndarray, dict] | None:
    """Apply climatology + Tukey left-tail filters to one ECOSTRESS scene.

    Parameters
    ----------
    scene_path : Path to the native ECOSTRESS GeoTIFF.
    tmin_reprojected : gridMET daily Tmin reprojected to scene grid.
    k : Tukey fence multiplier.
    clim_mean_reprojected : Monthly mean Tmin reprojected to scene grid.
    clim_std_reprojected : Monthly std of Tmin reprojected to scene grid.
    z : Number of standard deviations for climatology lower bound.

    Returns (filtered_lst, removal_mask, stats_dict) or None if the scene
    should be skipped (too few valid pixels).  ``removal_mask`` is a boolean
    array (True = pixel removed by any filter).
    """
    with rasterio.open(scene_path) as src:
        lst = src.read(1).astype(np.float32)
        profile = dict(src.profile)

    # Mask nodata
    lst[lst == profile.get("nodata", np.nan)] = np.nan

    # Count valid pixels before any filtering
    n_valid_before = int(np.count_nonzero(~np.isnan(lst)))

    # ── Pass 1: Climatology filter ────────────────────────────────────────
    n_removed_climatology = 0
    climatology_lower_bound_mean = np.nan
    clim_mask = np.zeros(lst.shape, dtype=bool)

    if clim_mean_reprojected is not None and clim_std_reprojected is not None:
        lower_bound = clim_mean_reprojected - z * clim_std_reprojected
        valid_clim = ~np.isnan(lst) & ~np.isnan(lower_bound)
        clim_mask = valid_clim & (lst < lower_bound)
        n_removed_climatology = int(clim_mask.sum())
        lst[clim_mask] = np.nan
        climatology_lower_bound_mean = round(float(np.nanmean(lower_bound)), 3)

    # ── Pass 2: Tukey fence filter ────────────────────────────────────────
    valid = ~np.isnan(lst) & ~np.isnan(tmin_reprojected)
    if valid.sum() < MIN_VALID_PIXELS:
        return None

    residuals = np.full_like(lst, np.nan)
    residuals[valid] = lst[valid] - tmin_reprojected[valid]

    q1 = float(np.nanpercentile(residuals[valid], 25))
    q3 = float(np.nanpercentile(residuals[valid], 75))
    iqr = q3 - q1
    lower_fence = q1 - k * iqr

    # Mask pixels below the lower fence
    cold_mask = valid & (residuals < lower_fence)
    n_removed_tukey = int(cold_mask.sum())
    lst[cold_mask] = np.nan

    # Hard floor: mask any pixel colder than -30 °C (physically implausible)
    floor_mask = ~np.isnan(lst) & (lst < LST_FLOOR_C)
    n_removed_floor = int(floor_mask.sum())
    lst[floor_mask] = np.nan

    n_removed = n_removed_climatology + n_removed_tukey + n_removed_floor

    n_valid_after = int(np.count_nonzero(~np.isnan(lst)))
    if n_valid_after < MIN_VALID_PIXELS:
        return None

    # Combined removal mask (True = removed by any filter)
    removal_mask = clim_mask | cold_mask | floor_mask

    fraction_removed = round(n_removed / n_valid_before, 6) if n_valid_before > 0 else 0.0

    stats = {
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_fence": round(lower_fence, 3),
        "climatology_lower_bound_mean": climatology_lower_bound_mean,
        "pixels_removed": n_removed,
        "pixels_removed_climatology": n_removed_climatology,
        "pixels_removed_tukey": n_removed_tukey,
        "pixels_removed_floor": n_removed_floor,
        "valid_pixels_after": n_valid_after,
        "n_valid_before": n_valid_before,
        "fraction_removed": fraction_removed,
    }
    return lst, removal_mask, stats


def run(k: float = 1.25, z: float = 3.0) -> Path:
    """Filter all ECOSTRESS scenes and write results.

    Parameters
    ----------
    k : Tukey fence multiplier (default 1.25).
    z : Climatology filter z-score threshold (default 3.0).
        Pixels with LST < (monthly_mean − z × monthly_std) are removed.

    Returns
    -------
    Path to the filtered scenes directory.
    """
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR = FILTERED_DIR / "masks"
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_parquet(SCENE_INVENTORY_PATH, engine="fastparquet")
    inventory["datetime_utc"] = pd.to_datetime(inventory["datetime_utc"])

    ds = _load_gridmet_tmin()

    # Load climatology (may not exist yet — warn and proceed without it)
    clim_ds = None
    try:
        clim_ds = _load_climatology()
        log.info("Loaded monthly Tmin climatology from %s", GRIDMET_CLIM_ZARR)
    except FileNotFoundError as e:
        log.warning("%s  — running without climatology filter.", e)

    # Cache reprojected climatology grids per (month, scene_profile_key)
    _clim_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}

    results: list[dict] = []
    skipped = 0

    for _, row in inventory.iterrows():
        scene_path = ECOSTRESS_SCENES_DIR / row["filename"]
        if not scene_path.exists():
            log.warning("Missing scene file: %s", scene_path)
            skipped += 1
            continue

        scene_date = np.datetime64(row["datetime_utc"].date(), "ns")
        scene_month = int(row["datetime_utc"].month)

        tmin_data = _gridmet_tmin_for_date(ds, scene_date)
        if tmin_data is None:
            log.warning("No gridMET Tmin for %s — skipping %s", scene_date, row["filename"])
            skipped += 1
            continue

        tmin_arr, tmin_meta = tmin_data

        # Read scene profile for reprojection target
        with rasterio.open(scene_path) as src:
            scene_profile = dict(src.profile)

        tmin_reprojected = _reproject_tmin_to_scene(tmin_arr, tmin_meta, scene_profile)

        # Reproject climatology grids for this month (cached across scenes)
        clim_mean_reproj = None
        clim_std_reproj = None
        if clim_ds is not None:
            if scene_month not in _clim_cache:
                (mean_arr, mean_meta), (std_arr, std_meta) = _climatology_for_month(
                    clim_ds, scene_month
                )
                _clim_cache[scene_month] = (
                    _reproject_tmin_to_scene(mean_arr, mean_meta, scene_profile),
                    _reproject_tmin_to_scene(std_arr, std_meta, scene_profile),
                )
            clim_mean_reproj, clim_std_reproj = _clim_cache[scene_month]

        result = filter_scene(
            scene_path,
            tmin_reprojected,
            k=k,
            clim_mean_reprojected=clim_mean_reproj,
            clim_std_reprojected=clim_std_reproj,
            z=z,
        )
        if result is None:
            log.debug("Scene %s: too few pixels after filter — skipped.", row["filename"])
            skipped += 1
            continue

        filtered_lst, removal_mask, stats = result

        # Write filtered scene
        out_path = FILTERED_DIR / row["filename"]
        write_profile = scene_profile.copy()
        write_profile.update(nodata=np.nan, compress="deflate", dtype="float32")
        with rasterio.open(out_path, "w", **write_profile) as dst:
            dst.write(filtered_lst.astype(np.float32), 1)

        # Write removal mask (uint8: 1 = removed, 0 = kept/nodata)
        mask_path = MASK_DIR / row["filename"]
        mask_profile = scene_profile.copy()
        mask_profile.update(dtype="uint8", nodata=255, compress="deflate")
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(removal_mask.astype(np.uint8), 1)

        results.append({
            "filename": row["filename"],
            "tukey_k": k,
            "climatology_z": z,
            **stats,
        })

    ds.close()
    if clim_ds is not None:
        clim_ds.close()

    # Save filter diagnostics
    if results:
        diag_df = pd.DataFrame(results)
        diag_path = FILTERED_DIR / "tukey_filter_diagnostics.parquet"
        diag_df.to_parquet(diag_path, index=False)

        total_removed = diag_df["pixels_removed"].sum()
        total_clim = diag_df["pixels_removed_climatology"].sum()
        total_tukey = diag_df["pixels_removed_tukey"].sum()
        log.info("── Filter summary (k=%.2f, z=%.1f) ──", k, z)
        log.info("  Scenes filtered: %d", len(results))
        log.info("  Scenes skipped:  %d", skipped)
        log.info("  Total pixels removed: %d", total_removed)
        log.info("    Climatology: %d", total_clim)
        log.info("    Tukey:       %d", total_tukey)
        log.info("    Hard floor:  %d", diag_df["pixels_removed_floor"].sum())
        log.info("  Median pixels removed per scene: %d", diag_df["pixels_removed"].median())
        log.info("  Diagnostics → %s", diag_path)
        log.info("  Filtered scenes → %s", FILTERED_DIR)
    else:
        log.warning("No scenes were filtered.")

    return FILTERED_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Climatology + Tukey left-tail filter for ECOSTRESS LST"
    )
    parser.add_argument(
        "--k", type=float, default=1.25,
        help="Tukey fence multiplier (default: 1.25)",
    )
    parser.add_argument(
        "--z", type=float, default=3.0,
        help="Climatology z-score threshold (default: 3.0). "
             "Pixels with LST < (monthly_mean - z*std) are removed.",
    )
    args = parser.parse_args()
    run(k=args.k, z=args.z)


if __name__ == "__main__":
    main()
