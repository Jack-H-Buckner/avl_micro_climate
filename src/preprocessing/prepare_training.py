"""Build pixel-level training dataset from ECOSTRESS scenes + covariates + gridMET.

Each row is one valid 100 m pixel from one nighttime ECOSTRESS scene, paired
with:
  - LST (response variable, °C)
  - 8 terrain covariates (static, from DEM-derived GeoTIFFs)
  - 8 land surface covariates (static: NLCD impervious/canopy/classes,
    SVF, distance-to-water)
  - 1 seasonally varying covariate (NDVI from HLS biweekly composites)
  - 11 gridMET meteorological variables (date-matched, bilinear-interpolated
    on-the-fly from the native ~4 km WGS84 Zarr)
  - hours_until_sunrise (temporal predictor)
  - scene metadata (scene_id, date, pixel row/col)

Only nighttime scenes (predawn + evening) with ≥10 % valid-pixel coverage
are included.  A ``--max-samples`` flag controls the total number of pixels
extracted (uniformly subsampled across scenes) to keep the parquet small.

Usage
-----
    python -m src.preprocessing.prepare_training
    python -m src.preprocessing.prepare_training --max-samples 500000
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CENTER_LAT,
    CENTER_LON,
    CRS_UTM,
    COVARIATES_DIR,
    DEM_100M_PATH,
    GRIDMET_DIR,
    NDVI_DIR,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
SATELLITE_DIR = PROCESSED_DIR / "satellite"
ALIGNED_DIR = SATELLITE_DIR / "ecostress_aligned"
ALIGNED_MASK_DIR = ALIGNED_DIR / "masks"
SCENE_INVENTORY = SATELLITE_DIR / "ecostress_scenes.parquet"
FILTER_DIAGNOSTICS = SATELLITE_DIR / "ecostress_filtered" / "tukey_filter_diagnostics.parquet"
GRIDMET_SOURCE_ZARR = GRIDMET_DIR / "gridmet_frost_season.zarr"

TRAINING_DIR = PROCESSED_DIR / "training"
OUTPUT_PATH = TRAINING_DIR / "ecostress_training_samples.parquet"

# Overpass classes considered "nighttime"
NIGHTTIME_CLASSES = ("predawn", "evening")

# Covariate names (must match filenames in COVARIATES_DIR)
TERRAIN_COVARIATE_NAMES = [
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi_300m", "tpi_1000m", "curvature", "twi",
]

# Land surface covariates from NLCD + DEM-derived SVF (Task 1.6, static)
LAND_SURFACE_COVARIATE_NAMES = [
    "impervious_pct",
    "tree_canopy_pct",
    "sky_view_factor",
    "dist_to_water_m",
    "is_forest",
    "is_developed",
    "is_agriculture",
    "is_water",
]

# Combined static covariates
COVARIATE_NAMES = TERRAIN_COVARIATE_NAMES + LAND_SURFACE_COVARIATE_NAMES

# NDVI composites Zarr (seasonally varying, matched per scene)
NDVI_ZARR = NDVI_DIR / "hls_ndvi_composites.zarr"

# ECOSTRESS composite covariates (static spatial layers at 100 m)
ECOSTRESS_COMPOSITE_NAMES = [
    "ecostress_nighttime_median",
    "ecostress_nighttime_q15",
    "ecostress_nighttime_q85",
    "ecostress_nighttime_iqr",
]

# gridMET variable names (must match Zarr data_vars)
GRIDMET_VARS = [
    "tmmn", "tmmx", "tmmx_prev", "diurnal_range",
    "vs", "sph", "vpd", "srad", "srad_prev", "pr", "rmin",
]


# ── Sunrise calculation ────────────────────────────────────────────────────

def _sunrise_utc(date, lat: float = CENTER_LAT, lon: float = CENTER_LON) -> datetime:
    """Approximate sunrise time (UTC) using the NOAA solar equations.

    Accurate to ~1–2 minutes for mid-latitudes.  Good enough for the
    ``hours_until_sunrise`` predictor.
    """
    from math import acos, cos, radians, sin, tan, degrees

    doy = date.timetuple().tm_yday
    # Fractional year (radians)
    gamma = 2 * np.pi / 365 * (doy - 1)

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * cos(gamma) - 0.032077 * sin(gamma)
        - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * cos(gamma) + 0.070257 * sin(gamma)
        - 0.006758 * cos(2 * gamma) + 0.000907 * sin(2 * gamma)
        - 0.002697 * cos(3 * gamma) + 0.00148 * sin(3 * gamma)
    )

    lat_rad = radians(lat)
    # Hour angle at sunrise (degrees)
    cos_ha = (cos(radians(90.833)) / (cos(lat_rad) * cos(decl))) - tan(lat_rad) * tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))  # clamp for polar edge cases
    ha = degrees(acos(cos_ha))

    # Sunrise in minutes from midnight UTC
    sunrise_min = 720 - 4 * (lon + ha) - eqtime
    h, m = divmod(int(sunrise_min), 60)
    h = h % 24

    return datetime(date.year, date.month, date.day, h, m, tzinfo=timezone.utc)


def _hours_until_sunrise(dt_utc: datetime) -> float:
    """Hours from observation time to next sunrise (positive = before sunrise)."""
    sr = _sunrise_utc(dt_utc.date())
    # If observation is after this morning's sunrise, use tomorrow's
    if dt_utc >= sr:
        sr = _sunrise_utc(dt_utc.date() + timedelta(days=1))
    diff = (sr - dt_utc).total_seconds() / 3600
    return max(diff, 0.0)


def _parse_scene_datetime(filename: str) -> datetime:
    """Parse UTC datetime from ECOSTRESS filename like '20190115T024352_evening.tif'."""
    stem = filename.split("_")[0]  # '20190115T024352'
    dt = datetime.strptime(stem, "%Y%m%dT%H%M%S")
    return dt.replace(tzinfo=timezone.utc)


def _scene_gridmet_date(dt_utc: datetime) -> np.datetime64:
    """Map an ECOSTRESS overpass time to the correct gridMET date.

    gridMET "day D" covers the 24 h ending 12:00 UTC on day D (= 07:00 EST).
    A predawn pass at e.g. 05:00 UTC on Jan 15 falls within gridMET day Jan 15.
    An evening pass at e.g. 02:00 UTC on Jan 15 (= 9 PM EST Jan 14) also falls
    within gridMET day Jan 15 (the 24 h ending 12 UTC Jan 15).
    """
    if dt_utc.hour >= 12:
        gm_date = dt_utc.date() + timedelta(days=1)
    else:
        gm_date = dt_utc.date()
    return np.datetime64(gm_date)


# ── Data loaders ───────────────────────────────────────────────────────────

# Binary land class layers: nodata means "not this class" → fill with 0
_BINARY_COVARIATES = {"is_forest", "is_developed", "is_agriculture", "is_water"}

# Continuous land surface layers where nodata should become 0
_ZERO_FILL_COVARIATES = {"impervious_pct"}


def _load_covariates() -> dict[str, np.ndarray]:
    """Load all terrain + land surface + ECOSTRESS composite covariates."""
    covs = {}
    # Terrain + land surface covariates from COVARIATES_DIR
    for name in COVARIATE_NAMES:
        path = COVARIATES_DIR / f"{name}.tif"
        if not path.exists():
            log.warning("Covariate not found: %s — skipping", path.name)
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            nodata_mask = np.isclose(arr, nodata)
            if name in _BINARY_COVARIATES or name in _ZERO_FILL_COVARIATES:
                arr[nodata_mask] = 0.0
            else:
                arr[nodata_mask] = np.nan
        covs[name] = arr

    # ECOSTRESS composite covariates (100 m aligned versions)
    for name in ECOSTRESS_COMPOSITE_NAMES:
        path = SATELLITE_DIR / f"{name}_100m.tif"
        if not path.exists():
            log.warning("ECOSTRESS composite not found: %s — skipping", path.name)
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            arr[np.isclose(arr, nodata)] = np.nan
        covs[name] = arr

    return covs


# gridMET variables where nearest-neighbour is more appropriate than bilinear
_NEAREST_GRIDMET_VARS = set()  # all variables use cubic spline


def _open_gridmet_source() -> tuple[xr.Dataset, dict]:
    """Open the native ~4 km WGS84 gridMET Zarr and pre-compute reprojection params.

    Returns (ds, reproj_params) where reproj_params contains everything needed
    for on-the-fly bilinear interpolation to the 100 m UTM grid.
    """
    ds = xr.open_zarr(str(GRIDMET_SOURCE_ZARR))
    lats = ds.lat.values
    lons = ds.lon.values
    lat_sorted = np.sort(lats)
    lon_sorted = np.sort(lons)

    dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 1 / 24
    dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 1 / 24

    src_left = float(lon_sorted[0]) - dlon / 2
    src_right = float(lon_sorted[-1]) + dlon / 2
    src_bottom = float(lat_sorted[0]) - dlat / 2
    src_top = float(lat_sorted[-1]) + dlat / 2

    src_h, src_w = len(lats), len(lons)
    src_transform = from_bounds(src_left, src_bottom, src_right, src_top, src_w, src_h)

    # Read reference grid from the DEM
    with rasterio.open(DEM_100M_PATH) as src:
        ref = dict(src.profile)

    reproj_params = {
        "src_transform": src_transform,
        "src_crs": "EPSG:4326",
        "dst_transform": ref["transform"],
        "dst_crs": CRS_UTM,
        "dst_h": ref["height"],
        "dst_w": ref["width"],
        "lats_descending": lats[0] > lats[-1],
    }
    return ds, reproj_params


def _interpolate_gridmet_for_date(
    ds: xr.Dataset, gm_date, reproj_params: dict,
) -> dict[str, np.ndarray] | None:
    """Bilinear-interpolate all gridMET variables for one date to the 100 m grid.

    Returns a dict of variable name → (dst_h, dst_w) float32 arrays, or None
    if the date is not available.
    """
    try:
        day_slice = ds.sel(time=gm_date)
    except KeyError:
        return None

    dst_h = reproj_params["dst_h"]
    dst_w = reproj_params["dst_w"]
    result = {}

    for var in GRIDMET_VARS:
        arr = day_slice[var].values.astype(np.float32)
        # Ensure lat is north→south
        if not reproj_params["lats_descending"]:
            arr = arr[::-1, :]

        dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        method = Resampling.nearest if var in _NEAREST_GRIDMET_VARS else Resampling.cubic_spline
        reproject(
            source=arr,
            destination=dst,
            src_transform=reproj_params["src_transform"],
            src_crs=reproj_params["src_crs"],
            dst_transform=reproj_params["dst_transform"],
            dst_crs=reproj_params["dst_crs"],
            resampling=method,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        result[var] = dst

    return result


def _load_ndvi() -> xr.Dataset | None:
    """Open the biweekly NDVI composite Zarr, or None if unavailable."""
    if not NDVI_ZARR.exists():
        log.warning("NDVI composites not found at %s — NDVI will be omitted.", NDVI_ZARR)
        return None
    ds = xr.open_zarr(str(NDVI_ZARR))
    log.info("Loaded NDVI composites: %d timesteps, grid %d × %d",
             ds.sizes["time"], ds.sizes["y"], ds.sizes["x"])
    return ds


def _get_nearest_ndvi(ndvi_ds: xr.Dataset, scene_date) -> np.ndarray | None:
    """Return the NDVI 2-D array from the composite nearest to scene_date."""
    target = np.datetime64(scene_date)
    time_diffs = np.abs(ndvi_ds.time.values - target)
    nearest_idx = int(np.argmin(time_diffs))
    return ndvi_ds["ndvi"].isel(time=nearest_idx).values


# ── Core extraction ────────────────────────────────────────────────────────

def extract_samples(
    min_coverage: float = 0.10,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Extract pixel-level training samples from all nighttime ECOSTRESS scenes.

    Parameters
    ----------
    min_coverage : float
        Minimum fraction of valid pixels for a scene to be included.
    max_samples : int or None
        If set, cap total extracted samples to approximately this many by
        randomly subsampling each scene proportionally.

    Returns
    -------
    DataFrame with one row per valid pixel per scene.
    """
    # ── Load static data ─────────────────────────────────────────────
    log.info("Loading covariates (terrain + land surface) ...")
    covariates = _load_covariates()
    grid_h, grid_w = next(iter(covariates.values())).shape
    log.info("  Grid: %d × %d, covariates: %s", grid_h, grid_w, list(covariates.keys()))

    log.info("Loading gridMET source Zarr (on-the-fly bilinear interpolation) ...")
    gridmet_ds, reproj_params = _open_gridmet_source()

    log.info("Loading NDVI composites ...")
    ndvi_ds = _load_ndvi()

    log.info("Loading scene inventory ...")
    inventory = pd.read_parquet(SCENE_INVENTORY)
    total_pixels = inventory["total_pixels"].iloc[0]

    # Load filter diagnostics for fraction_removed per scene
    filter_diag = None
    if FILTER_DIAGNOSTICS.exists():
        filter_diag = pd.read_parquet(FILTER_DIAGNOSTICS)
        filter_diag = filter_diag.drop_duplicates(subset="filename").set_index("filename")
        log.info("Loaded filter diagnostics for %d scenes", len(filter_diag))
    else:
        log.warning("Filter diagnostics not found at %s — fraction_scene_removed will be NaN",
                     FILTER_DIAGNOSTICS)

    # Filter to nighttime + coverage
    scenes = inventory[
        inventory["overpass_class"].isin(NIGHTTIME_CLASSES)
        & (inventory["valid_pixels"] > total_pixels * min_coverage)
    ].copy()
    log.info("Qualifying nighttime scenes: %d (of %d total nighttime)",
             len(scenes),
             inventory["overpass_class"].isin(NIGHTTIME_CLASSES).sum())

    if scenes.empty:
        log.warning("No qualifying scenes found.")
        return pd.DataFrame()

    # ── Build valid-covariate mask (all covariates finite) ───────────
    cov_mask = np.ones((grid_h, grid_w), dtype=bool)
    for arr in covariates.values():
        cov_mask &= np.isfinite(arr)
    log.info("Pixels with all covariates valid: %d / %d", cov_mask.sum(), cov_mask.size)

    # Pre-flatten covariates for valid pixels
    cov_idx = np.where(cov_mask.ravel())[0]  # flat indices of valid-covariate pixels
    cov_flat = {name: arr.ravel()[cov_idx] for name, arr in covariates.items()}
    row_idx, col_idx = np.unravel_index(cov_idx, (grid_h, grid_w))

    # ── Compute per-scene subsampling rate ─────────────────────────────
    # If max_samples is set, we subsample each scene to keep total samples
    # near the budget.  Assume ~n_valid_cov pixels per scene on average.
    n_valid_cov = int(cov_mask.sum())
    rng = np.random.default_rng(42)
    if max_samples is not None:
        # Rough estimate: each scene contributes ~n_valid_cov pixels
        est_total = n_valid_cov * len(scenes)
        sample_frac = min(1.0, max_samples / est_total)
        log.info("Subsampling: target %d samples, est. total %d, frac %.4f",
                 max_samples, est_total, sample_frac)
    else:
        sample_frac = 1.0

    # ── Process each scene ───────────────────────────────────────────
    # Stream batches directly to the output parquet via PyArrow writer
    # to avoid doubling disk usage with temp files.
    import pyarrow as pa
    import pyarrow.parquet as pq

    BATCH_SIZE = 50  # scenes per batch
    batch_chunks = []
    n_scenes = len(scenes)
    total_samples = 0
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    pq_writer = None

    # Cache for gridMET interpolations (same date may repeat across scenes)
    _gm_cache: dict[str, dict[str, np.ndarray]] = {}

    for i, (_, row) in enumerate(scenes.iterrows(), 1):
        scene_id = row["granule_id"]
        filename = row["filename"]
        scene_path = ALIGNED_DIR / filename

        if not scene_path.exists():
            continue

        # Parse datetime and compute temporal predictor
        dt_utc = _parse_scene_datetime(filename)
        hrs = _hours_until_sunrise(dt_utc)

        # Load scene LST
        with rasterio.open(scene_path) as src:
            lst = src.read(1).astype(np.float32)

        # Valid = finite LST AND valid covariates
        lst_flat = lst.ravel()
        valid = np.isfinite(lst_flat[cov_idx])
        n_valid = valid.sum()

        if n_valid == 0:
            continue

        # Subsample valid pixels if needed
        if sample_frac < 1.0:
            n_keep = max(1, int(n_valid * sample_frac))
            valid_indices = np.where(valid)[0]
            chosen = rng.choice(valid_indices, size=n_keep, replace=False)
            sub_mask = np.zeros_like(valid)
            sub_mask[chosen] = True
            valid = sub_mask
            n_valid = n_keep

        # On-the-fly bilinear interpolation of gridMET for this date
        gm_date = _scene_gridmet_date(dt_utc)
        gm_date_str = str(gm_date)
        if gm_date_str in _gm_cache:
            gm_grids = _gm_cache[gm_date_str]
        else:
            gm_grids = _interpolate_gridmet_for_date(gridmet_ds, gm_date, reproj_params)
            if gm_grids is None:
                log.debug("No gridMET data for %s — skipping scene %s", gm_date, scene_id)
                continue
            # Keep only the last few dates cached to limit memory
            if len(_gm_cache) > 5:
                _gm_cache.pop(next(iter(_gm_cache)))
            _gm_cache[gm_date_str] = gm_grids

        # Extract gridMET variables at valid pixel locations
        gm_arrays = {}
        for var in GRIDMET_VARS:
            gm_arrays[var] = gm_grids[var].ravel()[cov_idx][valid].astype(np.float32)

        # Compute residual: LST - gridMET Tmin (the downscaling target)
        lst_vals = lst_flat[cov_idx][valid]
        tmmn_vals = gm_arrays["tmmn"]
        lst_residual = lst_vals - tmmn_vals

        # Build chunk DataFrame
        chunk = {
            "lst": lst_vals,
            "lst_residual": lst_residual,
            "hours_until_sunrise": np.full(n_valid, hrs, dtype=np.float32),
            "scene_id": scene_id,
            "date": str(dt_utc.date()),
            "pixel_row": row_idx[valid],
            "pixel_col": col_idx[valid],
        }

        # Terrain covariates
        for name, arr_flat in cov_flat.items():
            chunk[name] = arr_flat[valid]

        # gridMET variables (already computed above)
        for var in GRIDMET_VARS:
            chunk[var] = gm_arrays[var]

        # NDVI (seasonally varying — nearest biweekly composite)
        if ndvi_ds is not None:
            ndvi_arr = _get_nearest_ndvi(ndvi_ds, dt_utc.date())
            if ndvi_arr is not None and ndvi_arr.shape == (grid_h, grid_w):
                chunk["ndvi"] = ndvi_arr.ravel()[cov_idx][valid].astype(np.float32)
            else:
                chunk["ndvi"] = np.full(n_valid, np.nan, dtype=np.float32)

        # Distance to nearest removed pixel (from cloud filter mask)
        mask_path = ALIGNED_MASK_DIR / filename
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                removal_mask = src.read(1)  # uint8: 1=removed, 0=kept, 255=nodata
            has_removed = np.any(removal_mask == 1)
            if has_removed:
                # distance_transform_edt on (mask != 1) gives each non-removed
                # pixel its distance to the nearest removed pixel, in pixel units
                dist_pixels = distance_transform_edt(removal_mask != 1).astype(np.float32)
                dist_m = dist_pixels * 100.0  # 100 m per pixel
            else:
                # No pixels removed — set distance to image diagonal
                diag_m = np.sqrt((grid_h * 100.0) ** 2 + (grid_w * 100.0) ** 2)
                dist_m = np.full((grid_h, grid_w), diag_m, dtype=np.float32)
            chunk["dist_to_removed_m"] = dist_m.ravel()[cov_idx][valid]
        else:
            chunk["dist_to_removed_m"] = np.full(n_valid, np.nan, dtype=np.float32)

        # Fraction of scene pixels removed by cloud filter
        if filter_diag is not None and filename in filter_diag.index:
            row_diag = filter_diag.loc[filename]
            # Handle duplicate index (take first row)
            if isinstance(row_diag, pd.DataFrame):
                row_diag = row_diag.iloc[0]
            if "fraction_removed" in filter_diag.columns:
                frac = float(row_diag["fraction_removed"])
            elif "pixels_removed" in filter_diag.columns and "valid_pixels_after" in filter_diag.columns:
                pr = float(row_diag["pixels_removed"])
                va = float(row_diag["valid_pixels_after"])
                frac = pr / (pr + va) if (pr + va) > 0 else np.nan
            else:
                frac = np.nan
        else:
            frac = np.nan
        chunk["fraction_scene_removed"] = np.full(n_valid, frac, dtype=np.float32)

        batch_chunks.append(pd.DataFrame(chunk))

        # Flush batch directly to the output parquet every BATCH_SIZE scenes
        if len(batch_chunks) >= BATCH_SIZE:
            batch_df = pd.concat(batch_chunks, ignore_index=True)
            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(str(OUTPUT_PATH), table.schema)
            pq_writer.write_table(table)
            total_samples += len(batch_df)
            log.info("  ... %d / %d scenes processed (%d samples so far)",
                     i, n_scenes, total_samples)
            del batch_df, table
            batch_chunks = []

    # Flush remaining scenes
    if batch_chunks:
        batch_df = pd.concat(batch_chunks, ignore_index=True)
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if pq_writer is None:
            pq_writer = pq.ParquetWriter(str(OUTPUT_PATH), table.schema)
        pq_writer.write_table(table)
        total_samples += len(batch_df)
        del batch_df, table
        batch_chunks = []

    if pq_writer is not None:
        pq_writer.close()

    if total_samples == 0:
        log.warning("No valid samples extracted.")
        return pd.DataFrame()

    # Read back for summary stats
    df = pd.read_parquet(OUTPUT_PATH)
    log.info("Total training samples: %d from %d scenes", len(df), df["scene_id"].nunique())
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────

def run(min_coverage: float = 0.10, max_samples: int | None = 500_000) -> Path:
    """Build and save the training dataset."""
    df = extract_samples(min_coverage=min_coverage, max_samples=max_samples)

    if df.empty:
        log.error("No training samples — aborting.")
        return OUTPUT_PATH

    # extract_samples() already writes directly to OUTPUT_PATH via PyArrow
    log.info("Saved → %s (%.1f MB)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size / 1e6)

    # ── Summary ──────────────────────────────────────────────────────
    log.info("── Training data summary ──")
    log.info("  Samples:  %d", len(df))
    log.info("  Scenes:   %d", df["scene_id"].nunique())
    log.info("  Dates:    %s to %s", df["date"].min(), df["date"].max())
    log.info("  LST:      %.1f to %.1f °C (mean %.1f)",
             df["lst"].min(), df["lst"].max(), df["lst"].mean())
    log.info("  LST residual (LST−Tmin): %.1f to %.1f °C (mean %.1f)",
             df["lst_residual"].min(), df["lst_residual"].max(), df["lst_residual"].mean())
    log.info("  Hours until sunrise: %.1f to %.1f (mean %.1f)",
             df["hours_until_sunrise"].min(),
             df["hours_until_sunrise"].max(),
             df["hours_until_sunrise"].mean())

    # Quick covariate stats
    log.info("  ── Covariate ranges ──")
    eco_cols = [c for c in ECOSTRESS_COMPOSITE_NAMES if c in df.columns]
    all_cov_cols = COVARIATE_NAMES + eco_cols + GRIDMET_VARS
    if "ndvi" in df.columns:
        all_cov_cols = all_cov_cols + ["ndvi"]
    for col in all_cov_cols:
        if col in df.columns:
            log.info("    %-22s  min=%.2f  max=%.2f  mean=%.2f",
                     col, df[col].min(), df[col].max(), df[col].mean())

    return OUTPUT_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Build pixel-level training dataset from ECOSTRESS + covariates + gridMET",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.10,
        help="Minimum valid pixel fraction per scene (default: 0.10)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=500_000,
        help="Max total pixel samples to extract (default: 500,000). Use 0 for no limit.",
    )
    args = parser.parse_args()
    max_s = args.max_samples if args.max_samples > 0 else None
    run(min_coverage=args.min_coverage, max_samples=max_s)


if __name__ == "__main__":
    main()
