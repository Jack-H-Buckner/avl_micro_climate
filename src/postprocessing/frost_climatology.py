"""Frost climatology: weekly P(Tmin < 0°C) maps at 100m resolution.

Two-stage pipeline:
  Stage 1 — Sample 2,000 locations via LHS on static covariates, predict daily
            Tmin across ~25 frost seasons, fit harmonic regression per point,
            compute weekly frost exceedance probabilities.
  Stage 2 — Train a GBM on logit(frost_prob) ~ covariates + week, predict
            across the full 283k-pixel grid for 30 frost-season weeks.

Functions
---------
generate_sample_points      LHS sampling on 16-D covariate space
predict_tmin_at_samples     Batch RF prediction across all dates
compute_weekly_frost_probs  Harmonic regression + Gaussian CDF per point
train_spatial_model         GBM on logit(frost_prob)
predict_frost_maps          Full-grid prediction for 30 weeks
"""

import json
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.stats import norm, qmc
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CENTER_LAT,
    CENTER_LON,
    COVARIATES_DIR,
    CRS_UTM,
    DEM_100M_PATH,
    FIGURES_DIR,
    GRIDMET_DIR,
    NDVI_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
)
from src.model.cross_validation import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
SATELLITE_DIR = PROCESSED_DIR / "satellite"
GRIDMET_SOURCE_ZARR = GRIDMET_DIR / "gridmet_frost_season.zarr"
NDVI_ZARR = NDVI_DIR / "hls_ndvi_composites.zarr"
DEFAULT_MODEL = OUTPUT_DIR / "models" / "rf_benchmark_hrs4.pkl"
FROST_CLIM_DIR = OUTPUT_DIR / "frost_climatology"

# Covariate name lists
TERRAIN_NAMES = [
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi_300m", "tpi_1000m", "curvature", "twi",
]
LAND_SURFACE_NAMES = [
    "impervious_pct", "tree_canopy_pct", "sky_view_factor",
    "dist_to_water_m", "is_forest", "is_developed",
    "is_agriculture", "is_water",
]
STATIC_COV_NAMES = TERRAIN_NAMES + LAND_SURFACE_NAMES

ECOSTRESS_COMPOSITE_NAMES = [
    "ecostress_nighttime_median", "ecostress_nighttime_q15",
    "ecostress_nighttime_q85", "ecostress_nighttime_iqr",
]
GRIDMET_VARS = [
    "tmmn", "tmmx", "tmmx_prev", "diurnal_range",
    "vs", "sph", "vpd", "srad", "srad_prev", "pr", "rmin",
]
_BINARY_COVARIATES = {"is_forest", "is_developed", "is_agriculture", "is_water"}
_ZERO_FILL_COVARIATES = {"impervious_pct"}

# Frost season weeks: 30 weeks from Sep wk1 (DOY ~244) through Apr wk4 (DOY ~120)
FROST_MONTHS = [1, 2, 3, 4, 5, 9, 10, 11, 12]


# ── Shared data loaders (mirrors predict_tmin_map.py) ───────────────────────

def _load_covariates():
    """Load all static + ECOSTRESS composite covariates."""
    covs = {}
    for name in TERRAIN_NAMES + LAND_SURFACE_NAMES:
        path = COVARIATES_DIR / f"{name}.tif"
        if not path.exists():
            log.warning("Covariate not found: %s — skipping", path.name)
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            mask = np.isclose(arr, nodata)
            if name in _BINARY_COVARIATES or name in _ZERO_FILL_COVARIATES:
                arr[mask] = 0.0
            else:
                arr[mask] = np.nan
        covs[name] = arr

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


def _load_gridmet_for_date(target_date, ds_cache=None):
    """Bilinear-interpolate gridMET to 100m grid for one date.

    Parameters
    ----------
    target_date : date
    ds_cache : xr.Dataset or None
        Pre-opened Zarr to avoid repeated open/close.
    """
    import xarray as xr
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    if ds_cache is None:
        ds = xr.open_zarr(str(GRIDMET_SOURCE_ZARR))
    else:
        ds = ds_cache

    gm_date = np.datetime64(target_date)
    try:
        day_slice = ds.sel(time=gm_date)
    except KeyError:
        if ds_cache is None:
            ds.close()
        return None

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

    with rasterio.open(DEM_100M_PATH) as src:
        ref = dict(src.profile)
    dst_h, dst_w = ref["height"], ref["width"]
    dst_transform = ref["transform"]

    gm = {}
    for var in GRIDMET_VARS:
        arr = day_slice[var].values.astype(np.float32)
        if lats[0] < lats[-1]:
            arr = arr[::-1, :]
        dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs="EPSG:4326",
            dst_transform=dst_transform, dst_crs=CRS_UTM,
            resampling=Resampling.cubic_spline,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
        gm[var] = dst

    if ds_cache is None:
        ds.close()
    return gm


def _load_nearest_ndvi(target_date, ndvi_ds=None):
    """Load NDVI composite nearest to target_date."""
    import xarray as xr

    if ndvi_ds is None:
        if not NDVI_ZARR.exists():
            return None
        ds = xr.open_zarr(str(NDVI_ZARR))
    else:
        ds = ndvi_ds

    target = np.datetime64(target_date)
    diffs = np.abs(ds.time.values - target)
    nearest_idx = int(np.argmin(diffs))
    arr = ds["ndvi"].isel(time=nearest_idx).values.astype(np.float32)

    if ndvi_ds is None:
        ds.close()
    return arr


def _build_ndvi_climatology(ndvi_ds):
    """Build a week-of-year NDVI climatology from available composites.

    Returns a dict mapping week_number (1-52) -> 2D NDVI array.
    """
    import xarray as xr

    if ndvi_ds is None:
        return None

    times = pd.DatetimeIndex(ndvi_ds.time.values)
    weeks = times.isocalendar().week.values
    unique_weeks = np.unique(weeks)

    clim = {}
    for w in unique_weeks:
        mask = weeks == w
        indices = np.where(mask)[0]
        if len(indices) > 0:
            stack = np.stack(
                [ndvi_ds["ndvi"].isel(time=int(i)).values for i in indices],
                axis=0,
            )
            clim[int(w)] = np.nanmedian(stack, axis=0).astype(np.float32)

    return clim


def _get_ndvi_for_date(target_date, ndvi_ds, ndvi_clim, earliest_ndvi_year=2013):
    """Get NDVI for a date: actual composite if post-2013, climatology if earlier."""
    if ndvi_ds is None:
        return None

    if target_date.year >= earliest_ndvi_year:
        return _load_nearest_ndvi(target_date, ndvi_ds=ndvi_ds)
    else:
        if ndvi_clim is None:
            return _load_nearest_ndvi(target_date, ndvi_ds=ndvi_ds)
        iso_week = target_date.isocalendar()[1]
        if iso_week in ndvi_clim:
            return ndvi_clim[iso_week]
        nearest_week = min(ndvi_clim.keys(), key=lambda w: abs(w - iso_week))
        return ndvi_clim[nearest_week]


def _sunrise_utc(dt, lat=CENTER_LAT, lon=CENTER_LON):
    """Approximate sunrise (UTC) using NOAA solar equations."""
    from datetime import datetime, timezone
    from math import acos, cos, degrees, radians, sin, tan

    doy = dt.timetuple().tm_yday
    gamma = 2 * np.pi / 365 * (doy - 1)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * cos(gamma) - 0.032077 * sin(gamma)
        - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * cos(gamma) + 0.070257 * sin(gamma)
        - 0.006758 * cos(2 * gamma) + 0.000907 * sin(2 * gamma)
        - 0.002697 * cos(3 * gamma) + 0.00148 * sin(3 * gamma)
    )
    lat_rad = radians(lat)
    cos_ha = (cos(radians(90.833)) / (cos(lat_rad) * cos(decl))) - tan(lat_rad) * tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = degrees(acos(cos_ha))
    sunrise_min = 720 - 4 * (lon + ha) - eqtime
    h, m = divmod(int(sunrise_min), 60)
    h = h % 24
    return datetime(dt.year, dt.month, dt.day, h, m, tzinfo=timezone.utc)


def _get_reference_profile():
    """Read raster profile from elevation GeoTIFF."""
    with rasterio.open(COVARIATES_DIR / "elevation.tif") as src:
        return dict(src.profile)


# ── Step 1: Sample point generation ─────────────────────────────────────────

def generate_sample_points(n=2000, min_separation_m=500, seed=42):
    """Generate n sample points via LHS on the 16 static covariates.

    Each sample maps to a real pixel on the 100m grid.

    Returns
    -------
    DataFrame with columns: pixel_row, pixel_col, + all 16 covariate values.
    """
    log.info("Loading static covariates for LHS sampling ...")
    covariates = _load_covariates()

    # Get grid shape from first covariate
    grid_h, grid_w = next(iter(covariates.values())).shape

    # Build valid pixel mask (all static covariates finite)
    valid_mask = np.ones((grid_h, grid_w), dtype=bool)
    for name in STATIC_COV_NAMES:
        if name in covariates:
            valid_mask &= np.isfinite(covariates[name])
    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)
    log.info("  Valid pixels: %d", n_valid)

    # Extract covariate values at all valid pixels
    cov_matrix = np.column_stack([
        covariates[name].ravel()[np.ravel_multi_index((valid_rows, valid_cols), (grid_h, grid_w))]
        for name in STATIC_COV_NAMES
    ])

    # Standardize to [0, 1] for LHS matching
    cov_min = cov_matrix.min(axis=0)
    cov_max = cov_matrix.max(axis=0)
    cov_range = cov_max - cov_min
    cov_range[cov_range == 0] = 1.0  # avoid div by zero for constant columns
    cov_std = (cov_matrix - cov_min) / cov_range

    # Strategy: use stratified random sampling on key covariates rather than
    # full 16-D LHS matching, which is prohibitively slow at 270k points.
    # Bin elevation + TPI into strata, randomly sample within each stratum,
    # then verify covariate coverage.
    log.info("  Generating stratified sample on key covariates ...")
    rng = np.random.default_rng(seed)

    # Use elevation (primary driver of frost) and TPI-300m (cold air pooling)
    elev_idx = STATIC_COV_NAMES.index("elevation")
    tpi_idx = STATIC_COV_NAMES.index("tpi_300m")
    elev = cov_matrix[:, elev_idx]
    tpi = cov_matrix[:, tpi_idx]

    # Create 2-D bins: 20 elevation bins × 10 TPI bins = 200 strata
    n_elev_bins = 20
    n_tpi_bins = 10
    elev_edges = np.linspace(elev.min(), elev.max() + 0.01, n_elev_bins + 1)
    tpi_edges = np.linspace(tpi.min(), tpi.max() + 0.01, n_tpi_bins + 1)

    elev_bin = np.digitize(elev, elev_edges) - 1
    tpi_bin = np.digitize(tpi, tpi_edges) - 1

    # Target samples per stratum
    n_strata = n_elev_bins * n_tpi_bins
    per_stratum = max(1, n // n_strata)
    extra = n - per_stratum * n_strata

    candidate_indices = []
    for ei in range(n_elev_bins):
        for ti in range(n_tpi_bins):
            mask = (elev_bin == ei) & (tpi_bin == ti)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            k = min(per_stratum, len(idxs))
            chosen = rng.choice(idxs, size=k, replace=False)
            candidate_indices.extend(chosen.tolist())

    # If we need more, sample randomly from the remaining
    if len(candidate_indices) < n * 2:
        remaining = set(range(n_valid)) - set(candidate_indices)
        extra_needed = min(n * 2 - len(candidate_indices), len(remaining))
        candidate_indices.extend(
            rng.choice(list(remaining), size=extra_needed, replace=False).tolist()
        )

    rng.shuffle(candidate_indices)
    indices = np.array(candidate_indices)
    log.info("  Generated %d candidates across %d strata", len(indices), n_strata)

    # Deduplicate and enforce minimum separation via occupied-cell set
    selected = []
    selected_set = set()
    occupied = set()  # set of (row, col) tuples within exclusion radius
    min_sep_pixels = int(np.ceil(min_separation_m / 100.0))

    for idx in indices:
        if idx in selected_set:
            continue
        r, c = int(valid_rows[idx]), int(valid_cols[idx])
        if (r, c) in occupied:
            continue
        selected.append((r, c))
        selected_set.add(idx)
        # Mark nearby cells as occupied
        for dr in range(-min_sep_pixels, min_sep_pixels + 1):
            for dc in range(-min_sep_pixels, min_sep_pixels + 1):
                if dr * dr + dc * dc <= min_sep_pixels * min_sep_pixels:
                    occupied.add((r + dr, c + dc))
        if len(selected) >= n:
            break

    # Force-include extreme covariate pixels
    extreme_features = ["elevation", "tpi_300m", "tpi_1000m"]
    for feat_name in extreme_features:
        if feat_name not in covariates:
            continue
        feat_idx = STATIC_COV_NAMES.index(feat_name)
        col = cov_matrix[:, feat_idx]
        for extreme_idx in [int(np.argmin(col)), int(np.argmax(col))]:
            r, c = valid_rows[extreme_idx], valid_cols[extreme_idx]
            if (r, c) not in [(s[0], s[1]) for s in selected]:
                selected.append((r, c))

    log.info("  Selected %d sample points (requested %d)", len(selected), n)

    # Build output DataFrame
    rows_arr = np.array([s[0] for s in selected])
    cols_arr = np.array([s[1] for s in selected])
    flat_idx = np.ravel_multi_index((rows_arr, cols_arr), (grid_h, grid_w))

    data = {"pixel_row": rows_arr, "pixel_col": cols_arr}
    for name in STATIC_COV_NAMES:
        if name in covariates:
            data[name] = covariates[name].ravel()[flat_idx]

    df = pd.DataFrame(data)
    return df


# ── Step 2: Batch Tmin prediction ───────────────────────────────────────────

def _extract_point_features(rows, cols, covariates, gridmet, ndvi, hours_until_sunrise):
    """Build feature matrix for N points in FEATURE_COLS order."""
    n_pts = len(rows)
    grid_h, grid_w = next(iter(covariates.values())).shape
    diag_m = np.sqrt((grid_h * 100.0) ** 2 + (grid_w * 100.0) ** 2)

    cols_list = []
    for feat in FEATURE_COLS:
        if feat == "hours_until_sunrise":
            cols_list.append(np.full(n_pts, hours_until_sunrise, dtype=np.float32))
        elif feat == "dist_to_removed_m":
            cols_list.append(np.full(n_pts, diag_m, dtype=np.float32))
        elif feat == "fraction_scene_removed":
            cols_list.append(np.zeros(n_pts, dtype=np.float32))
        elif feat == "ndvi":
            if ndvi is not None:
                cols_list.append(ndvi[rows, cols].astype(np.float32))
            else:
                cols_list.append(np.full(n_pts, np.nan, dtype=np.float32))
        elif feat in gridmet:
            cols_list.append(gridmet[feat][rows, cols].astype(np.float32))
        elif feat in covariates:
            cols_list.append(covariates[feat][rows, cols].astype(np.float32))
        else:
            cols_list.append(np.full(n_pts, np.nan, dtype=np.float32))

    return np.column_stack(cols_list)


def predict_tmin_at_samples(
    sample_df,
    model_path=DEFAULT_MODEL,
    hrs_candidates=None,
    progress_interval=100,
):
    """Predict daily Tmin at all sample points across all available dates.

    Parameters
    ----------
    sample_df : DataFrame with pixel_row, pixel_col columns.
    model_path : Path to RF model pickle.
    hrs_candidates : array of hours_until_sunrise values to sweep.
    progress_interval : Log progress every N dates.

    Returns
    -------
    DataFrame with columns: sample_idx, date, predicted_tmin_C, gridmet_tmin_C,
                            lst_residual_C, best_hours.
    """
    import xarray as xr

    if hrs_candidates is None:
        hrs_candidates = np.arange(0.5, 5.0, 0.25)

    rows = sample_df["pixel_row"].values
    cols = sample_df["pixel_col"].values
    n_pts = len(rows)

    # Load model
    log.info("Loading RF model ...")
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    # Load static covariates
    log.info("Loading static covariates ...")
    covariates = _load_covariates()

    # Open gridMET and NDVI Zarrs (keep open for the full loop)
    log.info("Opening gridMET Zarr ...")
    gm_ds = xr.open_zarr(str(GRIDMET_SOURCE_ZARR))
    all_dates = pd.DatetimeIndex(gm_ds.time.values)

    # Filter to frost-season months
    frost_dates = all_dates[all_dates.month.isin(FROST_MONTHS)]
    log.info("  Frost-season dates: %d (from %s to %s)",
             len(frost_dates), frost_dates[0].date(), frost_dates[-1].date())

    # Open NDVI and build climatology for pre-2013
    ndvi_ds = None
    ndvi_clim = None
    if NDVI_ZARR.exists():
        ndvi_ds = xr.open_zarr(str(NDVI_ZARR))
        log.info("Building NDVI climatology for pre-2013 dates ...")
        ndvi_clim = _build_ndvi_climatology(ndvi_ds)
        log.info("  NDVI climatology: %d weeks", len(ndvi_clim) if ndvi_clim else 0)

    # Pre-allocate arrays (avoid slow list-of-dicts append)
    n_dates = len(frost_dates)
    total_rows = n_dates * n_pts
    all_sample_idx = np.empty(total_rows, dtype=np.int32)
    all_dates = np.empty(total_rows, dtype="datetime64[D]")
    all_tmin = np.full(total_rows, np.nan, dtype=np.float32)
    all_gm_tmin = np.full(total_rows, np.nan, dtype=np.float32)
    all_residual = np.full(total_rows, np.nan, dtype=np.float32)
    all_hours = np.full(total_rows, np.nan, dtype=np.float32)

    # Pre-fill sample_idx (repeats for each date)
    sample_idx_block = np.arange(n_pts, dtype=np.int32)

    write_pos = 0
    dates_processed = 0

    for i, ts in enumerate(frost_dates):
        target_date = ts.date()

        # Load gridMET (with cached ds)
        gridmet = _load_gridmet_for_date(target_date, ds_cache=gm_ds)
        if gridmet is None:
            continue

        # Load NDVI (climatological for pre-2013)
        ndvi = _get_ndvi_for_date(target_date, ndvi_ds, ndvi_clim)

        # Extract gridMET Tmin at sample points
        tmmn_pts = gridmet["tmmn"][rows, cols]

        # Sweep hours_until_sunrise
        best_tmin = np.full(n_pts, np.inf, dtype=np.float32)
        best_residual = np.zeros(n_pts, dtype=np.float32)
        best_hours = np.zeros(n_pts, dtype=np.float32)

        for hrs in hrs_candidates:
            X = _extract_point_features(rows, cols, covariates, gridmet, ndvi, hrs)
            residual_pred = rf.predict(X).astype(np.float32)
            tmin_pred = tmmn_pts + residual_pred

            # Use np.nanless for NaN-safe comparison
            valid = np.isfinite(tmin_pred)
            colder = valid & (tmin_pred < best_tmin)
            best_tmin[colder] = tmin_pred[colder]
            best_residual[colder] = residual_pred[colder]
            best_hours[colder] = hrs

        # Replace inf with NaN for points that never got a valid prediction
        best_tmin[~np.isfinite(best_tmin)] = np.nan

        # Write to pre-allocated arrays
        sl = slice(write_pos, write_pos + n_pts)
        all_sample_idx[sl] = sample_idx_block
        all_dates[sl] = np.datetime64(target_date)
        all_tmin[sl] = best_tmin
        all_gm_tmin[sl] = tmmn_pts
        all_residual[sl] = best_residual
        all_hours[sl] = best_hours
        write_pos += n_pts
        dates_processed += 1

        if (dates_processed) % progress_interval == 0:
            valid_tmin = best_tmin[np.isfinite(best_tmin)]
            mean_t = float(np.mean(valid_tmin)) if len(valid_tmin) > 0 else float("nan")
            log.info("  [%d/%d] %s — mean Tmin: %.1f °C",
                     dates_processed, n_dates, target_date, mean_t)

    gm_ds.close()
    if ndvi_ds is not None:
        ndvi_ds.close()

    # Trim to actual size (some dates may have been skipped)
    df = pd.DataFrame({
        "sample_idx": all_sample_idx[:write_pos],
        "date": pd.to_datetime(all_dates[:write_pos]),
        "predicted_tmin_C": all_tmin[:write_pos],
        "gridmet_tmin_C": all_gm_tmin[:write_pos],
        "lst_residual_C": all_residual[:write_pos],
        "best_hours": all_hours[:write_pos],
    })
    log.info("Prediction complete: %d rows (%d dates × %d points)",
             len(df), dates_processed, n_pts)
    return df


# ── Step 3: Harmonic regression → weekly frost probabilities ─────────────────

def define_frost_weeks():
    """Define 39 frost-season weeks (Sep wk1 through May wk4).

    Returns list of dicts with keys: week_num, label, center_doy, start_doy, end_doy.
    """
    weeks = []
    # Reference: Sep 1 = DOY 244, May 31 = DOY 152 (next year, mapped to 365+152=517)
    # 39 weeks covers Sep 1 through ~May 28.
    start_doy = 244  # Sep 1

    for w in range(39):
        s = start_doy + w * 7
        e = s + 6
        center = s + 3

        # Map back to standard DOY (wrap around year-end)
        center_std = center if center <= 365 else center - 365
        s_std = s if s <= 365 else s - 365
        e_std = e if e <= 365 else e - 365

        # Build label
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        # Approximate month from center DOY
        approx_date = date(2024, 1, 1) + timedelta(days=center_std - 1)
        month_abbr = month_names.get(approx_date.month, f"M{approx_date.month}")
        week_in_month = (approx_date.day - 1) // 7 + 1

        weeks.append({
            "week_num": w + 1,
            "label": f"{month_abbr}-wk{week_in_month}",
            "center_doy": center_std,
            "start_doy": s_std,
            "end_doy": e_std,
        })

    return weeks


def _doy_from_date(d):
    """Day-of-year for a date."""
    return d.timetuple().tm_yday


def _frost_doy(doy):
    """Convert standard DOY to frost-season DOY (Sep 1 = 0)."""
    if doy >= 244:
        return doy - 244
    else:
        return doy + (365 - 244)


def _assign_frost_year(dates):
    """Assign frost year: Sep-Dec → that year, Jan-May → previous year.

    A frost year is labelled by the year of its September (e.g., frost year 2023
    covers Sep 2023 through May 2024).
    """
    months = dates.dt.month
    years = dates.dt.year
    return np.where(months >= 9, years, years - 1)


def _assign_frost_week(dates):
    """Map each date to its frost-season week number (1-30).

    Week 1 starts Sep 1 (DOY 244). Returns 0 for dates outside the 30-week window.
    """
    doys = dates.dt.dayofyear.values
    frost_doys = np.where(doys >= 244, doys - 244, doys + (365 - 244))
    week_nums = frost_doys // 7 + 1
    week_nums[week_nums > 30] = 0
    return week_nums


def compute_weekly_frost_probs(tmin_df, threshold=0.0):
    """Compute weekly frost probabilities per sample point, per frost year.

    Uses empirical frost-day counting with 3-week rolling smoothing.
    Also computes the seasonal Tmin anomaly for each (sample, frost_year).

    Parameters
    ----------
    tmin_df : DataFrame from predict_tmin_at_samples().
    threshold : Temperature threshold in °C (default 0 = frost).

    Returns
    -------
    DataFrame with columns: sample_idx, frost_year, week_num, week_label,
                            center_doy, frost_prob, seasonal_anomaly_C.
    """
    weeks = define_frost_weeks()
    week_lookup = {wk["week_num"]: wk for wk in weeks}

    df = tmin_df.copy()
    dates = pd.to_datetime(df["date"])
    df["frost_year"] = _assign_frost_year(dates)
    df["frost_week"] = _assign_frost_week(dates)
    df["is_frost"] = (df["predicted_tmin_C"] < threshold).astype(np.float32)

    # Drop dates outside the 30-week frost season
    df = df[df["frost_week"] > 0].copy()

    sample_indices = df["sample_idx"].unique()
    frost_years = np.sort(df["frost_year"].unique())
    n_samples = len(sample_indices)
    n_years = len(frost_years)
    log.info("Computing per-year frost probs: %d samples × %d frost years ...",
             n_samples, n_years)

    # ── Compute seasonal Tmin anomaly per (sample, frost_year) ───────────
    # Mean Tmin across all dates in each frost year for each sample
    season_means = (
        df.groupby(["sample_idx", "frost_year"])["predicted_tmin_C"]
        .mean()
        .rename("season_mean_tmin")
    )
    # Long-term mean per sample (climatological baseline)
    sample_clim = (
        season_means.groupby("sample_idx").mean().rename("clim_mean_tmin")
    )
    # Anomaly = season mean - climatological mean
    anomaly_df = season_means.reset_index()
    anomaly_df = anomaly_df.merge(
        sample_clim.reset_index(), on="sample_idx", how="left"
    )
    anomaly_df["seasonal_anomaly_C"] = (
        anomaly_df["season_mean_tmin"] - anomaly_df["clim_mean_tmin"]
    )
    anomaly_lookup = anomaly_df.set_index(
        ["sample_idx", "frost_year"]
    )["seasonal_anomaly_C"].to_dict()

    # ── Compute empirical frost probability per (sample, frost_year, week) ──
    # Count frost days and total days per group
    grouped = (
        df.groupby(["sample_idx", "frost_year", "frost_week"])
        .agg(n_frost=("is_frost", "sum"), n_days=("is_frost", "count"))
        .reset_index()
    )
    grouped["raw_frost_prob"] = grouped["n_frost"] / grouped["n_days"]

    # 3-week rolling smoothing within each (sample, frost_year)
    records = []
    for (sidx, fy), grp in grouped.groupby(["sample_idx", "frost_year"]):
        grp = grp.sort_values("frost_week")
        week_probs = dict(zip(grp["frost_week"], grp["raw_frost_prob"]))
        week_ndays = dict(zip(grp["frost_week"], grp["n_days"]))

        anomaly = anomaly_lookup.get((sidx, fy), 0.0)

        for wk_info in weeks:
            wn = wk_info["week_num"]
            # Weighted average over 3-week window (current ± 1 week)
            total_frost = 0.0
            total_days = 0
            for offset in [-1, 0, 1]:
                neighbor = wn + offset
                if neighbor in week_probs:
                    nd = week_ndays[neighbor]
                    total_frost += week_probs[neighbor] * nd
                    total_days += nd

            prob = total_frost / total_days if total_days > 0 else 0.0

            records.append({
                "sample_idx": sidx,
                "frost_year": fy,
                "week_num": wn,
                "week_label": wk_info["label"],
                "center_doy": wk_info["center_doy"],
                "frost_prob": float(prob),
                "seasonal_anomaly_C": float(anomaly),
            })

    result = pd.DataFrame(records)
    log.info("  Frost probs computed: %d rows (%d points × %d weeks × %d years)",
             len(result), n_samples, len(weeks), n_years)

    # Summary
    mid_winter = result[result["week_num"].between(13, 17)]
    log.info("  Mid-winter (Dec) mean frost prob: %.2f", mid_winter["frost_prob"].mean())
    early_fall = result[result["week_num"].between(1, 4)]
    log.info("  Early fall (Sep) mean frost prob: %.2f", early_fall["frost_prob"].mean())
    log.info("  Seasonal anomaly range: %.2f to %.2f °C",
             result["seasonal_anomaly_C"].min(), result["seasonal_anomaly_C"].max())

    return result


def compute_weekly_frost_probs_climatology(tmin_df, threshold=0.0):
    """Compute climatological (all-years-pooled) frost probabilities.

    This is the original method using harmonic regression. Retained for
    comparison and backward compatibility.

    Parameters
    ----------
    tmin_df : DataFrame from predict_tmin_at_samples().
    threshold : Temperature threshold in °C (default 0 = frost).

    Returns
    -------
    DataFrame with columns: sample_idx, week_num, week_label, center_doy,
                            mu_tmin, sigma_tmin, frost_prob.
    """
    weeks = define_frost_weeks()
    sample_indices = tmin_df["sample_idx"].unique()
    n_samples = len(sample_indices)

    log.info("Fitting harmonic regression for %d sample points ...", n_samples)

    records = []
    for idx in sample_indices:
        sub = tmin_df[tmin_df["sample_idx"] == idx].copy()
        tmin_vals = sub["predicted_tmin_C"].values
        dates = pd.to_datetime(sub["date"])
        doys = dates.dt.dayofyear.values.astype(float)

        # Build design matrix: intercept + 2 harmonics
        omega1 = 2 * np.pi / 365
        omega2 = 4 * np.pi / 365
        X = np.column_stack([
            np.ones(len(doys)),
            np.cos(omega1 * doys),
            np.sin(omega1 * doys),
            np.cos(omega2 * doys),
            np.sin(omega2 * doys),
        ])

        # OLS fit
        beta, residuals, _, _ = np.linalg.lstsq(X, tmin_vals, rcond=None)
        fitted = X @ beta
        n_obs = len(tmin_vals)
        ddof = min(5, max(1, n_obs - 1))
        sigma = np.std(tmin_vals - fitted, ddof=ddof)

        # Compute frost probability for each week
        for wk in weeks:
            d = float(wk["center_doy"])
            mu = (beta[0]
                  + beta[1] * np.cos(omega1 * d)
                  + beta[2] * np.sin(omega1 * d)
                  + beta[3] * np.cos(omega2 * d)
                  + beta[4] * np.sin(omega2 * d))
            prob = float(norm.cdf((threshold - mu) / sigma)) if sigma > 0 else (1.0 if mu < threshold else 0.0)

            records.append({
                "sample_idx": idx,
                "week_num": wk["week_num"],
                "week_label": wk["label"],
                "center_doy": wk["center_doy"],
                "mu_tmin": float(mu),
                "sigma_tmin": float(sigma),
                "frost_prob": prob,
            })

    df = pd.DataFrame(records)
    log.info("  Frost probs computed: %d rows (%d points × %d weeks)",
             len(df), n_samples, len(weeks))

    # Summary
    mid_winter = df[df["week_num"].between(13, 17)]  # ~Dec
    log.info("  Mid-winter (Dec) mean frost prob: %.2f", mid_winter["frost_prob"].mean())
    early_fall = df[df["week_num"].between(1, 4)]  # ~Sep
    log.info("  Early fall (Sep) mean frost prob: %.2f", early_fall["frost_prob"].mean())

    return df


# ── Step 4: Train spatial interpolation model ────────────────────────────────

def train_spatial_model(frost_probs_df, sample_df):
    """Train GBM on logit(frost_prob) ~ covariates + week + seasonal anomaly.

    The seasonal anomaly feature allows the model to learn how frost
    probabilities shift in warm/cold years. At prediction time, substitute
    a seasonal forecast anomaly (e.g., from NMME) or 0.0 for climatology.

    Parameters
    ----------
    frost_probs_df : DataFrame from compute_weekly_frost_probs().
        Must include columns: sample_idx, frost_year, week_num, frost_prob,
        seasonal_anomaly_C.
    sample_df : DataFrame from generate_sample_points() (has covariate values).

    Returns
    -------
    (fitted_model, cv_metrics_dict, feature_names)
    """
    log.info("Training spatial interpolation model (GBM) ...")

    # Merge covariates with frost probs
    merged = frost_probs_df.merge(sample_df, on="sample_idx", how="left")

    # Feature matrix: 16 static covariates + cyclical week encoding + anomaly
    omega = 2 * np.pi / 52
    merged["week_cos"] = np.cos(omega * merged["week_num"])
    merged["week_sin"] = np.sin(omega * merged["week_num"])

    feature_names = STATIC_COV_NAMES + ["week_cos", "week_sin", "seasonal_anomaly_C"]
    X = merged[feature_names].values

    # Response: logit(frost_prob), clamped
    probs = merged["frost_prob"].values.clip(0.001, 0.999)
    y = np.log(probs / (1 - probs))  # logit transform

    # Drop rows with NaN features
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    log.info("  Valid rows: %d (after dropping NaN)", len(y))

    # Subsample for training speed — GBM saturates well before 200k rows
    max_train = 200_000
    rng = np.random.default_rng(42)
    if len(y) > max_train:
        idx = rng.choice(len(y), size=max_train, replace=False)
        X_train = X[idx]
        y_train = y[idx]
        log.info("  Subsampled to %d rows for training (from %d)", max_train, len(y))
    else:
        X_train = X
        y_train = y
    log.info("  Training rows: %d", len(y_train))

    # Fit GBM — use HistGradientBoosting for speed (GPU-like histogram binning)
    from sklearn.ensemble import HistGradientBoostingRegressor
    gbm = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )

    # Cross-validation
    log.info("  Running 5-fold CV ...")
    cv_scores = cross_val_score(gbm, X_train, y_train, cv=5, scoring="r2")
    log.info("  CV R²: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Fit on subsampled data
    gbm.fit(X_train, y_train)

    # Feature importance (permutation-based for HistGBM compatibility)
    try:
        importances = dict(zip(feature_names, gbm.feature_importances_))
    except AttributeError:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(gbm, X_train, y_train, n_repeats=5, random_state=42)
        importances = dict(zip(feature_names, perm.importances_mean))
    top_5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    log.info("  Top 5 features: %s", [(k, f"{v:.3f}") for k, v in top_5])

    cv_metrics = {
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "cv_r2_folds": cv_scores.tolist(),
        "feature_importances": {k: float(v) for k, v in importances.items()},
        "n_training_rows": int(len(y_train)),
    }

    return gbm, cv_metrics, feature_names


# ── Step 5: Full-grid prediction ─────────────────────────────────────────────

def predict_frost_maps(model, feature_names, covariates=None,
                       seasonal_anomaly=0.0):
    """Predict frost probability for all valid pixels for each of 30 weeks.

    Parameters
    ----------
    model : fitted GBM model.
    feature_names : list of feature names matching model training order.
    covariates : dict of covariate name -> 2D array, or None to load.
    seasonal_anomaly : float or array-like
        Seasonal Tmin anomaly in °C. Use 0.0 for climatology, or pass a
        forecast anomaly (e.g., from NMME) for year-specific prediction.
        If array-like, produces maps for each value (ensemble members).

    Returns
    -------
    dict mapping week_num -> 2D frost_prob array (grid_h x grid_w), NaN outside valid.
    If seasonal_anomaly is array-like, returns dict mapping week_num -> 3D array
    (n_ensemble x grid_h x grid_w).
    """
    if covariates is None:
        covariates = _load_covariates()

    grid_h, grid_w = next(iter(covariates.values())).shape

    # Build valid mask
    valid_mask = np.ones((grid_h, grid_w), dtype=bool)
    for name in STATIC_COV_NAMES:
        if name in covariates:
            valid_mask &= np.isfinite(covariates[name])
    valid_rows, valid_cols = np.where(valid_mask)
    flat_idx = np.ravel_multi_index((valid_rows, valid_cols), (grid_h, grid_w))
    n_valid = len(valid_rows)

    # Handle scalar vs ensemble anomaly
    anomalies = np.atleast_1d(seasonal_anomaly).astype(np.float32)
    is_ensemble = len(anomalies) > 1

    if is_ensemble:
        log.info("Predicting frost maps for %d valid pixels × %d ensemble members ...",
                 n_valid, len(anomalies))
    else:
        log.info("Predicting frost maps for %d valid pixels (anomaly=%.2f°C) ...",
                 n_valid, float(anomalies[0]))

    # Extract static covariates at valid pixels
    static_features = np.column_stack([
        covariates[name].ravel()[flat_idx] for name in STATIC_COV_NAMES
    ])

    weeks = define_frost_weeks()
    omega = 2 * np.pi / 52
    maps = {}

    for wk in weeks:
        wn = wk["week_num"]
        week_cos = np.full(n_valid, np.cos(omega * wn), dtype=np.float32)
        week_sin = np.full(n_valid, np.sin(omega * wn), dtype=np.float32)

        if is_ensemble:
            ensemble_maps = []
            for anom in anomalies:
                anom_col = np.full(n_valid, anom, dtype=np.float32)
                X = np.column_stack([static_features, week_cos, week_sin, anom_col])
                logit_pred = model.predict(X)
                frost_prob = 1.0 / (1.0 + np.exp(-logit_pred))
                full_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
                full_map.ravel()[flat_idx] = frost_prob.astype(np.float32)
                ensemble_maps.append(full_map)
            maps[wn] = np.stack(ensemble_maps, axis=0)
            mean_prob = float(np.nanmean(maps[wn]))
        else:
            anom_col = np.full(n_valid, anomalies[0], dtype=np.float32)
            X = np.column_stack([static_features, week_cos, week_sin, anom_col])
            logit_pred = model.predict(X)
            frost_prob = 1.0 / (1.0 + np.exp(-logit_pred))
            full_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
            full_map.ravel()[flat_idx] = frost_prob.astype(np.float32)
            maps[wn] = full_map
            mean_prob = float(np.nanmean(full_map))

        log.info("  Week %02d (%s): mean P(frost) = %.3f",
                 wn, wk["label"], mean_prob)

    return maps


# ── Output I/O ───────────────────────────────────────────────────────────────

def save_frost_maps(maps, output_dir=None):
    """Save frost probability maps as GeoTIFFs.

    Writes:
      - Individual per-week GeoTIFFs
      - Single multi-band GeoTIFF
    """
    if output_dir is None:
        output_dir = FROST_CLIM_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = _get_reference_profile()
    write_profile = profile.copy()
    write_profile.update(dtype="float32", nodata=np.nan, compress="deflate")

    weeks = define_frost_weeks()

    # Individual GeoTIFFs
    for wk in weeks:
        wn = wk["week_num"]
        arr = maps[wn]
        out_path = output_dir / f"frost_prob_week_{wn:02d}.tif"
        write_profile["count"] = 1
        with rasterio.open(out_path, "w", **write_profile) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, wk["label"])

    # Multi-band GeoTIFF
    multi_path = output_dir / "frost_prob_weekly.tif"
    write_profile["count"] = len(weeks)
    with rasterio.open(multi_path, "w", **write_profile) as dst:
        for i, wk in enumerate(weeks):
            dst.write(maps[wk["week_num"]], i + 1)
            dst.set_band_description(i + 1, wk["label"])

    log.info("Saved %d individual + 1 multi-band GeoTIFF → %s", len(weeks), output_dir)


def save_metadata(cv_metrics, sample_df, output_dir=None):
    """Save pipeline metadata as JSON."""
    if output_dir is None:
        output_dir = FROST_CLIM_DIR
    output_dir = Path(output_dir)

    weeks = define_frost_weeks()
    meta = {
        "n_sample_points": len(sample_df),
        "weeks": weeks,
        "cv_metrics": cv_metrics,
        "crs": CRS_UTM,
        "resolution_m": 100,
        "threshold_C": 0.0,
    }
    out_path = output_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Saved metadata → %s", out_path)


# ── Visualization ────────────────────────────────────────────────────────────

def plot_frost_panel(maps, output_dir=None):
    """Plot 5×6 small-multiple grid of weekly frost probability maps."""
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = FIGURES_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    weeks = define_frost_weeks()
    n_weeks = len(weeks)
    ncols = 6
    nrows = (n_weeks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, nrows * 4))
    axes = axes.flatten()

    profile = _get_reference_profile()
    transform = profile["transform"]
    h, w = next(iter(maps.values())).shape
    west = transform.c
    north = transform.f
    east = west + w * transform.a
    south = north + h * transform.e
    extent_km = [west / 1000, east / 1000, south / 1000, north / 1000]

    for i, wk in enumerate(weeks):
        ax = axes[i]
        arr = maps[wk["week_num"]]
        im = ax.imshow(arr, cmap="YlOrRd", vmin=0, vmax=1,
                       extent=extent_km, interpolation="nearest")
        ax.set_title(wk["label"], fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused axes
    for i in range(n_weeks, len(axes)):
        axes[i].set_visible(False)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes[:n_weeks], shrink=0.6, pad=0.02,
                        label="P(Tmin < 0°C)")

    fig.suptitle("Weekly frost probability climatology (100m resolution)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = Path(output_dir) / "frost_climatology_panel.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved panel plot → %s", out_path)
