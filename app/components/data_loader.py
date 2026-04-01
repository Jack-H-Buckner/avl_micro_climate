"""Load pre-computed last-frost data and reproject to WGS84."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import OUTPUT_DIR, COVARIATES_DIR, BBOX_WGS84

FROST_CLIM_DIR = OUTPUT_DIR / "frost_climatology"


# ── Week metadata ────────────────────────────────────────────────────────────

def define_frost_weeks():
    """Define 39 frost-season weeks (Sep wk1 through May wk4)."""
    weeks = []
    start_doy = 244  # Sep 1
    for w in range(39):
        s = start_doy + w * 7
        e = s + 6
        center = s + 3
        center_std = center if center <= 365 else center - 365
        s_std = s if s <= 365 else s - 365
        e_std = e if e <= 365 else e - 365
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
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


def week_num_to_date_label(week_num):
    """Convert integer frost-season week number to approximate date string."""
    weeks = define_frost_weeks()
    wk = next(w for w in weeks if w["week_num"] == week_num)
    doy = wk["center_doy"]
    approx = date(2024, 1, 1) + timedelta(days=doy - 1)
    return approx.strftime("%b %d")


def fractional_week_to_date_label(fw):
    """Convert a fractional frost-season week to a date string."""
    weeks = define_frost_weeks()
    wn_floor = int(np.floor(fw))
    frac = fw - wn_floor
    wn_floor = max(1, min(wn_floor, len(weeks)))
    wn_ceil = min(wn_floor + 1, len(weeks))
    doy_floor = weeks[wn_floor - 1]["center_doy"]
    doy_ceil = weeks[wn_ceil - 1]["center_doy"]
    if doy_ceil < doy_floor:
        doy_ceil += 365
    doy = doy_floor + frac * (doy_ceil - doy_floor)
    if doy > 365:
        doy -= 365
    approx = date(2024, 1, 1) + timedelta(days=int(doy) - 1)
    return approx.strftime("%b %d")


# ── Load pre-computed data ───────────────────────────────────────────────────

def _get_frost_profile():
    """Get rasterio profile matching the prediction grid.

    Uses the elevation covariate (always regenerated with the current bbox)
    as the authoritative grid definition.  Falls back to frost probability
    TIFs only if the elevation file is missing.
    """
    elev_path = COVARIATES_DIR / "elevation.tif"
    if elev_path.exists():
        with rasterio.open(elev_path) as src:
            return dict(src.profile)
    for wn in range(1, 40):
        path = FROST_CLIM_DIR / f"frost_prob_week_{wn:02d}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                return dict(src.profile)
    return None


def _adjust_profile_to_grid(profile, grid_h, grid_w):
    """Rescale the profile transform so it matches the actual grid dimensions."""
    if profile is None:
        return None
    t = profile["transform"]
    orig_h = profile["height"]
    orig_w = profile["width"]
    if orig_h == grid_h and orig_w == grid_w:
        return profile
    from rasterio.transform import Affine
    scale_x = (orig_w * t.a) / grid_w
    scale_y = (orig_h * t.e) / grid_h
    new_transform = Affine(scale_x, t.b, t.c, t.d, scale_y, t.f)
    profile = dict(profile)
    profile["transform"] = new_transform
    profile["height"] = grid_h
    profile["width"] = grid_w
    return profile


def load_last_frost_data():
    """Load pre-computed last-frost date grids.

    Prefers the sub-weekly GBM version (last_frost_dates.npz) if available,
    falls back to weekly version (last_frost_cumsum.npz).

    Returns
    -------
    data : dict with keys depending on which file was loaded:
        GBM version:  'last_frost_grids' (n_thresh, H, W), 'thresholds' (1D)
        Weekly version: 'cumsum_grid' (n_steps, H, W), 'fractional_weeks' (1D)
    valid_mask : (H, W) bool
    profile : rasterio profile
    mode : str, 'gbm' or 'weekly'
    elev_valid_mask : (H, W) bool or None
        True where elevation is within the GBM training range.
        None when the mask is not available (older npz files).
    """
    profile = _get_frost_profile()

    # Prefer lightweight 400m version (for deploy), fall back to full-res
    gbm_path = FROST_CLIM_DIR / "last_frost_dates_400m.npz"
    if not gbm_path.exists():
        gbm_path = FROST_CLIM_DIR / "last_frost_dates.npz"
    if gbm_path.exists():
        d = np.load(gbm_path)
        elev_mask = d["elev_valid_mask"] if "elev_valid_mask" in d else None
        grids = d["last_frost_grids"]
        # Update profile transform to match actual grid dimensions
        profile = _adjust_profile_to_grid(profile, grids.shape[1], grids.shape[2])
        return {
            "last_frost_grids": grids,
            "thresholds": d["thresholds"],
        }, d["valid_mask"], profile, "gbm", elev_mask

    # Fall back to weekly version
    weekly_path = FROST_CLIM_DIR / "last_frost_cumsum.npz"
    if weekly_path.exists():
        d = np.load(weekly_path)
        elev_mask = d["elev_valid_mask"] if "elev_valid_mask" in d else None
        return {
            "cumsum_grid": d["cumsum_grid"],
            "fractional_weeks": d["fractional_weeks"],
        }, d["valid_mask"], profile, "weekly", elev_mask

    raise FileNotFoundError(
        "No pre-computed last-frost data found. Run:\n"
        "  python scripts/precompute_last_frost.py"
    )


def _fill_nearest(array, max_dist_px=5):
    """Fill small interior NaN gaps with the nearest valid pixel.

    Only fills NaN pixels whose nearest valid neighbour is within
    *max_dist_px* pixels.  Larger gaps (borders, water bodies, elevation-
    excluded zones) are left as NaN so they don't bleed into the bilinear
    reprojection.
    """
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(array)
    if not np.any(mask):
        return array
    dist, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = array.copy()
    fill_mask = mask & (dist <= max_dist_px)
    filled[fill_mask] = array[nearest_idx[0][fill_mask], nearest_idx[1][fill_mask]]
    return filled


def last_frost_date_for_threshold(data, mode, threshold):
    """Get the last-frost date grid for a given threshold.

    NaN pixels are filled with nearest-neighbor values so the bilinear
    reprojection gets a complete field with no holes.

    Returns (H, W) array of fractional week numbers.
    """
    if mode == "gbm":
        grids = data["last_frost_grids"]
        thresholds = data["thresholds"]
        idx = np.argmin(np.abs(thresholds - threshold))
        return _fill_nearest(grids[idx])
    else:
        # Weekly cumsum mode
        cumsum_grid = data["cumsum_grid"]
        fractional_weeks = data["fractional_weeks"]
        n_steps = cumsum_grid.shape[0]
        grid_h, grid_w = cumsum_grid.shape[1], cumsum_grid.shape[2]
        result = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        for t in range(n_steps):
            crosses = (cumsum_grid[t] >= threshold) & np.isnan(result)
            result[crosses] = fractional_weeks[t]
        return _fill_nearest(result)


def extract_frost_timeseries_at_pixel(data, mode, row, col):
    """Extract the cumulative last-frost time series at a pixel.

    Returns list of (week_num_or_frac, date_label, cumulative_prob).
    """
    if mode == "gbm":
        grids = data["last_frost_grids"]
        thresholds = data["thresholds"]
        # For each threshold, get the last-frost week at this pixel
        # Build a cumulative curve: for a set of time steps, find what
        # fraction of thresholds have been crossed
        weeks = define_frost_weeks()
        timeseries = []
        for wk in weeks:
            wn = wk["week_num"]
            # What fraction of thresholds have last-frost <= this week?
            fw_at_pixel = grids[:, row, col]  # fractional weeks for each threshold
            cum_prob = float(np.nanmean(fw_at_pixel <= wn))
            label = week_num_to_date_label(wn)
            timeseries.append((wn, label, cum_prob))
        return timeseries
    else:
        cumsum_grid = data["cumsum_grid"]
        fractional_weeks = data["fractional_weeks"]
        cumsum_at_pixel = cumsum_grid[:, row, col]
        if np.all(np.isnan(cumsum_at_pixel)):
            return None
        weeks = define_frost_weeks()
        timeseries = []
        for wk in weeks:
            wn = wk["week_num"]
            mask = (fractional_weeks >= wn - 0.5) & (fractional_weeks < wn + 0.5)
            if np.any(mask):
                idx = np.where(mask)[0]
                mid = idx[len(idx) // 2]
                cum_prob = float(cumsum_at_pixel[mid])
            else:
                cum_prob = None
            label = week_num_to_date_label(wn)
            timeseries.append((wn, label, cum_prob))
        return timeseries


# ── Reprojection ─────────────────────────────────────────────────────────────

def reproject_elev_mask_to_wgs84(elev_valid_mask, src_profile, dst_resolution=1000):
    """Reproject the boolean elevation validity mask to WGS84.

    Uses nearest-neighbor so the mask stays crisp (no partial transparency).
    Returns (H, W) bool array and bounds, same grid as reproject_to_wgs84.
    """
    src_transform = src_profile["transform"]

    west, east = BBOX_WGS84["west"], BBOX_WGS84["east"]
    south, north = BBOX_WGS84["south"], BBOX_WGS84["north"]

    lon_range = east - west
    lat_range = north - south
    if lon_range >= lat_range:
        dst_w = dst_resolution
        dst_h = int(round(dst_resolution * lat_range / lon_range))
    else:
        dst_h = dst_resolution
        dst_w = int(round(dst_resolution * lon_range / lat_range))

    dst_lons = np.linspace(west, east, dst_w)
    dst_lats = np.linspace(north, south, dst_h)
    lon_grid, lat_grid = np.meshgrid(dst_lons, dst_lats)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    x_utm, y_utm = transformer.transform(lon_grid, lat_grid)

    col_src = (x_utm - src_transform.c) / src_transform.a
    row_src = (y_utm - src_transform.f) / src_transform.e

    dst_mask = map_coordinates(
        elev_valid_mask.astype(np.float32), [row_src, col_src],
        order=0, mode="constant", cval=0.0,
    )
    bounds = [[south, west], [north, east]]
    return dst_mask > 0.5, bounds


def reproject_to_wgs84(array, src_profile, dst_resolution=1000):
    """Reproject a 2D array from UTM 17N to EPSG:4326 with bilinear interpolation."""
    src_transform = src_profile["transform"]

    west = BBOX_WGS84["west"]
    east = BBOX_WGS84["east"]
    south = BBOX_WGS84["south"]
    north = BBOX_WGS84["north"]

    lon_range = east - west
    lat_range = north - south
    if lon_range >= lat_range:
        dst_w = dst_resolution
        dst_h = int(round(dst_resolution * lat_range / lon_range))
    else:
        dst_h = dst_resolution
        dst_w = int(round(dst_resolution * lon_range / lat_range))

    dst_lons = np.linspace(west, east, dst_w)
    dst_lats = np.linspace(north, south, dst_h)
    lon_grid, lat_grid = np.meshgrid(dst_lons, dst_lats)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    x_utm, y_utm = transformer.transform(lon_grid, lat_grid)

    col_src = (x_utm - src_transform.c) / src_transform.a
    row_src = (y_utm - src_transform.f) / src_transform.e

    finite_mask = np.isfinite(array)
    src_filled = np.where(finite_mask, array, -9999.0)

    # Bilinear interpolation (order=1) for smooth rendering
    dst_values = map_coordinates(
        src_filled, [row_src, col_src], order=1, mode="constant", cval=-9999.0,
    )
    # Validity mask still uses nearest-neighbor to keep sharp edges
    dst_valid = map_coordinates(
        finite_mask.astype(np.float32), [row_src, col_src],
        order=0, mode="constant", cval=0.0,
    )

    dst_array = np.where(dst_valid > 0.5, dst_values, np.nan).astype(np.float32)
    bounds = [[south, west], [north, east]]
    return dst_array, bounds
