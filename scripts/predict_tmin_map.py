"""Predict high-resolution nighttime minimum temperature for a given date.

Loads the trained RF model and all covariates, then predicts the LST residual
for every 100 m pixel.  The ``hours_until_sunrise`` predictor is swept over a
range of candidate values and the value that yields the lowest mean predicted
temperature across the image is selected — approximating the timing of the
overnight minimum.

Final output:  predicted_Tmin = gridMET_Tmin + predicted_residual

Produces:
  1. A GeoTIFF of predicted Tmin at 100 m (EPSG:32617).
  2. A publication-quality heatmap figure.

Usage
-----
    python scripts/predict_tmin_map.py --date 2024-01-15
    python scripts/predict_tmin_map.py --date 2024-01-15 --model data/output/models/rf_benchmark_hrs4.pkl
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CENTER_LAT,
    CENTER_LON,
    CRS_UTM,
    COVARIATES_DIR,
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
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Static covariate names
TERRAIN_NAMES = [
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi_300m", "tpi_1000m", "curvature", "twi",
]
LAND_SURFACE_NAMES = [
    "impervious_pct", "tree_canopy_pct", "sky_view_factor",
    "dist_to_water_m", "is_forest", "is_developed",
    "is_agriculture", "is_water",
]
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


# ── Sunrise calculation (from prepare_training) ─────────────────────────────

def _sunrise_utc(date, lat: float = CENTER_LAT, lon: float = CENTER_LON) -> datetime:
    """Approximate sunrise time (UTC) using NOAA solar equations."""
    from math import acos, cos, radians, sin, tan, degrees

    doy = date.timetuple().tm_yday
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
    return datetime(date.year, date.month, date.day, h, m, tzinfo=timezone.utc)


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_covariates() -> dict[str, np.ndarray]:
    """Load all static covariates (terrain + land surface + ECOSTRESS composites)."""
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


# gridMET variables where nearest-neighbour is more appropriate than bilinear
_NEAREST_GRIDMET_VARS = set()  # all variables use cubic spline


def _load_gridmet_for_date(target_date) -> dict[str, np.ndarray] | None:
    """Bilinear-interpolate gridMET variables for one date to the 100 m grid.

    Reads from the native ~4 km WGS84 Zarr and reprojects on-the-fly,
    eliminating the need for a pre-computed 100 m Zarr.
    """
    ds = xr.open_zarr(str(GRIDMET_SOURCE_ZARR))
    gm_date = np.datetime64(target_date)
    try:
        day_slice = ds.sel(time=gm_date)
    except KeyError:
        log.error("No gridMET data for %s", gm_date)
        ds.close()
        return None

    # Build source transform from lat/lon
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

    # Read reference grid
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
        method = Resampling.nearest if var in _NEAREST_GRIDMET_VARS else Resampling.cubic_spline
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs="EPSG:4326",
            dst_transform=dst_transform, dst_crs=CRS_UTM,
            resampling=method, src_nodata=np.nan, dst_nodata=np.nan,
        )
        gm[var] = dst

    ds.close()
    return gm


def _load_nearest_ndvi(target_date) -> np.ndarray | None:
    """Load the NDVI composite nearest to target_date."""
    if not NDVI_ZARR.exists():
        log.warning("NDVI Zarr not found — NDVI will be NaN.")
        return None
    ds = xr.open_zarr(str(NDVI_ZARR))
    target = np.datetime64(target_date)
    diffs = np.abs(ds.time.values - target)
    nearest_idx = int(np.argmin(diffs))
    arr = ds["ndvi"].isel(time=nearest_idx).values.astype(np.float32)
    ds.close()
    return arr


def _get_reference_profile() -> dict:
    """Read the raster profile from the elevation GeoTIFF for output writing."""
    with rasterio.open(COVARIATES_DIR / "elevation.tif") as src:
        return dict(src.profile)


# ── Prediction ───────────────────────────────────────────────────────────────

def build_feature_array(
    covariates: dict[str, np.ndarray],
    gridmet: dict[str, np.ndarray],
    ndvi: np.ndarray | None,
    hours_until_sunrise: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Assemble the feature matrix for all valid pixels.

    Returns X with shape (n_valid_pixels, n_features) in FEATURE_COLS order.
    """
    n_valid = valid_mask.sum()
    flat_idx = np.where(valid_mask.ravel())[0]

    cols = []
    for feat in FEATURE_COLS:
        if feat == "hours_until_sunrise":
            cols.append(np.full(n_valid, hours_until_sunrise, dtype=np.float32))
        elif feat == "dist_to_removed_m":
            # No cloud filtering for prediction — set to large distance (image diagonal)
            grid_h, grid_w = valid_mask.shape
            diag_m = np.sqrt((grid_h * 100.0) ** 2 + (grid_w * 100.0) ** 2)
            cols.append(np.full(n_valid, diag_m, dtype=np.float32))
        elif feat == "fraction_scene_removed":
            # No cloud filtering for prediction — set to 0
            cols.append(np.zeros(n_valid, dtype=np.float32))
        elif feat == "ndvi":
            if ndvi is not None:
                cols.append(ndvi.ravel()[flat_idx].astype(np.float32))
            else:
                cols.append(np.full(n_valid, np.nan, dtype=np.float32))
        elif feat in gridmet:
            cols.append(gridmet[feat].ravel()[flat_idx].astype(np.float32))
        elif feat in covariates:
            cols.append(covariates[feat].ravel()[flat_idx].astype(np.float32))
        else:
            log.warning("Feature %s not available — filling with NaN", feat)
            cols.append(np.full(n_valid, np.nan, dtype=np.float32))

    return np.column_stack(cols)


def predict_tmin_map(
    model_path: Path,
    target_date,
    hrs_candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Predict the full-image Tmin map for a given date.

    Parameters
    ----------
    model_path : Path to saved RF model pickle.
    target_date : date object for gridMET lookup.
    hrs_candidates : Array of hours_until_sunrise values to sweep.
        Default: np.arange(0.5, 5.0, 0.25).

    Returns
    -------
    (tmin_map, residual_map, best_hours, metadata)
        tmin_map : 2-D array of predicted Tmin (°C), NaN where invalid.
        residual_map : 2-D array of predicted LST residual (°C).
        best_hours : optimal hours_until_sunrise value.
        metadata : dict with grid info and prediction stats.
    """
    if hrs_candidates is None:
        hrs_candidates = np.arange(0.5, 5.0, 0.25)

    # Load model
    log.info("Loading model from %s ...", model_path)
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    # Load data
    log.info("Loading static covariates ...")
    covariates = _load_covariates()
    grid_h, grid_w = next(iter(covariates.values())).shape
    log.info("  Grid: %d × %d", grid_h, grid_w)

    log.info("Loading gridMET for %s ...", target_date)
    gridmet = _load_gridmet_for_date(target_date)
    if gridmet is None:
        raise ValueError(f"No gridMET data for {target_date}")

    log.info("Loading NDVI ...")
    ndvi = _load_nearest_ndvi(target_date)

    # Build valid pixel mask (all covariates + gridMET finite)
    valid_mask = np.ones((grid_h, grid_w), dtype=bool)
    for arr in covariates.values():
        valid_mask &= np.isfinite(arr)
    for var in GRIDMET_VARS:
        valid_mask &= np.isfinite(gridmet[var])
    n_valid = valid_mask.sum()
    log.info("Valid pixels: %d / %d (%.1f%%)", n_valid, grid_h * grid_w,
             100 * n_valid / (grid_h * grid_w))

    # Sweep hours_until_sunrise and keep the per-pixel minimum Tmin
    # across all candidate values.  Each pixel gets its own coldest
    # predicted temperature rather than sharing a single "best" hour.
    log.info("Sweeping hours_until_sunrise over %d candidates [%.1f – %.1f] ...",
             len(hrs_candidates), hrs_candidates[0], hrs_candidates[-1])

    flat_idx = np.where(valid_mask.ravel())[0]
    tmmn_valid = gridmet["tmmn"].ravel()[flat_idx]

    # Per-pixel tracking: best (lowest) Tmin and corresponding residual & hour
    best_tmin_per_pixel = np.full(n_valid, np.inf, dtype=np.float32)
    best_residual_per_pixel = np.zeros(n_valid, dtype=np.float32)
    best_hours_per_pixel = np.zeros(n_valid, dtype=np.float32)

    for hrs in hrs_candidates:
        X = build_feature_array(covariates, gridmet, ndvi, hrs, valid_mask)
        residual_pred = rf.predict(X).astype(np.float32)
        tmin_pred = tmmn_valid + residual_pred

        colder = tmin_pred < best_tmin_per_pixel
        best_tmin_per_pixel[colder] = tmin_pred[colder]
        best_residual_per_pixel[colder] = residual_pred[colder]
        best_hours_per_pixel[colder] = hrs

        log.info("  hours=%.2f → mean Tmin=%.2f °C (pixels updated: %d)",
                 hrs, float(np.mean(tmin_pred)), int(colder.sum()))

    log.info("Per-pixel min Tmin: mean=%.2f °C, hours range=[%.2f – %.2f]",
             float(np.mean(best_tmin_per_pixel)),
             float(best_hours_per_pixel.min()),
             float(best_hours_per_pixel.max()))

    # Assemble full-image maps
    residual_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    residual_map.ravel()[flat_idx] = best_residual_per_pixel

    tmin_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    tmin_map.ravel()[flat_idx] = best_tmin_per_pixel

    best_hours_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    best_hours_map.ravel()[flat_idx] = best_hours_per_pixel

    # Sunrise time for reference
    sr = _sunrise_utc(target_date)

    metadata = {
        "date": str(target_date),
        "hours_range": f"{float(best_hours_per_pixel.min()):.2f} – {float(best_hours_per_pixel.max()):.2f}",
        "sunrise_utc": sr.strftime("%H:%M UTC"),
        "mean_tmin": float(np.nanmean(tmin_map)),
        "min_tmin": float(np.nanmin(tmin_map)),
        "max_tmin": float(np.nanmax(tmin_map)),
        "tmin_range": float(np.nanmax(tmin_map) - np.nanmin(tmin_map)),
        "mean_gridmet_tmin": float(np.nanmean(gridmet["tmmn"])),
        "n_valid_pixels": int(n_valid),
        "grid_shape": (grid_h, grid_w),
    }

    return tmin_map, residual_map, best_hours_map, metadata


# ── Output ───────────────────────────────────────────────────────────────────

def save_geotiff(arr: np.ndarray, out_path: Path, profile: dict):
    """Write a 2-D float32 array as a GeoTIFF."""
    write_profile = profile.copy()
    write_profile.update(dtype="float32", nodata=np.nan, compress="deflate", count=1)
    with rasterio.open(out_path, "w", **write_profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    log.info("Saved GeoTIFF → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def plot_tmin_heatmap(
    tmin_map: np.ndarray,
    metadata: dict,
    profile: dict,
    out_path: Path,
):
    """Generate a publication-quality heatmap of predicted Tmin."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Compute extent in km from the raster transform
    transform = profile["transform"]
    h, w = tmin_map.shape
    west = transform.c
    north = transform.f
    east = west + w * transform.a
    south = north + h * transform.e  # e is negative

    # Convert UTM metres to km for axis labels
    extent_km = [west / 1000, east / 1000, south / 1000, north / 1000]

    im = ax.imshow(
        tmin_map, cmap="RdYlBu_r", extent=extent_km,
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax, label="Predicted Tmin (°C)", shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Easting (km)", fontsize=12)
    ax.set_ylabel("Northing (km)", fontsize=12)
    ax.tick_params(labelsize=10)

    date_str = metadata["date"]
    mean_t = metadata["mean_tmin"]
    min_t = metadata["min_tmin"]
    max_t = metadata["max_tmin"]
    t_range = metadata["tmin_range"]
    hrs_range = metadata["hours_range"]
    sr = metadata["sunrise_utc"]
    gm_tmin = metadata["mean_gridmet_tmin"]

    ax.set_title(
        f"Predicted nighttime minimum temperature — {date_str}\n"
        f"100 m resolution, Buncombe County, NC\n"
        f"Mean: {mean_t:.1f} °C  |  Range: {min_t:.1f} to {max_t:.1f} °C  "
        f"(Δ{t_range:.1f} °C)\n"
        f"Per-pixel min over hours_until_sunrise: {hrs_range} h  |  "
        f"Sunrise: {sr}  |  gridMET Tmin: {gm_tmin:.1f} °C",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s", out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(date_str: str, model_path: Path = DEFAULT_MODEL):
    """Full prediction pipeline for one date."""
    from datetime import date as date_type

    target_date = date_type.fromisoformat(date_str)

    tmin_map, residual_map, best_hours_map, metadata = predict_tmin_map(
        model_path=model_path,
        target_date=target_date,
    )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    profile = _get_reference_profile()

    # Save GeoTIFFs
    save_geotiff(tmin_map, PREDICTION_DIR / f"tmin_predicted_{date_str}.tif", profile)
    save_geotiff(residual_map, PREDICTION_DIR / f"lst_residual_predicted_{date_str}.tif", profile)
    save_geotiff(best_hours_map, PREDICTION_DIR / f"best_hours_{date_str}.tif", profile)

    # Heatmap
    plot_tmin_heatmap(tmin_map, metadata, profile, FIGURES_DIR / f"tmin_heatmap_{date_str}.png")

    # Print summary
    log.info("── Prediction summary ──")
    for k, v in metadata.items():
        log.info("  %s: %s", k, v)

    return tmin_map, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Predict high-resolution nighttime Tmin map for a given date",
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="Target date (YYYY-MM-DD), e.g. 2024-01-15",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to saved RF model pickle",
    )
    args = parser.parse_args()
    run(date_str=args.date, model_path=args.model)


if __name__ == "__main__":
    main()
