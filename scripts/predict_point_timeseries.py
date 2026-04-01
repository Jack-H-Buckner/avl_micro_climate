"""Predict nighttime minimum temperature time series at user-specified locations.

Extracts covariates only at the requested lat/lon pixel locations and runs the
RF model for each date in the range — much faster than generating full rasters.

Outputs:
  1. A CSV of daily predicted Tmin at each location.
  2. A time series plot with one line per location.

Usage
-----
    python scripts/predict_point_timeseries.py \
        --locations 35.595,-82.551 35.45,-82.40 \
        --start 2023-10-01 --end 2024-04-30

    # With custom labels:
    python scripts/predict_point_timeseries.py \
        --locations 35.595,-82.551 35.45,-82.40 \
        --labels "Asheville" "South valley" \
        --start 2023-10-01 --end 2024-01-15
"""

import argparse
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
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

# Covariate name lists (mirrors predict_tmin_map.py)
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


# ── Sunrise calculation ─────────────────────────────────────────────────────

def _sunrise_utc(dt, lat: float = CENTER_LAT, lon: float = CENTER_LON):
    """Approximate sunrise time (UTC) using NOAA solar equations."""
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


# ── Lat/lon → pixel index ───────────────────────────────────────────────────

def latlon_to_pixel(lats, lons, dem_path=DEM_100M_PATH):
    """Convert WGS84 lat/lon pairs to (row, col) on the 100m reference grid.

    Returns
    -------
    rows, cols : arrays of int pixel indices
    """
    transformer = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    with rasterio.open(dem_path) as src:
        transform = src.transform
        height, width = src.height, src.width

    rows, cols = [], []
    for x, y in zip(xs, ys):
        col, row = ~transform * (x, y)
        row, col = int(round(row)), int(round(col))
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(
                f"Location ({y:.4f}N UTM, {x:.4f}E UTM) falls outside the "
                f"study area grid ({height}x{width})."
            )
        rows.append(row)
        cols.append(col)

    return np.array(rows), np.array(cols)


# ── Data loaders (full arrays, then index at points) ────────────────────────

def _load_covariates():
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


def _load_gridmet_for_date(target_date):
    """Bilinear-interpolate gridMET variables for one date to the 100m grid."""
    ds = xr.open_zarr(str(GRIDMET_SOURCE_ZARR))
    gm_date = np.datetime64(target_date)
    try:
        day_slice = ds.sel(time=gm_date)
    except KeyError:
        log.warning("No gridMET data for %s — skipping", gm_date)
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

    _NEAREST_GRIDMET_VARS = set()
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


def _load_nearest_ndvi(target_date):
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


# ── Feature extraction at points ────────────────────────────────────────────

def extract_point_features(
    rows, cols,
    covariates, gridmet, ndvi,
    hours_until_sunrise,
):
    """Build the feature matrix for N points at given pixel locations.

    Returns X with shape (N, n_features) in FEATURE_COLS order.
    """
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
            log.warning("Feature %s not available — filling with NaN", feat)
            cols_list.append(np.full(n_pts, np.nan, dtype=np.float32))

    return np.column_stack(cols_list)


# ── Main prediction loop ────────────────────────────────────────────────────

def predict_point_timeseries(
    lats, lons, labels,
    start_date, end_date,
    model_path=DEFAULT_MODEL,
    hrs_candidates=None,
):
    """Predict daily Tmin time series at specified lat/lon locations.

    Returns a DataFrame with columns:
        date, lat, lon, label, predicted_tmin_C, gridmet_tmin_C,
        lst_residual_C, best_hours
    """
    if hrs_candidates is None:
        hrs_candidates = np.arange(0.5, 5.0, 0.25)

    # Resolve pixel locations
    log.info("Mapping %d locations to 100m grid pixels ...", len(lats))
    rows, cols = latlon_to_pixel(np.array(lats), np.array(lons))
    for i, (lat, lon, r, c) in enumerate(zip(lats, lons, rows, cols)):
        log.info("  %s: (%.4f, %.4f) → pixel (%d, %d)", labels[i], lat, lon, r, c)

    # Load model
    log.info("Loading model from %s ...", model_path)
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    # Load static covariates (once)
    log.info("Loading static covariates ...")
    covariates = _load_covariates()

    n_pts = len(lats)
    records = []

    # Iterate over dates
    current = start_date
    n_days = (end_date - start_date).days + 1
    log.info("Predicting for %d dates (%s to %s) ...", n_days, start_date, end_date)

    while current <= end_date:
        # Load dynamic data for this date
        gridmet = _load_gridmet_for_date(current)
        if gridmet is None:
            current += timedelta(days=1)
            continue

        ndvi = _load_nearest_ndvi(current)

        # Extract gridMET Tmin at points
        tmmn_pts = gridmet["tmmn"][rows, cols]

        # Sweep hours_until_sunrise, keep per-point minimum Tmin
        best_tmin = np.full(n_pts, np.inf, dtype=np.float32)
        best_residual = np.zeros(n_pts, dtype=np.float32)
        best_hours = np.zeros(n_pts, dtype=np.float32)

        for hrs in hrs_candidates:
            X = extract_point_features(rows, cols, covariates, gridmet, ndvi, hrs)
            residual_pred = rf.predict(X).astype(np.float32)
            tmin_pred = tmmn_pts + residual_pred

            colder = tmin_pred < best_tmin
            best_tmin[colder] = tmin_pred[colder]
            best_residual[colder] = residual_pred[colder]
            best_hours[colder] = hrs

        for i in range(n_pts):
            records.append({
                "date": current,
                "lat": lats[i],
                "lon": lons[i],
                "label": labels[i],
                "predicted_tmin_C": float(best_tmin[i]),
                "gridmet_tmin_C": float(tmmn_pts[i]),
                "lst_residual_C": float(best_residual[i]),
                "best_hours": float(best_hours[i]),
            })

        log.info("  %s — mean predicted Tmin: %.2f °C", current, float(np.mean(best_tmin)))
        current += timedelta(days=1)

    df = pd.DataFrame(records)
    return df


# ── Output ───────────────────────────────────────────────────────────────────

def save_csv(df, out_path):
    """Save the results DataFrame to CSV."""
    df.to_csv(out_path, index=False, float_format="%.3f")
    log.info("Saved CSV → %s (%d rows)", out_path, len(df))


def plot_timeseries(df, out_path):
    """Generate a time series plot with one line per location."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for label, grp in df.groupby("label"):
        grp = grp.sort_values("date")
        ax.plot(grp["date"], grp["predicted_tmin_C"], label=label, linewidth=1.2)

    # Frost threshold
    ax.axhline(0, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7, label="0 °C (frost)")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Predicted Tmin (°C)", fontsize=12)
    ax.set_title("Predicted nighttime minimum temperature time series", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", out_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_locations(loc_strings):
    """Parse 'lat,lon' strings into separate lists."""
    lats, lons = [], []
    for s in loc_strings:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid location format '{s}' — expected 'lat,lon'")
        lats.append(float(parts[0]))
        lons.append(float(parts[1]))
    return lats, lons


def main():
    parser = argparse.ArgumentParser(
        description="Predict Tmin time series at specific lat/lon locations",
    )
    parser.add_argument(
        "--locations", nargs="+", required=True,
        help="One or more 'lat,lon' pairs (e.g. 35.595,-82.551)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Optional labels for each location (must match number of locations)",
    )
    parser.add_argument(
        "--start", type=str, required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to saved RF model pickle",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: data/output/predictions)",
    )
    args = parser.parse_args()

    lats, lons = parse_locations(args.locations)

    if args.labels:
        if len(args.labels) != len(lats):
            parser.error(f"Got {len(args.labels)} labels but {len(lats)} locations")
        labels = args.labels
    else:
        labels = [f"loc_{i+1} ({lats[i]:.3f},{lons[i]:.3f})" for i in range(len(lats))]

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    df = predict_point_timeseries(
        lats=lats, lons=lons, labels=labels,
        start_date=start_date, end_date=end_date,
        model_path=args.model,
    )

    # Output paths
    out_dir = args.out_dir or PREDICTION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"{args.start}_to_{args.end}"
    csv_path = out_dir / f"point_timeseries_{tag}.csv"
    plot_path = FIGURES_DIR / f"point_timeseries_{tag}.png"

    save_csv(df, csv_path)
    plot_timeseries(df, plot_path)

    # Print summary
    log.info("── Summary ──")
    log.info("  Locations: %d", len(lats))
    log.info("  Date range: %s to %s (%d days)", start_date, end_date,
             (end_date - start_date).days + 1)
    log.info("  Dates with data: %d", df["date"].nunique())
    for label in labels:
        sub = df[df["label"] == label]
        if len(sub) > 0:
            log.info("  %s — mean Tmin: %.1f °C, min: %.1f °C, max: %.1f °C",
                     label, sub["predicted_tmin_C"].mean(),
                     sub["predicted_tmin_C"].min(), sub["predicted_tmin_C"].max())


if __name__ == "__main__":
    main()
