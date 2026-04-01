"""Fetch auxiliary meteorological variables from gridMET via OpenDAP.

Downloads the full suite of variables needed for nocturnal cooling and
frost modeling: temperature (min/max), wind speed, humidity, VPD,
shortwave radiation, precipitation, and minimum relative humidity.

After download the script clips to the study area, converts units,
computes derived fields (diurnal range, previous-day lags), and stores
the result as a single Zarr archive.

Source: http://www.climatologylab.org/gridmet.html
OpenDAP: http://thredds.northwestknowledge.net:8080/thredds/dodsC/

gridMET day definition: 24 h ending 12:00 UTC (07:00 EST).
  - tmmn on date D captures the overnight minimum through morning of D
  - tmmx on date D captures the afternoon max from D-1 through midday D
  - For nighttime ECOSTRESS passes, use tmmx and srad from date D-1
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, GRIDMET_DIR, RAW_GRIDMET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
OPENDAP_BASE = (
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
    "agg_met_{var}_1979_CurrentYear_CONUS.nc"
)

# Map from gridMET short code → internal variable name in the NetCDF file
VARIABLES = {
    "tmmn": "daily_minimum_temperature",   # K
    "tmmx": "daily_maximum_temperature",   # K
    "vs":   "daily_mean_wind_speed",       # m/s
    "sph":  "daily_mean_specific_humidity", # kg/kg
    "vpd":  "daily_mean_vapor_pressure_deficit",  # kPa
    "srad": "daily_mean_shortwave_radiation_at_surface",  # W/m²
    "pr":   "precipitation_amount",        # mm
    "rmin": "daily_minimum_relative_humidity",  # %
}

# Variables stored in Kelvin that need conversion to °C
KELVIN_VARS = {"tmmn", "tmmx"}

# Frost season months (Sep–May)
FROST_MONTHS = [1, 2, 3, 4, 5, 9, 10, 11, 12]

# Study period — aligned with ECOSTRESS availability (2018–present)
START_YEAR = 2000
END_YEAR = 2025

# Output path
OUTPUT_PATH = GRIDMET_DIR / "gridmet_frost_season.zarr"

# Spatial buffer for interpolation safety (~5 km)
SPATIAL_BUFFER = 0.05


def _opendap_url(var_code: str) -> str:
    """Build the OpenDAP URL for a given gridMET variable."""
    return OPENDAP_BASE.format(var=var_code)


def _subset_spatial(ds: xr.Dataset) -> xr.Dataset:
    """Clip dataset to study area bounding box with buffer.

    gridMET latitudes are descending, so slice north → south.
    """
    return ds.sel(
        lon=slice(
            BBOX_WGS84["west"] - SPATIAL_BUFFER,
            BBOX_WGS84["east"] + SPATIAL_BUFFER,
        ),
        lat=slice(
            BBOX_WGS84["north"] + SPATIAL_BUFFER,
            BBOX_WGS84["south"] - SPATIAL_BUFFER,
        ),
    )


def fetch_variable(var_code: str) -> xr.DataArray:
    """Download a single gridMET variable via OpenDAP, subset and load.

    Parameters
    ----------
    var_code : str
        gridMET short code (e.g. ``"tmmn"``, ``"vs"``).

    Returns
    -------
    xr.DataArray with dimensions (day, lat, lon).
    """
    nc_name = VARIABLES[var_code]
    url = _opendap_url(var_code)
    log.info("Opening %s via OpenDAP…", var_code)

    ds = xr.open_dataset(url, engine="netcdf4")
    ds = _subset_spatial(ds)
    ds = ds.sel(day=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

    log.info(
        "  %s grid: lon=%d, lat=%d, days=%d",
        var_code, ds.sizes["lon"], ds.sizes["lat"], ds.sizes["day"],
    )

    # Load into memory — clipped study area is small (~5–15 MB per variable)
    log.info("  Downloading %s…", var_code)
    da = ds[nc_name].load()
    ds.close()

    # Filter to frost-season months
    da = da.sel(day=da["day"].dt.month.isin(FROST_MONTHS))
    log.info("  After frost-month filter: %d days", da.sizes["day"])

    # Unit conversion for temperature variables
    if var_code in KELVIN_VARS:
        da = da - 273.15
        da.attrs["units"] = "degC"
        log.info("  Converted %s from K → °C", var_code)

    # Rename to short code
    da.name = var_code
    return da


def fetch_all_variables() -> xr.Dataset:
    """Download all gridMET variables and merge into a single Dataset."""
    arrays = {}
    for var_code in VARIABLES:
        arrays[var_code] = fetch_variable(var_code)

    # Merge — all share the same (day, lat, lon) grid
    ds = xr.Dataset(arrays)

    # Rename time dimension for consistency
    ds = ds.rename({"day": "time"})

    return ds


def compute_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    """Add derived fields useful for frost modeling.

    - diurnal_range: tmmx - tmmn (°C) — large range signals clear radiative nights
    - tmmx_prev: previous day's tmmx (afternoon before the night)
    - srad_prev: previous day's shortwave radiation
    """
    log.info("Computing derived variables…")

    # Diurnal temperature range
    ds["diurnal_range"] = ds["tmmx"] - ds["tmmn"]
    ds["diurnal_range"].attrs = {"units": "degC", "long_name": "diurnal temperature range"}

    # Previous-day lags (shift forward by 1 time step)
    ds["tmmx_prev"] = ds["tmmx"].shift(time=1)
    ds["tmmx_prev"].attrs = {"units": "degC", "long_name": "previous day max temperature"}

    ds["srad_prev"] = ds["srad"].shift(time=1)
    ds["srad_prev"].attrs = {"units": "W/m2", "long_name": "previous day shortwave radiation"}

    log.info("  Added: diurnal_range, tmmx_prev, srad_prev")
    return ds


def save_zarr(ds: xr.Dataset) -> Path:
    """Save the combined dataset as a Zarr store."""
    GRIDMET_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    # Chunk by time for efficient date-based lookups (requires dask)
    try:
        ds = ds.chunk({"time": 90, "lat": -1, "lon": -1})
    except ImportError:
        log.info("dask not installed — saving without explicit chunking")
    ds.to_zarr(OUTPUT_PATH, mode="w")
    log.info("Saved Zarr → %s", OUTPUT_PATH)
    return OUTPUT_PATH


def run() -> Path:
    """Execute the full gridMET auxiliary variable download pipeline."""
    ds = fetch_all_variables()
    ds = compute_derived_variables(ds)

    # Summary
    log.info("── Dataset summary ──")
    log.info("  Variables: %s", list(ds.data_vars))
    log.info("  Shape: %s", dict(ds.sizes))
    log.info("  Lon range: %.3f to %.3f", float(ds.lon.min()), float(ds.lon.max()))
    log.info("  Lat range: %.3f to %.3f", float(ds.lat.min()), float(ds.lat.max()))
    log.info("  Time range: %s to %s",
             str(ds.time.values[0])[:10], str(ds.time.values[-1])[:10])

    for var in ["tmmn", "tmmx", "vs", "sph", "vpd", "srad", "pr", "rmin"]:
        vmin = float(ds[var].min())
        vmax = float(ds[var].max())
        log.info("  %s range: %.3f to %.3f", var, vmin, vmax)

    out = save_zarr(ds)
    ds.close()
    return out


if __name__ == "__main__":
    run()
