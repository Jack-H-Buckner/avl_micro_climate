"""Fetch gridMET daily Tmin via OpenDAP, clip to study area, save as Zarr.

gridMET provides daily meteorological data at ~4 km (1/24°) resolution
over CONUS from 1979–present.  The OpenDAP interface allows server-side
spatial and temporal subsetting so we only download the study area.

Source: http://www.climatologylab.org/gridmet.html
OpenDAP: http://thredds.northwestknowledge.net:8080/thredds/dodsC/
"""

import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, GRIDDED_TMIN_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
GRIDMET_OPENDAP_URL = (
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
    "agg_met_tmmn_1979_CurrentYear_CONUS.nc"
)
TMIN_VAR = "daily_minimum_temperature"

# Frost season months
FROST_MONTHS = [1, 2, 3, 4, 5, 9, 10, 11, 12]

# Study period
START_YEAR = 1991
END_YEAR = 2025

OUTPUT_PATH = GRIDDED_TMIN_DIR / "gridmet_tmin_daily.zarr"


def fetch_gridmet_tmin() -> xr.Dataset:
    """Open the gridMET aggregated Tmin dataset via OpenDAP and subset
    to the study area bounding box and frost-season dates.

    Returns
    -------
    xr.Dataset clipped to study area with Tmin in °C.
    """
    log.info("Opening gridMET Tmin via OpenDAP…")
    ds = xr.open_dataset(GRIDMET_OPENDAP_URL, engine="netcdf4")

    # Spatial subset (small buffer for interpolation safety)
    # gridMET latitudes are descending, so slice north → south
    buf = 0.05  # ~5 km buffer
    ds = ds.sel(
        lon=slice(BBOX_WGS84["west"] - buf, BBOX_WGS84["east"] + buf),
        lat=slice(BBOX_WGS84["north"] + buf, BBOX_WGS84["south"] - buf),
    )

    # Temporal subset
    ds = ds.sel(day=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

    log.info(
        "Subsetted grid: lon %d, lat %d, days %d",
        ds.sizes["lon"], ds.sizes["lat"], ds.sizes["day"],
    )

    # Load into memory (small dataset — ~5 MB)
    log.info("Downloading subsetted data from server…")
    ds = ds.load()

    # Filter to frost-season months only
    ds = ds.sel(day=ds["day"].dt.month.isin(FROST_MONTHS))
    log.info("After frost-month filter: %d days", ds.sizes["day"])

    # Convert Kelvin → Celsius
    if ds[TMIN_VAR].attrs.get("units", "") == "K":
        ds[TMIN_VAR] = ds[TMIN_VAR] - 273.15
        ds[TMIN_VAR].attrs["units"] = "degC"
        log.info("Converted Tmin from K → °C.")

    # Rename for clarity
    ds = ds.rename({TMIN_VAR: "tmin", "day": "time"})

    return ds


def save_zarr(ds: xr.Dataset) -> Path:
    """Save the clipped dataset as a Zarr store."""
    GRIDDED_TMIN_DIR.mkdir(parents=True, exist_ok=True)

    # Remove existing store if present
    if OUTPUT_PATH.exists():
        import shutil
        shutil.rmtree(OUTPUT_PATH)

    ds.to_zarr(OUTPUT_PATH, mode="w")
    log.info("Saved Zarr → %s", OUTPUT_PATH)
    return OUTPUT_PATH


def run() -> Path:
    """Execute gridded Tmin download pipeline."""
    ds = fetch_gridmet_tmin()

    # Summary
    log.info("── Dataset summary ──")
    log.info("  Shape: %s", dict(ds.sizes))
    log.info("  Lon range: %.3f to %.3f", float(ds.lon.min()), float(ds.lon.max()))
    log.info("  Lat range: %.3f to %.3f", float(ds.lat.min()), float(ds.lat.max()))
    log.info("  Time range: %s to %s",
             str(ds.time.values[0])[:10], str(ds.time.values[-1])[:10])
    log.info("  Tmin range: %.1f to %.1f °C",
             float(ds.tmin.min()), float(ds.tmin.max()))

    out = save_zarr(ds)
    ds.close()
    return out


if __name__ == "__main__":
    run()
