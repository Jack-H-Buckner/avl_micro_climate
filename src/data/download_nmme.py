"""Fetch NMME seasonal forecast data from CPC FTP server.

Downloads monthly mean temperature anomaly forecasts from the
North American Multi-Model Ensemble (NMME), extracts the grid cells
covering the study area, and computes seasonal anomalies for use
as the GBM seasonal_anomaly_C feature.

Data source: https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/
Reference:   https://www.cpc.ncep.noaa.gov/products/NMME/data.html

Available products per initialization:
  - ENSMEAN/{init}/NMME.tmin.{YYYYMM}.ENSMEAN.anom.nc  (multi-model mean)
  - ENSMEAN/{init}/{Model}.tmp2m.{YYYYMM}.ENSMEAN.anom.nc  (per-model mean)
  - {Model}/{init}/{Model}.tmp2m.{YYYYMM}.anom.nc  (per-model full ensemble)

Models on CPC server: CFSv2, CanESM5, GEM5.2_NEMO, GFDL_SPEAR,
                       NASA_GEOS5v2, NCAR_CCSM4, NCAR_CESM1

Grid: 1° x 1° global (360 x 181), lat 90 to -90, lon 0-359.
      Anomalies in Kelvin (= °C for temperature differences).
      Lead times: 12 months from initialization.

Usage
-----
    python -m src.data.download_nmme --init-year 2026 --init-month 8
    python -m src.data.download_nmme --init-year 2026 --init-month 3 --list
"""

import argparse
import logging
import os
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
FORECAST_DIR = DATA_DIR / "raw" / "forecasts" / "nmme"
PROCESSED_FORECAST_DIR = DATA_DIR / "processed" / "forecasts"

# ── CPC FTP server ───────────────────────────────────────────────────────────
CPC_BASE = "https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom"

# Model directories on CPC (for individual ensemble member data)
CPC_MODELS = [
    "CFSv2", "CanESM5", "GEM5.2_NEMO", "GFDL_SPEAR",
    "NASA_GEOS5v2", "NCAR_CCSM4", "NCAR_CESM1",
]

# Variables of interest
NMME_VARS = {
    "tmin": "Minimum 2m temperature anomaly",
    "tmp2m": "Mean 2m temperature anomaly",
    "tmax": "Maximum 2m temperature anomaly (ENSMEAN only)",
    "prate": "Precipitation rate anomaly",
}

# Study area in 0-360 longitude convention (NMME grid)
# Asheville: ~35.4-35.8N, 82.3-82.8W → lon 277.2-277.7
LAT_SLICE = slice(36.5, 34.5)  # descending lat
LON_SLICE = slice(276.5, 278.5)

# Frost season months
FROST_MONTHS = [9, 10, 11, 12, 1, 2, 3, 4, 5]


def _list_init_dates(model_or_ensmean="ENSMEAN"):
    """List available initialization dates on CPC server."""
    url = f"{CPC_BASE}/{model_or_ensmean}/"
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode()
    dirs = re.findall(r'href="(\d{10})/"', html)
    return sorted(dirs)


def _init_date_str(init_year, init_month):
    """Format initialization date as CPC directory name (YYYYMM0800)."""
    return f"{init_year:04d}{init_month:02d}0800"


def _download_file(url, local_path):
    """Download a file from URL to local path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(local_path))
    return local_path


def _decode_target_months(ds, init_year, init_month):
    """Convert NMME 'target' coordinate to actual year-month labels.

    Target values are 'months since 1960-01-01', e.g. 794 = March 2026.
    Returns list of (year, month) tuples.
    """
    targets = []
    for t in ds.target.values:
        months_offset = int(t)
        year = 1960 + months_offset // 12
        month = months_offset % 12 + 1
        targets.append((year, month))
    return targets


def fetch_nmme_forecast(init_year, init_month, variable="tmin",
                        ensemble_mean=True):
    """Fetch NMME forecast anomalies for a given initialization.

    Parameters
    ----------
    init_year : int
    init_month : int (1-12)
    variable : str, one of 'tmin', 'tmp2m', 'prate'
    ensemble_mean : bool
        If True, fetch the multi-model ensemble mean (ENSMEAN/NMME.{var}.*).
        If False, fetch per-model ensemble means from ENSMEAN directory.

    Returns
    -------
    xr.Dataset clipped to study area with decoded target months.
    """
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    init_str = _init_date_str(init_year, init_month)
    yyyymm = f"{init_year:04d}{init_month:02d}"

    if ensemble_mean:
        filename = f"NMME.{variable}.{yyyymm}.ENSMEAN.anom.nc"
        url = f"{CPC_BASE}/ENSMEAN/{init_str}/{filename}"
        local_path = FORECAST_DIR / filename

        log.info("Downloading %s ...", filename)
        try:
            _download_file(url, local_path)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"NMME forecast not found: {url}\n"
                f"Check available dates with --list flag."
            ) from e

        ds = xr.open_dataset(local_path, decode_times=False)
        ds = ds.sel(lat=LAT_SLICE, lon=LON_SLICE)

        # Decode target months
        target_months = _decode_target_months(ds, init_year, init_month)
        ds = ds.assign_coords(
            target_label=("target", [f"{y}-{m:02d}" for y, m in target_months])
        )

        log.info("  Grid: %d lat x %d lon, %d lead months",
                 len(ds.lat), len(ds.lon), len(ds.target))
        log.info("  Leads: %s to %s", target_months[0], target_months[-1])

        return ds

    else:
        # Fetch per-model ensemble means
        datasets = {}
        for model in CPC_MODELS:
            filename = f"{model}.tmp2m.{yyyymm}.ENSMEAN.anom.nc"
            url = f"{CPC_BASE}/ENSMEAN/{init_str}/{filename}"
            local_path = FORECAST_DIR / filename

            try:
                log.info("  Downloading %s ...", filename)
                _download_file(url, local_path)
                ds = xr.open_dataset(local_path, decode_times=False)
                ds = ds.sel(lat=LAT_SLICE, lon=LON_SLICE)
                datasets[model] = ds
            except urllib.error.HTTPError:
                log.warning("  %s not available for this init — skipping", model)

        if not datasets:
            raise RuntimeError("No per-model NMME data could be fetched")

        return datasets


def extract_seasonal_anomaly(ds, frost_year=None, init_month=None):
    """Extract frost-season mean anomaly from an NMME forecast dataset.

    Parameters
    ----------
    ds : xr.Dataset from fetch_nmme_forecast() (clipped to study area).
    frost_year : int, optional — year of the September of the frost season.
    init_month : int, optional — initialization month (to compute lead offsets).

    Returns
    -------
    float : study-area-mean frost-season temperature anomaly in °C.
    dict : per-month anomalies {(year, month): float}.
    """
    target_months = _decode_target_months(ds, frost_year or 2026, init_month or 8)
    anom = ds["fcst"]  # units: K (= °C for anomalies)

    # Identify which target indices correspond to frost season months
    frost_indices = []
    monthly_anomalies = {}

    for i, (y, m) in enumerate(target_months):
        if m in FROST_MONTHS:
            val = float(anom.isel(target=i).mean())
            if abs(val) > 1e-6 or m in [9, 10, 11]:  # skip zero-filled far leads
                frost_indices.append(i)
                monthly_anomalies[(y, m)] = val

    if not frost_indices:
        log.warning("No frost-season months found in forecast leads")
        return 0.0, {}

    # Spatial + lead-month average
    frost_anom = anom.isel(target=frost_indices)
    season_mean = float(frost_anom.mean())

    log.info("  Frost-season months: %d leads, mean anomaly: %+.3f °C",
             len(frost_indices), season_mean)

    return season_mean, monthly_anomalies


def compute_gridmet_seasonal_anomaly(gridmet_zarr_path, frost_year):
    """Compute observed seasonal Tmin anomaly from gridMET for a frost year.

    This serves as the "synthetic reforecast" — the 1° scale seasonal
    anomaly that the GBM was trained on.

    Parameters
    ----------
    gridmet_zarr_path : Path to gridMET Zarr archive.
    frost_year : int (year of the September, e.g. 2023 for Sep 2023 - May 2024).

    Returns
    -------
    float : seasonal mean Tmin anomaly in °C (study area average).
    """
    ds = xr.open_zarr(str(gridmet_zarr_path))

    # Extract frost season dates for this year
    frost_start = np.datetime64(f"{frost_year}-09-01")
    frost_end = np.datetime64(f"{frost_year + 1}-05-31")
    season = ds["tmmn"].sel(time=slice(frost_start, frost_end))

    # Compute long-term climatology (all Sep-May in archive)
    all_times = ds.time.values
    frost_mask = np.isin(
        np.array([t.astype("datetime64[M]").astype(int) % 12 + 1
                  for t in all_times]),
        FROST_MONTHS,
    )
    clim_mean = float(ds["tmmn"].isel(time=frost_mask).mean().values)
    season_mean = float(season.mean().values)

    ds.close()
    return season_mean - clim_mean


def bias_correct_anomaly(forecast_anomaly, hindcast_anomalies, observed_anomalies):
    """Bias-correct a forecast anomaly using variance scaling.

    Adjusts the forecast anomaly so that the mean and variance of
    hindcast anomalies match the observed anomaly distribution.

    Parameters
    ----------
    forecast_anomaly : float — raw forecast anomaly (°C)
    hindcast_anomalies : array — NMME hindcast anomalies for reference period
    observed_anomalies : array — gridMET observed anomalies for same period

    Returns
    -------
    float : bias-corrected anomaly (°C)
    """
    hc = np.asarray(hindcast_anomalies)
    obs = np.asarray(observed_anomalies)

    hc_mean, hc_std = hc.mean(), hc.std()
    obs_mean, obs_std = obs.mean(), obs.std()

    if hc_std < 1e-6:
        return forecast_anomaly - hc_mean + obs_mean

    # Standardize, then rescale to observed distribution
    z = (forecast_anomaly - hc_mean) / hc_std
    corrected = z * obs_std + obs_mean

    return float(corrected)


def main():
    parser = argparse.ArgumentParser(description="Fetch NMME seasonal forecasts")
    parser.add_argument("--init-year", type=int, help="Forecast initialization year")
    parser.add_argument("--init-month", type=int, help="Forecast initialization month")
    parser.add_argument("--variable", default="tmin",
                        choices=["tmin", "tmp2m", "prate"],
                        help="Variable to download (default: tmin)")
    parser.add_argument("--per-model", action="store_true",
                        help="Fetch per-model means instead of multi-model mean")
    parser.add_argument("--list", action="store_true",
                        help="List available initialization dates and exit")
    args = parser.parse_args()

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        log.info("═══ Available NMME initialization dates ═══")
        dates = _list_init_dates()
        for d in dates[-24:]:  # last 24 months
            y, m = int(d[:4]), int(d[4:6])
            log.info("  %04d-%02d  (%s)", y, m, d)
        log.info("  Total: %d initializations available", len(dates))
        return

    if not args.init_year or not args.init_month:
        parser.error("--init-year and --init-month required (or use --list)")

    log.info("═══ Fetching NMME forecast: %04d-%02d ═══",
             args.init_year, args.init_month)

    ds = fetch_nmme_forecast(
        args.init_year, args.init_month,
        variable=args.variable,
        ensemble_mean=not args.per_model,
    )

    if isinstance(ds, dict):
        # Per-model results
        for model, model_ds in ds.items():
            anom = model_ds["fcst"]
            spatial_mean = float(anom.mean())
            log.info("  %s: mean anomaly = %+.3f K", model, spatial_mean)
    else:
        # Multi-model ensemble mean
        season_anom, monthly = extract_seasonal_anomaly(
            ds, frost_year=args.init_year, init_month=args.init_month
        )
        log.info("Frost-season anomaly: %+.3f °C", season_anom)
        for (y, m), val in sorted(monthly.items()):
            log.info("  %04d-%02d: %+.3f °C", y, m, val)


if __name__ == "__main__":
    main()
