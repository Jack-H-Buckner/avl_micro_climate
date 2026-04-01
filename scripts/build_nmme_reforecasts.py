"""Build a reforecast dataset of NMME seasonal anomalies for frost seasons.

Downloads NMME tmin ensemble-mean anomalies for each initialization,
extracts the study-area frost-season mean anomaly, and compiles a
tidy dataset pairing forecast anomalies with the target frost season.

Spring frost season (last spring frost risk):
  - Init months: Feb, Mar, Apr
  - Target months: Feb, Mar, Apr, May (leads vary by init)

Fall frost season (first fall frost risk):
  - Init months: Aug, Sep, Oct
  - Target months: Sep, Oct, Nov, Dec (leads vary by init)

Output
------
    data/processed/forecasts/nmme_reforecasts.parquet
    data/processed/forecasts/nmme_reforecasts.csv

Columns: init_year, init_month, season, frost_year, target_months,
         forecast_anomaly_C, n_leads

Usage
-----
    python scripts/build_nmme_reforecasts.py
    python scripts/build_nmme_reforecasts.py --keep-raw  # retain NetCDF files
"""

import argparse
import logging
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CPC_BASE = "https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom"
PROCESSED_DIR = DATA_DIR / "processed" / "forecasts"
RAW_DIR = DATA_DIR / "raw" / "forecasts" / "nmme"

# Study area (1° grid cells covering Asheville, descending lat)
LAT_SLICE = slice(36.5, 34.5)
LON_SLICE = slice(276.5, 278.5)

# Season definitions: which target months matter for each season
SPRING_TARGET_MONTHS = [2, 3, 4, 5]   # Feb-May (last frost risk)
FALL_TARGET_MONTHS = [9, 10, 11, 12]  # Sep-Dec (first frost risk)

# Init months for each season
SPRING_INIT_MONTHS = [2, 3, 4]
FALL_INIT_MONTHS = [8, 9, 10]


def _list_available_inits():
    """Get all available initialization dates from CPC."""
    url = f"{CPC_BASE}/ENSMEAN/"
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode()
    dirs = sorted(re.findall(r'href="(\d{10})/"', html))
    return [(int(d[:4]), int(d[4:6]), d) for d in dirs]


def _download_and_extract(init_year, init_month, init_str, target_months,
                          keep_raw=False):
    """Download one NMME file, extract study-area anomaly, optionally delete.

    Returns
    -------
    dict with keys: forecast_anomaly_C, n_leads, target_month_anomalies
    or None if download fails.
    """
    yyyymm = f"{init_year:04d}{init_month:02d}"
    filename = f"NMME.tmin.{yyyymm}.ENSMEAN.anom.nc"
    url = f"{CPC_BASE}/ENSMEAN/{init_str}/{filename}"

    if keep_raw:
        local_path = RAW_DIR / filename
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    else:
        local_path = Path(tempfile.mktemp(suffix=".nc"))

    try:
        urllib.request.urlretrieve(url, str(local_path))
    except urllib.error.HTTPError:
        log.warning("  %s not found — skipping", filename)
        return None

    try:
        ds = xr.open_dataset(local_path, decode_times=False)
        sub = ds.sel(lat=LAT_SLICE, lon=LON_SLICE)

        # Decode target months
        monthly_anoms = {}
        for i, t in enumerate(sub.target.values):
            months_offset = int(t)
            year = 1960 + months_offset // 12
            month = months_offset % 12 + 1

            if month in target_months:
                val = float(sub["fcst"].isel(target=i).mean())
                # Skip zero-filled leads (far-out leads sometimes = 0.0 exactly)
                if abs(val) > 1e-6:
                    monthly_anoms[(year, month)] = val

        ds.close()
    finally:
        if not keep_raw and local_path.exists():
            os.remove(local_path)

    if not monthly_anoms:
        return None

    season_mean = float(np.mean(list(monthly_anoms.values())))
    return {
        "forecast_anomaly_C": season_mean,
        "n_leads": len(monthly_anoms),
        "monthly_anomalies": monthly_anoms,
    }


def _determine_frost_year(init_year, init_month, season):
    """Determine which frost year a forecast targets.

    Frost year is labelled by the year of its September:
      - Fall inits (Aug-Oct of year Y) → frost_year = Y
      - Spring inits (Feb-Apr of year Y) → frost_year = Y - 1
        (they target the spring of the frost season that started prev Sep)
    """
    if season == "fall":
        return init_year
    else:  # spring
        return init_year - 1


def main():
    parser = argparse.ArgumentParser(
        description="Build NMME reforecast dataset for frost seasons"
    )
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep raw NetCDF files after extraction")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Get available dates
    log.info("═══ Listing available NMME initializations ═══")
    available = _list_available_inits()
    log.info("  %d total initializations on CPC server", len(available))

    # Filter to our target init months
    spring_inits = [(y, m, s) for y, m, s in available if m in SPRING_INIT_MONTHS]
    fall_inits = [(y, m, s) for y, m, s in available if m in FALL_INIT_MONTHS]
    log.info("  Spring inits (Feb/Mar/Apr): %d", len(spring_inits))
    log.info("  Fall inits (Aug/Sep/Oct): %d", len(fall_inits))

    records = []

    # ── Fall season reforecasts ──────────────────────────────────────────
    log.info("═══ Downloading fall frost season reforecasts ═══")
    for i, (year, month, init_str) in enumerate(fall_inits):
        log.info("  [%d/%d] %04d-%02d ...", i + 1, len(fall_inits), year, month)
        result = _download_and_extract(
            year, month, init_str,
            target_months=FALL_TARGET_MONTHS,
            keep_raw=args.keep_raw,
        )
        if result is None:
            continue

        frost_year = _determine_frost_year(year, month, "fall")
        records.append({
            "init_year": year,
            "init_month": month,
            "season": "fall",
            "frost_year": frost_year,
            "forecast_anomaly_C": result["forecast_anomaly_C"],
            "n_leads": result["n_leads"],
            "target_months": str(sorted(result["monthly_anomalies"].keys())),
        })

    # ── Spring season reforecasts ────────────────────────────────────────
    log.info("═══ Downloading spring frost season reforecasts ═══")
    for i, (year, month, init_str) in enumerate(spring_inits):
        log.info("  [%d/%d] %04d-%02d ...", i + 1, len(spring_inits), year, month)
        result = _download_and_extract(
            year, month, init_str,
            target_months=SPRING_TARGET_MONTHS,
            keep_raw=args.keep_raw,
        )
        if result is None:
            continue

        frost_year = _determine_frost_year(year, month, "spring")
        records.append({
            "init_year": year,
            "init_month": month,
            "season": "spring",
            "frost_year": frost_year,
            "forecast_anomaly_C": result["forecast_anomaly_C"],
            "n_leads": result["n_leads"],
            "target_months": str(sorted(result["monthly_anomalies"].keys())),
        })

    # ── Save ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.sort_values(["season", "frost_year", "init_month"]).reset_index(drop=True)

    parquet_path = PROCESSED_DIR / "nmme_reforecasts.parquet"
    csv_path = PROCESSED_DIR / "nmme_reforecasts.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, float_format="%.4f")

    log.info("═══ Reforecast dataset complete ═══")
    log.info("  Records: %d", len(df))
    log.info("  Saved: %s", parquet_path)
    log.info("  Saved: %s", csv_path)
    log.info("")

    # Summary stats
    for season in ["fall", "spring"]:
        sub = df[df["season"] == season]
        log.info("  %s season: %d records, frost years %d–%d",
                 season.capitalize(), len(sub),
                 sub["frost_year"].min(), sub["frost_year"].max())
        for m in sorted(sub["init_month"].unique()):
            msub = sub[sub["init_month"] == m]
            log.info("    Init month %02d: %d years, mean anomaly %+.3f°C (range %+.2f to %+.2f)",
                     m, len(msub),
                     msub["forecast_anomaly_C"].mean(),
                     msub["forecast_anomaly_C"].min(),
                     msub["forecast_anomaly_C"].max())


if __name__ == "__main__":
    main()
