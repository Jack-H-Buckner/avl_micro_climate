"""Fetch GHCN-Daily station data for the Asheville study area.

Downloads station metadata and daily Tmin/Tmax records from the NOAA
bulk data archive (no API key required).  Stations are filtered to the
bounding box + 20 km buffer and screened for data availability during
the frost season (Sep–May), 1991–present.
"""

import gzip
import logging
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, RAW_STATIONS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
GHCN_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
STATIONS_URL = f"{GHCN_BASE}/ghcnd-stations.txt"
INVENTORY_URL = f"{GHCN_BASE}/ghcnd-inventory.txt"
STATION_DATA_URL = f"{GHCN_BASE}/all/{{station_id}}.dly"

# 20 km buffer in approximate degrees
BUFFER_DEG = 0.18

# Study period
START_YEAR = 1991
END_YEAR = 2025


def _buffered_bbox() -> dict:
    """Return the study-area bounding box expanded by ~20 km."""
    return {
        "west": BBOX_WGS84["west"] - BUFFER_DEG,
        "east": BBOX_WGS84["east"] + BUFFER_DEG,
        "south": BBOX_WGS84["south"] - BUFFER_DEG,
        "north": BBOX_WGS84["north"] + BUFFER_DEG,
    }


# ── Step 1: station metadata ───────────────────────────────────────────────

def fetch_station_metadata() -> pd.DataFrame:
    """Download and parse the GHCN-Daily station list.

    Returns a DataFrame with columns: station_id, lat, lon, elevation, name.
    """
    cache = RAW_STATIONS_DIR / "ghcnd-stations.txt"
    RAW_STATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        log.info("Using cached station list: %s", cache)
        text = cache.read_text()
    else:
        log.info("Downloading station metadata from NOAA…")
        resp = requests.get(STATIONS_URL, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache.write_text(text)

    # Fixed-width format described at:
    # https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
    rows = []
    for line in text.splitlines():
        if len(line) < 85:
            continue
        rows.append({
            "station_id": line[0:11].strip(),
            "lat": float(line[12:20]),
            "lon": float(line[21:30]),
            "elevation": float(line[31:37]),
            "name": line[41:71].strip(),
        })

    df = pd.DataFrame(rows)
    log.info("Parsed %d total GHCN-Daily stations.", len(df))
    return df


def filter_stations_by_bbox(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only stations within the buffered bounding box."""
    bb = _buffered_bbox()
    mask = (
        (df["lat"] >= bb["south"]) & (df["lat"] <= bb["north"]) &
        (df["lon"] >= bb["west"]) & (df["lon"] <= bb["east"])
    )
    filtered = df[mask].copy()
    log.info("Stations within buffered bbox: %d", len(filtered))
    return filtered


# ── Step 2: check inventory for TMIN coverage ──────────────────────────────

def fetch_inventory() -> pd.DataFrame:
    """Download and parse the GHCN-Daily inventory file.

    Returns DataFrame: station_id, element, first_year, last_year.
    """
    cache = RAW_STATIONS_DIR / "ghcnd-inventory.txt"
    RAW_STATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        log.info("Using cached inventory: %s", cache)
        text = cache.read_text()
    else:
        log.info("Downloading inventory from NOAA…")
        resp = requests.get(INVENTORY_URL, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache.write_text(text)

    df = pd.read_fwf(
        StringIO(text),
        colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40)],
        names=["station_id", "lat", "lon", "element", "first_year"],
        dtype={"station_id": str, "element": str},
    )
    # The last column pair is actually element, first_year, last_year
    # Re-parse more carefully
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append({
                "station_id": parts[0],
                "element": parts[3],
                "first_year": int(parts[4]) if len(parts) > 4 else None,
                "last_year": int(parts[5]) if len(parts) > 5 else None,
            })

    df = pd.DataFrame(rows)
    return df


def filter_inventory_tmin(inventory: pd.DataFrame, station_ids: list[str]) -> list[str]:
    """Return station IDs that have TMIN data spanning the study period."""
    tmin = inventory[
        (inventory["element"] == "TMIN") &
        (inventory["station_id"].isin(station_ids))
    ].copy()

    # Require data starting by START_YEAR and extending to at least 2020
    tmin = tmin[
        (tmin["first_year"] <= START_YEAR) &
        (tmin["last_year"] >= 2020)
    ]
    ids = tmin["station_id"].unique().tolist()
    log.info("Stations with TMIN coverage %d–2020+: %d", START_YEAR, len(ids))
    return ids


# ── Step 3: download daily .dly files ──────────────────────────────────────

def _parse_dly(text: str, station_id: str) -> pd.DataFrame:
    """Parse a GHCN-Daily .dly fixed-width file into a long DataFrame.

    Returns columns: station_id, date, element, value, qflag.
    """
    rows = []
    for line in text.splitlines():
        if len(line) < 269:
            continue
        sid = line[0:11].strip()
        year = int(line[11:15])
        month = int(line[15:17])
        element = line[17:21].strip()

        if element not in ("TMIN", "TMAX"):
            continue
        if year < START_YEAR:
            continue

        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            val_str = line[offset:offset + 5].strip()
            qflag = line[offset + 6:offset + 7].strip()

            if val_str == "-9999" or val_str == "":
                continue

            try:
                date = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue  # invalid date (e.g., Feb 30)

            # GHCN-D stores temps in tenths of °C
            rows.append({
                "station_id": station_id,
                "date": date,
                "element": element,
                "value": int(val_str) / 10.0,
                "qflag": qflag if qflag else "",
            })

    return pd.DataFrame(rows)


def download_station_data(station_ids: list[str]) -> pd.DataFrame:
    """Download and parse .dly files for all stations. Returns long-format DataFrame."""
    RAW_STATIONS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for i, sid in enumerate(station_ids, 1):
        cache = RAW_STATIONS_DIR / f"{sid}.dly"
        if cache.exists():
            text = cache.read_text()
        else:
            url = STATION_DATA_URL.format(station_id=sid)
            log.info("[%d/%d] Downloading %s…", i, len(station_ids), sid)
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
            except requests.RequestException as e:
                log.warning("Failed to download %s: %s", sid, e)
                continue
            text = resp.text
            cache.write_text(text)

        df = _parse_dly(text, sid)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        log.error("No station data downloaded!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Downloaded %d observations from %d stations.", len(combined), len(all_dfs))
    return combined


# ── Step 4: quality control ────────────────────────────────────────────────

def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    """Apply QC filters to daily temperature observations.

    Removes:
    - Observations with non-empty QC flags (GHCN-D quality problems)
    - Tmin values below -30°C (instrument error in this region)
    - Records where Tmin > Tmax for the same station-day
    """
    n_start = len(df)

    # Remove flagged observations
    df = df[df["qflag"] == ""].copy()
    log.info("After QC flag filter: %d → %d", n_start, len(df))

    # Screen extreme values
    tmin_mask = (df["element"] == "TMIN") & (df["value"] < -30.0)
    tmax_mask = (df["element"] == "TMAX") & (df["value"] > 50.0)
    df = df[~tmin_mask & ~tmax_mask].copy()

    # Pivot to check Tmin > Tmax
    daily = df.pivot_table(
        index=["station_id", "date"], columns="element", values="value", aggfunc="first"
    )
    if "TMIN" in daily.columns and "TMAX" in daily.columns:
        bad = daily[daily["TMIN"] > daily["TMAX"]].index
        if len(bad) > 0:
            log.info("Removing %d days where Tmin > Tmax.", len(bad))
            bad_set = set(bad)
            df = df[~df.set_index(["station_id", "date"]).index.isin(bad_set)].copy()

    log.info("After all QC: %d records.", len(df))
    return df


# ── Step 5: save processed station data ────────────────────────────────────

def save_daily_tmin(df: pd.DataFrame, meta: pd.DataFrame) -> Path:
    """Pivot to daily Tmin, merge with station metadata, save as parquet."""
    tmin = df[df["element"] == "TMIN"][["station_id", "date", "value"]].copy()
    tmin = tmin.rename(columns={"value": "tmin"})

    # Merge with metadata
    tmin = tmin.merge(
        meta[["station_id", "lat", "lon", "elevation", "name"]],
        on="station_id", how="left",
    )

    out_path = RAW_STATIONS_DIR / "ghcn_daily_tmin.parquet"
    tmin.to_parquet(out_path, index=False)
    log.info("Saved daily Tmin → %s  (%d records, %d stations)",
             out_path, len(tmin), tmin["station_id"].nunique())
    return out_path


# ── Main pipeline ───────────────────────────────────────────────────────────

def run() -> Path:
    """Execute full station data acquisition pipeline."""
    # 1. Get station metadata
    all_stations = fetch_station_metadata()

    # 2. Filter to study area
    local_stations = filter_stations_by_bbox(all_stations)

    # 3. Check inventory for TMIN coverage
    inventory = fetch_inventory()
    good_ids = filter_inventory_tmin(inventory, local_stations["station_id"].tolist())

    if not good_ids:
        log.warning("No stations with sufficient TMIN coverage found. "
                     "Relaxing criteria to include any station with TMIN data…")
        tmin_inv = inventory[
            (inventory["element"] == "TMIN") &
            (inventory["station_id"].isin(local_stations["station_id"]))
        ]
        good_ids = tmin_inv["station_id"].unique().tolist()
        log.info("Found %d stations with any TMIN data.", len(good_ids))

    # 4. Download daily data
    daily_df = download_station_data(good_ids)
    if daily_df.empty:
        raise RuntimeError("No station data was downloaded.")

    # 5. QC
    daily_df = quality_control(daily_df)

    # 6. Save
    meta = local_stations[local_stations["station_id"].isin(good_ids)]
    out_path = save_daily_tmin(daily_df, meta)

    # Summary
    stations_used = daily_df[daily_df["element"] == "TMIN"]["station_id"].nunique()
    date_range = daily_df["date"].agg(["min", "max"])
    log.info("Station data pipeline complete: %d stations, %s to %s",
             stations_used, date_range["min"].date(), date_range["max"].date())

    return out_path


if __name__ == "__main__":
    run()
