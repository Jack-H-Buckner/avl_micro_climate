"""Extract first-fall and last-spring frost dates from daily Tmin records.

Frost threshold: Tmin <= 0 °C.
Frost season: September 1 – May 31.
  - First fall frost: first day in Sep–Dec with Tmin <= 0 °C
  - Last spring frost: last day in Jan–May with Tmin <= 0 °C

A "frost year" is defined as the period Sep 1 of year Y through May 31
of year Y+1.  The frost year is labelled by the *fall* year Y.

Completeness filter: station-years with < 80% of frost-season days
observed are excluded.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_STATIONS_DIR, STATION_FROST_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

FROST_THRESHOLD = 0.0  # °C
MIN_COMPLETENESS = 0.80
FROST_SEASON_DAYS = 273  # Sep 1 → May 31 = 273 days


def _assign_frost_year(dates: pd.Series) -> pd.Series:
    """Map each date to a frost year (Sep–May). Sep–Dec → year, Jan–May → year-1."""
    return np.where(dates.dt.month >= 9, dates.dt.year, dates.dt.year - 1)


def load_daily_tmin() -> pd.DataFrame:
    """Load the QC'd daily Tmin parquet file."""
    path = RAW_STATIONS_DIR / "ghcn_daily_tmin.parquet"
    df = pd.read_parquet(path)
    log.info("Loaded %d daily Tmin records from %d stations.", len(df), df["station_id"].nunique())
    return df


def compute_frost_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first fall frost and last spring frost per station per frost year.

    Parameters
    ----------
    df : DataFrame with columns station_id, date, tmin, lat, lon, elevation, name

    Returns
    -------
    DataFrame with columns:
        station_id, lat, lon, elevation, name, network, frost_year,
        first_fall_frost_doy, last_spring_frost_doy, n_days, completeness
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Filter to frost season months (Sep–May)
    df = df[df["date"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5])].copy()

    df["frost_year"] = _assign_frost_year(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["is_frost"] = df["tmin"] <= FROST_THRESHOLD

    results = []

    for (sid, fy), grp in df.groupby(["station_id", "frost_year"]):
        n_days = len(grp)
        completeness = n_days / FROST_SEASON_DAYS

        if completeness < MIN_COMPLETENESS:
            continue

        meta = grp.iloc[0]

        # Fall portion: Sep–Dec of frost_year
        fall = grp[(grp["date"].dt.year == fy) & (grp["date"].dt.month >= 9)]
        fall_frost = fall[fall["is_frost"]]

        # Spring portion: Jan–May of frost_year + 1
        spring = grp[(grp["date"].dt.year == fy + 1) & (grp["date"].dt.month <= 5)]
        spring_frost = spring[spring["is_frost"]]

        first_fall_doy = fall_frost["doy"].min() if not fall_frost.empty else np.nan
        last_spring_doy = spring_frost["doy"].max() if not spring_frost.empty else np.nan

        results.append({
            "station_id": sid,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "elevation": meta["elevation"],
            "name": meta.get("name", ""),
            "network": "GHCN-D",
            "frost_year": fy,
            "first_fall_frost_doy": first_fall_doy,
            "last_spring_frost_doy": last_spring_doy,
            "n_days": n_days,
            "completeness": round(completeness, 3),
        })

    result_df = pd.DataFrame(results)
    log.info("Computed frost dates: %d station-years from %d stations.",
             len(result_df), result_df["station_id"].nunique())
    return result_df


def save_frost_dates(df: pd.DataFrame) -> Path:
    """Save frost dates to parquet."""
    STATION_FROST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STATION_FROST_PATH, index=False)
    log.info("Saved frost dates → %s", STATION_FROST_PATH)
    return STATION_FROST_PATH


def run() -> Path:
    """Execute frost date extraction pipeline."""
    daily = load_daily_tmin()
    frost_df = compute_frost_dates(daily)

    if frost_df.empty:
        raise RuntimeError("No valid frost dates computed — check station data.")

    save_frost_dates(frost_df)

    # Summary stats
    log.info("── Summary ──")
    log.info("Stations: %d", frost_df["station_id"].nunique())
    log.info("Frost years: %d–%d", frost_df["frost_year"].min(), frost_df["frost_year"].max())
    log.info("First fall frost DOY — mean: %.0f, std: %.1f",
             frost_df["first_fall_frost_doy"].mean(), frost_df["first_fall_frost_doy"].std())
    log.info("Last spring frost DOY — mean: %.0f, std: %.1f",
             frost_df["last_spring_frost_doy"].mean(), frost_df["last_spring_frost_doy"].std())

    return STATION_FROST_PATH


if __name__ == "__main__":
    run()
