"""Create biweekly maximum-value NDVI composites from HLS-VI scenes.

Reads individual HLS NDVI scenes from data/raw/hls/, applies quality
filtering (cloud, cloud shadow, snow/ice removal), creates 14-day
maximum-value composites, aggregates from 30 m to 100 m, and stores
the result as a Zarr archive.

Also produces an ECOSTRESS scene → NDVI composite lookup table.

Outputs:
  data/processed/ndvi/hls_ndvi_composites.zarr
  data/processed/ndvi/ndvi_lookup.parquet
"""

import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    BBOX_WGS84,
    CRS_UTM,
    DEM_100M_PATH,
    NDVI_DIR,
    RAW_HLS_DIR,
    TARGET_RESOLUTION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Composite window in days
COMPOSITE_WINDOW = 14

# HLS Fmask bit positions for quality filtering
# Bit 0: cirrus, Bit 1: cloud, Bit 2: cloud shadow,
# Bit 3: snow/ice, Bit 4: aerosol level
FMASK_BAD_BITS = 0b00001111  # mask cirrus, cloud, shadow, snow


def _parse_hls_date(filename: str) -> datetime | None:
    """Extract acquisition date from HLS filename.

    HLS filenames follow: HLS.{sensor}.T{tile}.{date}T{time}.v2.0.{layer}.tif
    where date is YYYYDDD (year + day of year).
    """
    # Match patterns like HLS.L30.T17SNA.2023045T... or HLS.S30.T17SNA.2023045T...
    match = re.search(r"\.(\d{7})T", filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, "%Y%j")
    return None


def _find_ndvi_files() -> list[tuple[Path, datetime]]:
    """Find all HLS NDVI files in RAW_HLS_DIR and extract dates."""
    if not RAW_HLS_DIR.exists():
        raise FileNotFoundError(
            f"HLS directory not found at {RAW_HLS_DIR}. "
            "Run src/data/download_hls_ndvi.py first."
        )

    files = []
    for p in RAW_HLS_DIR.glob("*.tif"):
        # Only NDVI layers (skip Fmask, EVI, etc.)
        if "NDVI" not in p.name.upper():
            continue
        dt = _parse_hls_date(p.name)
        if dt is not None:
            files.append((p, dt))

    files.sort(key=lambda x: x[1])
    log.info("Found %d NDVI scenes spanning %s to %s",
             len(files),
             files[0][1].date() if files else "N/A",
             files[-1][1].date() if files else "N/A")
    return files


def _find_fmask_for(ndvi_path: Path) -> Path | None:
    """Find the corresponding Fmask quality file for an NDVI scene."""
    # Replace NDVI with Fmask in the filename
    fmask_name = ndvi_path.name.replace("NDVI", "Fmask").replace("ndvi", "Fmask")
    fmask_path = ndvi_path.parent / fmask_name
    if fmask_path.exists():
        return fmask_path
    # Try alternate naming
    for candidate in ndvi_path.parent.glob("*Fmask*"):
        # Match same granule by checking the date/tile portion
        ndvi_parts = ndvi_path.stem.split(".")
        cand_parts = candidate.stem.split(".")
        if len(ndvi_parts) >= 4 and len(cand_parts) >= 4:
            if ndvi_parts[2] == cand_parts[2] and ndvi_parts[3] == cand_parts[3]:
                return candidate
    return None


def _read_and_mask_ndvi(ndvi_path: Path) -> tuple[np.ndarray, dict] | None:
    """Read an NDVI scene and apply quality masking.

    Returns (masked_ndvi, profile) or None if the scene is mostly cloudy.
    """
    try:
        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            nodata = src.nodata
    except Exception as e:
        log.warning("Skipping unreadable file %s: %s", ndvi_path.name, e)
        return None

    # Apply nodata mask
    if nodata is not None:
        ndvi[ndvi == nodata] = np.nan

    # Apply Fmask if available
    fmask_path = _find_fmask_for(ndvi_path)
    if fmask_path is not None:
        with rasterio.open(fmask_path) as fsrc:
            fmask = fsrc.read(1)
        bad = (fmask & FMASK_BAD_BITS) != 0
        ndvi[bad] = np.nan

    # Skip scenes that are mostly masked
    valid_frac = np.sum(~np.isnan(ndvi)) / ndvi.size
    if valid_frac < 0.1:
        return None

    return ndvi, profile


def _get_target_grid() -> tuple[dict, tuple[int, int]]:
    """Get the 100 m target grid from the reference DEM."""
    with rasterio.open(DEM_100M_PATH) as src:
        profile = src.profile.copy()
    return profile, (profile["height"], profile["width"])


def _reproject_to_target(data: np.ndarray, src_profile: dict,
                          target_profile: dict) -> np.ndarray:
    """Reproject a 30 m array to the 100 m target grid using average."""
    dst = np.full((target_profile["height"], target_profile["width"]),
                  np.nan, dtype=np.float64)
    reproject(
        source=data.astype(np.float64),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _generate_composite_dates(start_date: datetime,
                               end_date: datetime) -> list[datetime]:
    """Generate biweekly composite center dates spanning the data range.

    Composites are centered on these dates, each covering ±7 days.
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=COMPOSITE_WINDOW)
    return dates


def run() -> tuple[Path, Path]:
    """Build biweekly NDVI composites and the scene lookup table.

    Returns paths to (zarr_archive, lookup_parquet).
    """
    NDVI_DIR.mkdir(parents=True, exist_ok=True)

    # Find all NDVI files
    ndvi_files = _find_ndvi_files()
    if not ndvi_files:
        raise RuntimeError("No NDVI files found. Run download_hls_ndvi.py first.")

    target_profile, (nrows, ncols) = _get_target_grid()

    # Determine composite date grid
    all_dates = [dt for _, dt in ndvi_files]
    comp_dates = _generate_composite_dates(min(all_dates), max(all_dates))
    log.info("Creating %d biweekly composites from %s to %s",
             len(comp_dates), comp_dates[0].date(), comp_dates[-1].date())

    # Initialize composite arrays
    composites = np.full((len(comp_dates), nrows, ncols), np.nan, dtype=np.float32)

    # For each composite window, collect max NDVI across all scenes in that window
    half_window = timedelta(days=COMPOSITE_WINDOW // 2)

    for ci, center_date in enumerate(comp_dates):
        window_start = center_date - half_window
        window_end = center_date + half_window

        # Find scenes within this window
        window_scenes = [
            (p, dt) for p, dt in ndvi_files
            if window_start <= dt < window_end
        ]

        if not window_scenes:
            continue

        log.info("  Composite %s: %d scenes", center_date.date(), len(window_scenes))

        for path, dt in window_scenes:
            result = _read_and_mask_ndvi(path)
            if result is None:
                continue
            ndvi_30m, scene_profile = result

            # Reproject to 100 m target grid
            ndvi_100m = _reproject_to_target(ndvi_30m, scene_profile, target_profile)

            # Maximum-value compositing
            current = composites[ci]
            update = ~np.isnan(ndvi_100m) & (
                np.isnan(current) | (ndvi_100m > current)
            )
            current[update] = ndvi_100m[update].astype(np.float32)

    # Temporal interpolation for gaps
    log.info("Interpolating temporal gaps …")
    for r in range(nrows):
        for c in range(ncols):
            ts = composites[:, r, c]
            valid = ~np.isnan(ts)
            if valid.sum() >= 2 and (~valid).sum() > 0:
                indices = np.arange(len(ts))
                ts[~valid] = np.interp(indices[~valid], indices[valid], ts[valid])
                composites[:, r, c] = ts

    # Save as Zarr
    zarr_path = NDVI_DIR / "hls_ndvi_composites.zarr"
    log.info("Writing composites to %s …", zarr_path)

    # Get coordinates from target profile
    transform = target_profile["transform"]
    xs = np.array([transform.c + (j + 0.5) * transform.a for j in range(ncols)])
    ys = np.array([transform.f + (i + 0.5) * transform.e for i in range(nrows)])
    times = pd.DatetimeIndex([pd.Timestamp(d) for d in comp_dates])

    ds = xr.Dataset(
        {"ndvi": (["time", "y", "x"], composites)},
        coords={
            "time": times,
            "y": ys,
            "x": xs,
        },
        attrs={
            "crs": str(target_profile["crs"]),
            "resolution_m": TARGET_RESOLUTION,
            "composite_window_days": COMPOSITE_WINDOW,
            "description": "Biweekly max-value NDVI composites from HLS-VI at 100 m",
        },
    )
    ds.to_zarr(str(zarr_path), mode="w")
    log.info("Wrote %s", zarr_path)

    # Build ECOSTRESS → NDVI lookup table
    lookup_path = NDVI_DIR / "ndvi_lookup.parquet"
    _build_lookup_table(comp_dates, lookup_path)

    return zarr_path, lookup_path


def _build_lookup_table(comp_dates: list[datetime], out_path: Path) -> None:
    """Create a lookup mapping ECOSTRESS scene dates to nearest NDVI composite.

    This creates the template table. Actual ECOSTRESS scene_ids are populated
    when ECOSTRESS scenes are processed (Task 2.1).
    """
    df = pd.DataFrame({
        "composite_center_date": [d.date() for d in comp_dates],
        "composite_index": range(len(comp_dates)),
    })
    df.to_parquet(out_path, index=False)
    log.info("Wrote NDVI lookup table (%d composites) → %s", len(df), out_path)


if __name__ == "__main__":
    run()
