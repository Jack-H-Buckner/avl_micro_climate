"""Download ECOSTRESS LST scenes, clip to study area, apply QC, delete raw.

Two-step pipeline that keeps disk usage minimal:
  1. Download one granule's files to a temp directory
  2. Clip LST to study area, apply cloud/QC/error filters, save clipped
     result, delete the raw files

Each processed scene is saved as a small GeoTIFF (~50–200 KB) containing
only the study-area pixels that pass quality filters.

A scene inventory (parquet) tracks metadata: datetime, local overpass time,
cloud fraction, and valid pixel count.
"""

import logging
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, CRS_UTM, RAW_SATELLITE_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Output paths ────────────────────────────────────────────────────────────
ECOSTRESS_PROCESSED_DIR = PROCESSED_DIR / "satellite"
ECOSTRESS_SCENES_DIR = ECOSTRESS_PROCESSED_DIR / "ecostress_native"
SCENE_INVENTORY_PATH = ECOSTRESS_PROCESSED_DIR / "ecostress_scenes.parquet"

# ── Study area bounds in UTM (computed from DEM) ────────────────────────────
# We'll read these from the DEM to ensure exact alignment
DEM_PATH = PROCESSED_DIR / "dem_100m.tif"

# ── Constants ───────────────────────────────────────────────────────────────
LST_ERR_MAX = 2.0        # K — discard pixels with error > 2 K
ASHEVILLE_UTC_OFFSET_EST = -5  # hours (EST)
ASHEVILLE_UTC_OFFSET_EDT = -4  # hours (EDT)

# Frost seasons to download
FROST_SEASONS = [
    (f"{y}-09-01", f"{y+1}-05-31") for y in range(2018, 2025)
]


def _get_study_bounds_utm() -> tuple[float, float, float, float]:
    """Get study area bounds in UTM from the DEM."""
    with rasterio.open(DEM_PATH) as src:
        return src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top


def _parse_granule_datetime(filename: str) -> datetime | None:
    """Extract UTC acquisition datetime from ECOSTRESS filename.

    Filename pattern: ECOv002_L2T_LSTE_XXXXX_XXX_XXXXX_YYYYMMDDTHHMMSS_...
    """
    match = re.search(r"(\d{8}T\d{6})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return None


def _utc_to_local_hour(dt_utc: datetime) -> float:
    """Convert UTC datetime to Asheville local hour (accounting for DST)."""
    # Simple DST rule: EDT from 2nd Sunday in March to 1st Sunday in November
    year = dt_utc.year
    # March: 2nd Sunday
    mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    mar_second_sun = 14 - mar1.weekday()  # day of 2nd Sunday
    dst_start = datetime(year, 3, mar_second_sun, 2, tzinfo=timezone.utc)
    # November: 1st Sunday
    nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    nov_first_sun = 7 - nov1.weekday()
    if nov_first_sun == 0:
        nov_first_sun = 7
    dst_end = datetime(year, 11, nov_first_sun, 2, tzinfo=timezone.utc)

    if dst_start <= dt_utc < dst_end:
        offset = ASHEVILLE_UTC_OFFSET_EDT
    else:
        offset = ASHEVILLE_UTC_OFFSET_EST

    local = dt_utc + timedelta(hours=offset)
    return local.hour + local.minute / 60.0


def _classify_overpass(local_hour: float) -> str:
    """Classify overpass by local time of day."""
    if 2.0 <= local_hour < 6.0:
        return "predawn"
    elif 20.0 <= local_hour or local_hour < 2.0:
        return "evening"
    elif 6.0 <= local_hour < 10.0:
        return "morning"
    else:
        return "daytime"


def _group_granule_files(files: list[Path]) -> dict[str, dict[str, Path]]:
    """Group downloaded files by granule ID, keyed by layer name.

    ECOSTRESS filenames have a fixed 9-segment prefix:
      ECOv002_L2T_LSTE_ORBIT_SCENE_TILE_DATETIME_BUILD_VER
    Everything after that prefix is the layer name (e.g., LST, LST_err,
    cloud, view_zenith, QC, EmisWB, height, water).
    """
    granules: dict[str, dict[str, Path]] = {}
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 10:
            prefix = "_".join(parts[:9])
            layer = "_".join(parts[9:])
        else:
            prefix = f.stem
            layer = "unknown"
        granules.setdefault(prefix, {})[layer] = f
    return granules


def process_granule(
    layers: dict[str, Path],
    granule_id: str,
    study_bounds: tuple[float, float, float, float],
) -> dict | None:
    """Clip one granule to study area, apply QC, save filtered LST.

    Parameters
    ----------
    layers : dict mapping layer name → file path (LST, cloud, QC, LST_err, …)
    granule_id : identifier string for this granule
    study_bounds : (left, bottom, right, top) in UTM

    Returns
    -------
    dict of scene metadata, or None if scene has no valid data.
    """
    lst_path = layers.get("LST")
    if lst_path is None or not lst_path.exists():
        log.warning("No LST layer for %s — skipping.", granule_id)
        return None

    # Parse acquisition time
    dt_utc = _parse_granule_datetime(granule_id)
    if dt_utc is None:
        log.warning("Cannot parse datetime from %s — skipping.", granule_id)
        return None

    local_hour = _utc_to_local_hour(dt_utc)
    overpass_class = _classify_overpass(local_hour)

    left, bottom, right, top = study_bounds

    with rasterio.open(lst_path) as src:
        # Check if scene overlaps study area
        if (src.bounds.right < left or src.bounds.left > right or
                src.bounds.top < bottom or src.bounds.bottom > top):
            log.debug("Scene %s does not overlap study area.", granule_id)
            return None

        # Compute window for study area
        window = from_bounds(left, bottom, right, top, src.transform)

        # Clamp window to raster extent
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return None

        lst = src.read(1, window=window).astype(np.float32)
        win_transform = src.window_transform(window)
        nodata = src.nodata

    if nodata is not None:
        lst[lst == nodata] = np.nan

    # Also mark zero/negative as invalid (ECOSTRESS uses 0 for fill)
    lst[lst <= 0] = np.nan

    total_pixels = lst.size
    valid_before_qc = np.count_nonzero(~np.isnan(lst))

    # ── Apply cloud mask ────────────────────────────────────────────────
    cloud_path = layers.get("cloud")
    if cloud_path is not None and cloud_path.exists():
        with rasterio.open(cloud_path) as csrc:
            cwindow = from_bounds(left, bottom, right, top, csrc.transform)
            cwindow = cwindow.intersection(rasterio.windows.Window(0, 0, csrc.width, csrc.height))
            cloud = csrc.read(1, window=cwindow)
        # Cloud mask: 0 = clear, nonzero = cloudy
        if cloud.shape == lst.shape:
            lst[cloud != 0] = np.nan

    # ── Apply LST error filter ──────────────────────────────────────────
    err_path = layers.get("LST_err") or layers.get("LST err")
    if err_path is not None and err_path.exists():
        with rasterio.open(err_path) as esrc:
            ewindow = from_bounds(left, bottom, right, top, esrc.transform)
            ewindow = ewindow.intersection(rasterio.windows.Window(0, 0, esrc.width, esrc.height))
            lst_err = esrc.read(1, window=ewindow).astype(np.float32)
        if lst_err.shape == lst.shape:
            lst[lst_err > LST_ERR_MAX] = np.nan

    # ── Apply QC mask ───────────────────────────────────────────────────
    qc_path = layers.get("QC")
    if qc_path is not None and qc_path.exists():
        with rasterio.open(qc_path) as qsrc:
            qwindow = from_bounds(left, bottom, right, top, qsrc.transform)
            qwindow = qwindow.intersection(rasterio.windows.Window(0, 0, qsrc.width, qsrc.height))
            qc = qsrc.read(1, window=qwindow)
        if qc.shape == lst.shape:
            # QC bits 0-1: 00 = pixel produced with good quality
            # Keep only pixels where the two LSBs are 00
            lst[(qc & 0b11) != 0] = np.nan

    valid_after_qc = np.count_nonzero(~np.isnan(lst))
    cloud_fraction = 1.0 - (valid_after_qc / total_pixels) if total_pixels > 0 else 1.0

    # Skip scenes with very few valid pixels
    if valid_after_qc < 10:
        log.debug("Scene %s: too few valid pixels (%d) — skipping.", granule_id, valid_after_qc)
        return None

    # Convert K → °C
    lst = lst - 273.15

    # ── Save clipped, filtered LST ──────────────────────────────────────
    ECOSTRESS_SCENES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{dt_utc.strftime('%Y%m%dT%H%M%S')}_{overpass_class}.tif"
    out_path = ECOSTRESS_SCENES_DIR / out_name

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": lst.shape[1],
        "height": lst.shape[0],
        "count": 1,
        "crs": CRS_UTM,
        "transform": win_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(lst), np.nan, lst).astype(np.float32), 1)

    return {
        "granule_id": granule_id,
        "datetime_utc": dt_utc.isoformat(),
        "local_hour": round(local_hour, 2),
        "overpass_class": overpass_class,
        "cloud_fraction": round(cloud_fraction, 3),
        "valid_pixels": valid_after_qc,
        "total_pixels": total_pixels,
        "lst_min_c": round(float(np.nanmin(lst)), 2) if valid_after_qc > 0 else None,
        "lst_max_c": round(float(np.nanmax(lst)), 2) if valid_after_qc > 0 else None,
        "lst_mean_c": round(float(np.nanmean(lst)), 2) if valid_after_qc > 0 else None,
        "filename": out_name,
    }


def run(max_scenes: int | None = None) -> Path:
    """Execute the full ECOSTRESS download-clip-delete pipeline.

    Parameters
    ----------
    max_scenes : optional limit on total granules to process (for testing)
    """
    earthaccess.login()

    study_bounds = _get_study_bounds_utm()
    log.info("Study bounds (UTM): %s", study_bounds)

    # Load existing inventory to skip already-processed scenes
    existing_ids: set[str] = set()
    if SCENE_INVENTORY_PATH.exists():
        existing_df = pd.read_parquet(SCENE_INVENTORY_PATH)
        existing_ids = set(existing_df["granule_id"])
        log.info("Found existing inventory with %d scenes — will skip those.", len(existing_ids))

    all_metadata: list[dict] = []
    scenes_processed = 0

    for start_date, end_date in FROST_SEASONS:
        log.info("── Searching %s to %s ──", start_date, end_date)

        results = earthaccess.search_data(
            short_name="ECO_L2T_LSTE",
            version="002",
            bounding_box=(
                BBOX_WGS84["west"], BBOX_WGS84["south"],
                BBOX_WGS84["east"], BBOX_WGS84["north"],
            ),
            temporal=(start_date, end_date),
            count=500,
        )
        log.info("Found %d granules.", len(results))

        for i, granule in enumerate(results):
            if max_scenes is not None and scenes_processed >= max_scenes:
                log.info("Reached max_scenes limit (%d).", max_scenes)
                break

            # Use a temp directory for raw downloads
            with tempfile.TemporaryDirectory(prefix="eco_") as tmpdir:
                try:
                    files = earthaccess.download([granule], tmpdir)
                except Exception as e:
                    log.warning("Download failed: %s", e)
                    continue

                file_paths = [Path(f) for f in files]
                granule_groups = _group_granule_files(file_paths)

                for gid, layers in granule_groups.items():
                    if gid in existing_ids:
                        continue

                    meta = process_granule(layers, gid, study_bounds)
                    if meta is not None:
                        all_metadata.append(meta)
                        scenes_processed += 1

                        if scenes_processed % 25 == 0:
                            log.info("Processed %d scenes so far…", scenes_processed)

                # Raw files are automatically deleted when tmpdir exits

        if max_scenes is not None and scenes_processed >= max_scenes:
            break

    # ── Save / update inventory ─────────────────────────────────────────
    if all_metadata:
        new_df = pd.DataFrame(all_metadata)

        if SCENE_INVENTORY_PATH.exists():
            existing_df = pd.read_parquet(SCENE_INVENTORY_PATH)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset="granule_id", keep="last")
        else:
            combined = new_df

        ECOSTRESS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(SCENE_INVENTORY_PATH, index=False)
        log.info("Saved inventory → %s (%d total scenes)", SCENE_INVENTORY_PATH, len(combined))
    else:
        log.warning("No new scenes processed.")

    # ── Summary ─────────────────────────────────────────────────────────
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        log.info("── Pipeline summary ──")
        log.info("  New scenes processed: %d", len(df))
        log.info("  By overpass class:")
        for cls, count in df["overpass_class"].value_counts().items():
            log.info("    %s: %d", cls, count)
        log.info("  Mean cloud fraction: %.1f%%", df["cloud_fraction"].mean() * 100)

    return SCENE_INVENTORY_PATH


if __name__ == "__main__":
    run()
