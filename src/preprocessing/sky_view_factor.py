"""Compute Sky View Factor (SVF) from the DEM.

SVF quantifies the fraction of the sky hemisphere visible from each point,
ranging from 0 (fully enclosed) to 1 (flat open plain). It directly controls
the rate of longwave radiative cooling at night.

Pipeline:
  1. Reproject raw DEM to UTM 17N at 30 m (good balance of resolution
     and speed — still captures valley geometry)
  2. For each pixel, compute horizon angles at 36 azimuths (every 10°)
  3. SVF = 1 - mean(sin²(horizon_angle)) integrated over all azimuths
  4. Aggregate from 30 m to 100 m using mean

Output: data/processed/covariates/sky_view_factor.tif
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    COVARIATES_DIR,
    DEM_100M_PATH,
    RAW_DEM_DIR,
    CRS_UTM,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# SVF computation parameters
N_AZIMUTHS = 36           # directions (every 10° — sufficient for SVF)
SEARCH_RADIUS_M = 1000    # how far to look for horizon in metres
SVF_DEM_RES = 30          # compute SVF at 30 m (aggregated to 100 m output)

DEM_30M_PATH = PROCESSED_DIR / "dem_30m_svf.tif"


def _ensure_dem_30m() -> Path:
    """Reproject the raw WGS84 DEM to UTM 17N at 30 m for SVF computation."""
    if DEM_30M_PATH.exists():
        log.info("30 m DEM already exists at %s", DEM_30M_PATH)
        return DEM_30M_PATH

    raw_dem = RAW_DEM_DIR / "dem_raw_wgs84.tif"
    if not raw_dem.exists():
        raise FileNotFoundError(
            f"Raw DEM not found at {raw_dem}. Run src/data/download_dem.py first."
        )

    log.info("Reprojecting raw DEM to 30 m UTM for SVF …")
    DEM_30M_PATH.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raw_dem) as src:
        dst_crs = CRS.from_epsg(32617)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=SVF_DEM_RES,
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs, transform=transform,
            width=width, height=height,
            dtype="float32", nodata=-9999.0, compress="deflate",
        )
        with rasterio.open(DEM_30M_PATH, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    log.info("Wrote 30 m DEM → %s (%d × %d)", DEM_30M_PATH, width, height)
    return DEM_30M_PATH


def compute_horizon_angle(elev: np.ndarray, cell_size: float,
                          azimuth_deg: float, max_dist_px: int) -> np.ndarray:
    """Compute the maximum horizon elevation angle along a given azimuth.

    Vectorized: for each step distance, shifts the entire array and computes
    elevation angles in bulk, keeping a running maximum.
    """
    az_rad = np.radians(azimuth_deg)
    dy = -np.cos(az_rad)  # row increases south
    dx = np.sin(az_rad)

    nrows, ncols = elev.shape
    horizon = np.zeros_like(elev)

    for step in range(1, max_dist_px + 1):
        row_off = int(round(step * dy))
        col_off = int(round(step * dx))
        dist_m = step * cell_size

        src_r0 = max(0, -row_off)
        src_r1 = min(nrows, nrows - row_off)
        src_c0 = max(0, -col_off)
        src_c1 = min(ncols, ncols - col_off)

        dst_r0 = src_r0 + row_off
        dst_r1 = src_r1 + row_off
        dst_c0 = src_c0 + col_off
        dst_c1 = src_c1 + col_off

        if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
            continue

        rise = elev[dst_r0:dst_r1, dst_c0:dst_c1] - elev[src_r0:src_r1, src_c0:src_c1]
        angle = np.arctan2(rise, dist_m)

        view = horizon[src_r0:src_r1, src_c0:src_c1]
        np.maximum(view, angle, out=view, where=~np.isnan(angle))

    return horizon


def compute_svf(dem_path: Path) -> tuple[np.ndarray, dict]:
    """Compute Sky View Factor from a DEM raster."""
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        cell_size = abs(src.transform.a)

    nodata = profile.get("nodata", -9999.0)
    elev[elev == nodata] = np.nan

    max_dist_px = int(round(SEARCH_RADIUS_M / cell_size))
    azimuths = np.linspace(0, 360, N_AZIMUTHS, endpoint=False)

    log.info("Computing SVF: %d azimuths, search radius %d px (%.0f m), "
             "grid %d × %d …",
             N_AZIMUTHS, max_dist_px, SEARCH_RADIUS_M,
             elev.shape[0], elev.shape[1])

    sin2_sum = np.zeros_like(elev)
    for i, az in enumerate(azimuths):
        h = compute_horizon_angle(elev, cell_size, az, max_dist_px)
        h = np.maximum(h, 0.0)
        sin2_sum += np.sin(h) ** 2
        if (i + 1) % 6 == 0:
            log.info("  azimuth %d/%d (%.0f°)", i + 1, N_AZIMUTHS, az)

    svf = 1.0 - sin2_sum / N_AZIMUTHS
    svf = np.clip(svf, 0.0, 1.0)
    svf[np.isnan(elev)] = np.nan

    return svf, profile


def _aggregate_to_100m(svf_arr: np.ndarray, src_profile: dict) -> tuple[np.ndarray, dict]:
    """Aggregate SVF to 100 m using mean resampling."""
    with rasterio.open(DEM_100M_PATH) as ref:
        target_profile = ref.profile.copy()

    dst = np.full((target_profile["height"], target_profile["width"]),
                  np.nan, dtype=np.float64)
    reproject(
        source=svf_arr.astype(np.float64),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst, target_profile


def run() -> Path:
    """Compute SVF and write to covariates directory."""
    dem_path = _ensure_dem_30m()
    svf_arr, src_profile = compute_svf(dem_path)

    log.info("Aggregating SVF from %d m to 100 m …", SVF_DEM_RES)
    svf_100m, target_profile = _aggregate_to_100m(svf_arr, src_profile)

    COVARIATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COVARIATES_DIR / "sky_view_factor.tif"
    p = target_profile.copy()
    p.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    arr = np.where(np.isnan(svf_100m), -9999.0, svf_100m).astype(np.float32)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr, 1)

    log.info("Wrote sky_view_factor.tif → %s", out_path)
    return out_path


if __name__ == "__main__":
    run()
