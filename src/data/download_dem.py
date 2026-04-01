"""Fetch USGS 3DEP 1/3 arc-second DEM, mosaic, reproject to UTM 100 m grid.

Uses the USGS 3DEP Elevation Program via py3dep which wraps the National Map
Web Coverage Service.  Falls back to direct WMS/WCS if py3dep is unavailable.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    BBOX_WGS84,
    CRS_GEO,
    CRS_UTM,
    DEM_100M_PATH,
    RAW_DEM_DIR,
    TARGET_RESOLUTION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _fetch_dem_tile(bbox: dict, resolution: int = 10) -> Path:
    """Download a DEM tile from the USGS 3DEP 1/3 arc-second WCS.

    Parameters
    ----------
    bbox : dict with keys west, south, east, north (EPSG:4326)
    resolution : native resolution in metres (≈10 m for 1/3 arc-second)

    Returns
    -------
    Path to the downloaded GeoTIFF in RAW_DEM_DIR.
    """
    RAW_DEM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DEM_DIR / "dem_raw_wgs84.tif"

    if out_path.exists():
        log.info("Raw DEM tile already exists at %s — skipping download.", out_path)
        return out_path

    try:
        import py3dep

        log.info("Fetching DEM via py3dep (USGS 3DEP 1/3 arc-second)…")
        dem = py3dep.get_dem(
            (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            resolution=resolution,
            crs=CRS_GEO,
        )
        dem.rio.to_raster(out_path)
        log.info("Saved raw DEM → %s", out_path)
        return out_path

    except ImportError:
        log.info("py3dep not available — falling back to direct WCS request.")

    # ── Fallback: direct HTTPS request to the 3DEP WCS endpoint ─────────
    import requests

    wcs_url = (
        "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"
    )

    # Compute pixel dimensions from approximate metric resolution
    lat_mid = (bbox["south"] + bbox["north"]) / 2
    deg_per_m_lon = 1.0 / (111_320 * np.cos(np.radians(lat_mid)))
    deg_per_m_lat = 1.0 / 110_540
    res_lon = resolution * deg_per_m_lon
    res_lat = resolution * deg_per_m_lat
    width = int(np.ceil((bbox["east"] - bbox["west"]) / res_lon))
    height = int(np.ceil((bbox["north"] - bbox["south"]) / res_lat))

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": "DEP3Elevation",
        "CRS": "EPSG:4326",
        "BBOX": f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "GeoTIFF",
    }

    log.info("Requesting DEM from USGS 3DEP WCS (%d × %d pixels)…", width, height)
    resp = requests.get(wcs_url, params=params, timeout=300)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    log.info("Saved raw DEM → %s", out_path)
    return out_path


def reproject_dem(src_path: Path, dst_path: Path, target_res: int) -> Path:
    """Reproject a DEM GeoTIFF to UTM 17N at *target_res* metres using bilinear.

    Parameters
    ----------
    src_path : input GeoTIFF (any CRS)
    dst_path : output GeoTIFF (EPSG:32617, *target_res* m)
    target_res : cell size in metres

    Returns
    -------
    dst_path
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(32617)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=target_res,
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

    log.info("Reprojected DEM → %s  (%d × %d @ %d m)", dst_path, width, height, target_res)
    return dst_path


def run() -> Path:
    """Execute full DEM pipeline: download → reproject to 100 m UTM."""
    raw_path = _fetch_dem_tile(BBOX_WGS84, resolution=10)
    dem_path = reproject_dem(raw_path, DEM_100M_PATH, TARGET_RESOLUTION)
    return dem_path


if __name__ == "__main__":
    run()
