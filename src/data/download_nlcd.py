"""Download NLCD 2021 products (land cover, impervious surface, tree canopy).

Uses the MRLC GeoServer WCS 2.0.1 endpoint to fetch only the study area
extent (server-side clipping), then reprojects to UTM 17N at 30 m.

The clipped rasters are stored in data/raw/nlcd/ for subsequent processing
by src/preprocessing/nlcd_covariates.py.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, CRS_UTM, RAW_NLCD_DIR
from config.data_sources import NLCD_PRODUCTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# MRLC GeoServer WCS endpoint
WCS_BASE = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"

# Buffer (degrees) around bounding box to avoid edge effects during reprojection
BBOX_BUFFER = 0.05


def _download_via_wcs(product_key: str) -> Path:
    """Download an NLCD product clipped to the study area via MRLC WCS.

    Returns the path to the downloaded GeoTIFF (in EPSG:5070) in RAW_NLCD_DIR.
    """
    info = NLCD_PRODUCTS[product_key]
    coverage_id = info["coverage_id"]

    RAW_NLCD_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_NLCD_DIR / f"nlcd_{product_key}_raw.tif"

    if raw_path.exists():
        log.info("NLCD %s raw download exists at %s — skipping.", product_key, raw_path)
        return raw_path

    # Build WCS 2.0.1 GetCoverage request with lat/lon subsetting
    south = BBOX_WGS84["south"] - BBOX_BUFFER
    north = BBOX_WGS84["north"] + BBOX_BUFFER
    west = BBOX_WGS84["west"] - BBOX_BUFFER
    east = BBOX_WGS84["east"] + BBOX_BUFFER

    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "CoverageId": coverage_id,
        "FORMAT": "image/geotiff",
        "SUBSET": [f"Long({west},{east})", f"Lat({south},{north})"],
        "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }

    log.info("Requesting NLCD %s from MRLC WCS (bbox: %.2f,%.2f to %.2f,%.2f) …",
             product_key, west, south, east, north)
    resp = requests.get(WCS_BASE, params=params, timeout=300)
    resp.raise_for_status()

    # Check we got a GeoTIFF, not an XML error
    content_type = resp.headers.get("content-type", "")
    if "xml" in content_type.lower() or resp.content[:5] == b"<?xml":
        raise RuntimeError(
            f"WCS returned XML instead of GeoTIFF for {product_key}. "
            f"Response: {resp.text[:500]}"
        )

    raw_path.write_bytes(resp.content)
    log.info("Downloaded NLCD %s → %s (%.1f MB)",
             product_key, raw_path.name, len(resp.content) / 1e6)
    return raw_path


def _reproject_to_utm(src_path: Path, product_key: str) -> Path:
    """Reproject a WCS-downloaded NLCD raster (EPSG:5070) to UTM 17N at 30 m.

    Returns the path to the reprojected GeoTIFF in RAW_NLCD_DIR.
    """
    out_path = RAW_NLCD_DIR / f"nlcd_{product_key}_clipped.tif"
    if out_path.exists():
        log.info("Reprojected %s already exists — skipping.", out_path.name)
        return out_path

    with rasterio.open(src_path) as src:
        is_categorical = product_key == "land_cover"
        resamp = Resampling.nearest if is_categorical else Resampling.bilinear

        dst_crs = CRS.from_user_input(CRS_UTM)
        transform_utm, width_utm, height_utm = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=30,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=transform_utm,
            width=width_utm,
            height=height_utm,
            compress="deflate",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform_utm,
                    dst_crs=dst_crs,
                    resampling=resamp,
                )

    log.info("Reprojected %s → %s (%d × %d @ 30 m UTM)",
             product_key, out_path.name, width_utm, height_utm)
    return out_path


def run() -> dict[str, Path]:
    """Download and reproject all three NLCD products."""
    paths = {}
    for product_key in NLCD_PRODUCTS:
        log.info("── Processing NLCD %s ──", product_key)
        raw_path = _download_via_wcs(product_key)
        clipped_path = _reproject_to_utm(raw_path, product_key)
        paths[product_key] = clipped_path
    log.info("All NLCD products downloaded and reprojected.")
    return paths


if __name__ == "__main__":
    run()
