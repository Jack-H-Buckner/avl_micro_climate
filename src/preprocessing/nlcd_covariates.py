"""Process clipped NLCD rasters into 100 m covariates.

Reads the 30 m clipped NLCD products produced by download_nlcd.py and
generates the following covariates on the project's 100 m UTM grid:

Static continuous:
  - impervious_pct.tif      (0–100, mean of 30 m pixels)
  - tree_canopy_pct.tif      (0–100, mean of 30 m pixels)

Static binary (majority within 100 m cell):
  - is_forest.tif            (NLCD 41/42/43 fraction > 0.5)
  - is_developed.tif         (NLCD 21/22/23/24 fraction > 0.5)
  - is_agriculture.tif       (NLCD 81/82 fraction > 0.5)
  - is_water.tif             (NLCD 11 fraction > 0.5)

Static continuous:
  - dist_to_water_m.tif      (Euclidean distance to nearest water pixel)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    COVARIATES_DIR,
    DEM_100M_PATH,
    RAW_NLCD_DIR,
    TARGET_RESOLUTION,
    CRS_UTM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# NLCD class codes
WATER_CLASSES = [11]
FOREST_CLASSES = [41, 42, 43]
DEVELOPED_CLASSES = [21, 22, 23, 24]
AGRICULTURE_CLASSES = [81, 82]


def _get_target_grid() -> dict:
    """Read the 100 m DEM profile as the target grid definition."""
    with rasterio.open(DEM_100M_PATH) as src:
        return src.profile.copy()


def _read_clipped(product_key: str) -> tuple[np.ndarray, dict]:
    """Read a clipped 30 m NLCD raster."""
    path = RAW_NLCD_DIR / f"nlcd_{product_key}_clipped.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"Clipped NLCD {product_key} not found at {path}. "
            "Run src/data/download_nlcd.py first."
        )
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def _write_covariate(data: np.ndarray, name: str, profile: dict) -> Path:
    """Write a single-band float32 covariate GeoTIFF."""
    COVARIATES_DIR.mkdir(parents=True, exist_ok=True)
    out = COVARIATES_DIR / f"{name}.tif"
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    arr = np.where(np.isnan(data), -9999.0, data).astype(np.float32)
    with rasterio.open(out, "w", **p) as dst:
        dst.write(arr, 1)
    log.info("Wrote %s → %s", name, out)
    return out


def _aggregate_continuous(src_data: np.ndarray, src_profile: dict,
                          target_profile: dict) -> np.ndarray:
    """Aggregate a continuous 30 m raster to 100 m using average resampling."""
    dst_shape = (target_profile["height"], target_profile["width"])
    dst = np.full(dst_shape, np.nan, dtype=np.float64)

    # Mask nodata values before reprojection
    nodata = src_profile.get("nodata")
    src_float = src_data.astype(np.float64)
    if nodata is not None:
        src_float[src_data == nodata] = np.nan

    reproject(
        source=src_float,
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


def _aggregate_class_fraction(lc_data: np.ndarray, lc_profile: dict,
                               target_profile: dict,
                               class_codes: list[int]) -> np.ndarray:
    """Compute the fraction of 30 m pixels belonging to given NLCD classes
    within each 100 m target cell.

    Creates a binary mask (1 = in class, 0 = not) at 30 m, then uses
    average resampling to get fraction at 100 m.
    """
    binary = np.isin(lc_data, class_codes).astype(np.float64)
    # Mask nodata from land cover
    nodata = lc_profile.get("nodata")
    if nodata is not None:
        binary[lc_data == nodata] = np.nan

    # Pass profile with nodata=None — we already encoded nodata as NaN in
    # the binary array; the profile's nodata=0 would incorrectly clobber
    # all 0-valued (not-in-class) pixels.
    clean_profile = lc_profile.copy()
    clean_profile["nodata"] = None
    return _aggregate_continuous(binary, clean_profile, target_profile)


def _compute_distance_to_water(lc_data: np.ndarray, lc_profile: dict,
                                target_profile: dict) -> np.ndarray:
    """Compute Euclidean distance (metres) from each 100 m cell to nearest water.

    Works at the 30 m resolution for accuracy, then reprojects the distance
    raster to the 100 m grid.
    """
    nodata = lc_profile.get("nodata")
    water_mask = np.isin(lc_data, WATER_CLASSES)
    if nodata is not None:
        water_mask[lc_data == nodata] = False

    # distance_transform_edt gives distance in pixel units from non-water cells
    # to nearest water cell. Invert: 1 where NOT water, 0 where water.
    not_water = ~water_mask
    dist_pixels = distance_transform_edt(not_water)

    # Convert to metres (30 m pixel size)
    cell_size = abs(lc_profile["transform"].a)  # should be ~30 m
    dist_m = (dist_pixels * cell_size).astype(np.float64)

    # Reproject to 100 m target grid
    return _aggregate_continuous(dist_m, lc_profile, target_profile)


def run() -> dict[str, Path]:
    """Process all NLCD covariates and write to disk."""
    target_profile = _get_target_grid()
    paths: dict[str, Path] = {}

    # ── Impervious surface ────────────────────────────────────────────────
    # NLCD impervious has nodata=0 in the GeoTIFF header, but 0 is a valid
    # value (0% impervious).  Use the land-cover nodata mask instead.
    log.info("Processing impervious surface …")
    imp_data, imp_profile = _read_clipped("impervious")
    imp_profile = imp_profile.copy()
    imp_profile["nodata"] = None  # 0 is valid for impervious
    imp_100m = _aggregate_continuous(imp_data, imp_profile, target_profile)
    paths["impervious_pct"] = _write_covariate(imp_100m, "impervious_pct", target_profile)

    # ── Tree canopy cover ─────────────────────────────────────────────────
    # Same issue: nodata=0 but 0% canopy is valid.
    log.info("Processing tree canopy cover …")
    tc_data, tc_profile = _read_clipped("tree_canopy")
    tc_profile = tc_profile.copy()
    tc_profile["nodata"] = None  # 0 is valid for tree canopy
    tc_100m = _aggregate_continuous(tc_data, tc_profile, target_profile)
    paths["tree_canopy_pct"] = _write_covariate(tc_100m, "tree_canopy_pct", target_profile)

    # ── Land cover class fractions → binary indicators ────────────────────
    log.info("Processing land cover classes …")
    lc_data, lc_profile = _read_clipped("land_cover")

    for name, codes in [
        ("is_forest", FOREST_CLASSES),
        ("is_developed", DEVELOPED_CLASSES),
        ("is_agriculture", AGRICULTURE_CLASSES),
        ("is_water", WATER_CLASSES),
    ]:
        fraction = _aggregate_class_fraction(lc_data, lc_profile, target_profile, codes)
        binary = (fraction > 0.5).astype(np.float64)
        binary[np.isnan(fraction)] = np.nan
        paths[name] = _write_covariate(binary, name, target_profile)

    # ── Distance to water ─────────────────────────────────────────────────
    log.info("Computing distance to water …")
    dist_water = _compute_distance_to_water(lc_data, lc_profile, target_profile)
    paths["dist_to_water_m"] = _write_covariate(dist_water, "dist_to_water_m", target_profile)

    log.info("All %d NLCD covariates written.", len(paths))
    return paths


if __name__ == "__main__":
    run()
