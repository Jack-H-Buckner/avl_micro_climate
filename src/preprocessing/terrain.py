"""Compute terrain covariates from the 100 m DEM.

Covariates
----------
1. elevation         — height above sea level (m)
2. slope             — surface gradient (degrees)
3. aspect_sin        — sin(aspect), N–S component
4. aspect_cos        — cos(aspect), E–W component
5. tpi_300m          — Topographic Position Index, 300 m radius
6. tpi_1000m         — Topographic Position Index, 1000 m radius
7. curvature         — profile curvature
8. twi              — Topographic Wetness Index

All outputs are written as GeoTIFFs on the same 100 m UTM grid.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, generic_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import COVARIATES_DIR, DEM_100M_PATH, TARGET_RESOLUTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _read_dem() -> tuple[np.ndarray, dict]:
    """Read the 100 m DEM and return (array, profile)."""
    with rasterio.open(DEM_100M_PATH) as src:
        elev = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    nodata = profile.get("nodata", -9999.0)
    elev[elev == nodata] = np.nan
    return elev, profile


def _write_raster(data: np.ndarray, name: str, profile: dict) -> Path:
    """Write a single-band float32 GeoTIFF to COVARIATES_DIR."""
    COVARIATES_DIR.mkdir(parents=True, exist_ok=True)
    out = COVARIATES_DIR / f"{name}.tif"
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    arr = np.where(np.isnan(data), -9999.0, data).astype(np.float32)
    with rasterio.open(out, "w", **p) as dst:
        dst.write(arr, 1)
    log.info("Wrote %s → %s", name, out)
    return out


# ── Covariate computations ─────────────────────────────────────────────────

def compute_slope_aspect(elev: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return slope (degrees) and aspect (radians, clockwise from north)."""
    # Central differences for partial derivatives
    dy, dx = np.gradient(elev, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect: angle clockwise from north (0=N, 90=E, 180=S, 270=W)
    aspect = np.arctan2(-dx, dy)  # radians, N=0
    aspect = np.mod(aspect, 2 * np.pi)
    return slope_deg, aspect


def _nanmean_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Mean filter that ignores NaN values (NaN-safe uniform_filter).

    Computes sum-of-valid / count-of-valid in each window so that NaN
    pixels don't poison the neighbourhood average.
    """
    valid = np.where(np.isnan(arr), 0.0, 1.0)
    filled = np.where(np.isnan(arr), 0.0, arr)
    sum_vals = uniform_filter(filled, size=size, mode="nearest")
    sum_valid = uniform_filter(valid, size=size, mode="nearest")
    with np.errstate(invalid="ignore"):
        result = sum_vals / sum_valid
    result[sum_valid == 0] = np.nan
    return result


def compute_tpi(elev: np.ndarray, radius_m: float, cell_size: float) -> np.ndarray:
    """Topographic Position Index = elevation – mean(neighbourhood).

    Negative → valley/hollow (cold pool), positive → ridge.
    """
    kernel_size = int(round(radius_m / cell_size)) * 2 + 1
    mean_elev = _nanmean_filter(elev, size=kernel_size)
    return elev - mean_elev


def compute_curvature(elev: np.ndarray, cell_size: float) -> np.ndarray:
    """Profile curvature via second derivatives (Zevenbergen & Thorne).

    Positive = convex, negative = concave (cold air accumulation).
    """
    dy, dx = np.gradient(elev, cell_size)
    dyy, _ = np.gradient(dy, cell_size)
    _, dxx = np.gradient(dx, cell_size)
    curv = -(dxx + dyy)
    return curv


def compute_twi(elev: np.ndarray, slope_deg: np.ndarray, cell_size: float) -> np.ndarray:
    """Topographic Wetness Index = ln(a / tan(β)).

    Uses a simple approximation where *a* (specific catchment area) is
    estimated from the inverse of slope as a proxy.  For a rigorous TWI
    you'd route flow with D8/D-inf, but this first-order approximation
    captures the main signal (flat low areas = high TWI).
    """
    slope_rad = np.radians(np.clip(slope_deg, 0.1, None))  # avoid log(0)

    # Approximate contributing area: flatter, lower cells accumulate more.
    # Use a smoothed inverse-slope proxy scaled by cell area.
    inv_slope = 1.0 / np.tan(slope_rad)
    # Smooth to mimic local accumulation (larger kernel ≈ more flow routing)
    a_proxy = _nanmean_filter(inv_slope, size=5) * cell_size
    a_proxy = np.clip(a_proxy, 1.0, None)

    twi = np.log(a_proxy / np.tan(slope_rad))
    return twi


# ── Main pipeline ───────────────────────────────────────────────────────────

def run() -> dict[str, Path]:
    """Compute all terrain covariates and write to disk."""
    log.info("Reading DEM from %s", DEM_100M_PATH)
    elev, profile = _read_dem()
    cs = float(TARGET_RESOLUTION)

    paths: dict[str, Path] = {}

    # 1. Elevation (just copy the DEM)
    paths["elevation"] = _write_raster(elev, "elevation", profile)

    # 2–3. Slope and aspect
    slope_deg, aspect_rad = compute_slope_aspect(elev, cs)
    paths["slope"] = _write_raster(slope_deg, "slope", profile)

    # 4–5. Aspect sin/cos components
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    paths["aspect_sin"] = _write_raster(aspect_sin, "aspect_sin", profile)
    paths["aspect_cos"] = _write_raster(aspect_cos, "aspect_cos", profile)

    # 6–7. TPI at 300 m and 1000 m
    tpi_300 = compute_tpi(elev, 300.0, cs)
    tpi_1000 = compute_tpi(elev, 1000.0, cs)
    paths["tpi_300m"] = _write_raster(tpi_300, "tpi_300m", profile)
    paths["tpi_1000m"] = _write_raster(tpi_1000, "tpi_1000m", profile)

    # 8. Curvature
    curv = compute_curvature(elev, cs)
    paths["curvature"] = _write_raster(curv, "curvature", profile)

    # 9. TWI
    twi = compute_twi(elev, slope_deg, cs)
    paths["twi"] = _write_raster(twi, "twi", profile)

    log.info("All %d covariates computed.", len(paths))
    return paths


if __name__ == "__main__":
    run()
