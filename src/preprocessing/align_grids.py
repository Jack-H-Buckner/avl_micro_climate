"""Reproject and resample all rasters to the common 100 m UTM grid.

The 100 m DEM (``dem_100m.tif``) defines the reference grid.  This module
aligns three data sources to that grid:

1. **ECOSTRESS composites** (70 m → 100 m, same CRS, bilinear)
2. **ECOSTRESS individual scenes** (70 m → 100 m, same CRS, bilinear)
3. **gridMET Zarr** (WGS 84 ~4 km → 100 m UTM, bilinear for smooth
   continuous fields, nearest for precipitation which is spatially
   discontinuous)

Usage
-----
    python -m src.preprocessing.align_grids
    python -m src.preprocessing.align_grids --skip-scenes   # composites + gridMET only
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CRS_UTM,
    DEM_100M_PATH,
    GRIDMET_DIR,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
SATELLITE_DIR = PROCESSED_DIR / "satellite"
ECOSTRESS_FILTERED_DIR = SATELLITE_DIR / "ecostress_filtered"
ECOSTRESS_MASK_DIR = ECOSTRESS_FILTERED_DIR / "masks"
ECOSTRESS_ALIGNED_DIR = SATELLITE_DIR / "ecostress_aligned"
ECOSTRESS_ALIGNED_MASK_DIR = ECOSTRESS_ALIGNED_DIR / "masks"

GRIDMET_ZARR = GRIDMET_DIR / "gridmet_frost_season.zarr"
GRIDMET_ALIGNED_DIR = PROCESSED_DIR / "gridded" / "gridmet_aligned"

# gridMET variables where nearest-neighbour is more appropriate than bilinear
# (precipitation is spatially discontinuous — interpolating creates spurious
# non-zero values in dry cells)
_NEAREST_GRIDMET_VARS = {"pr"}

COMPOSITE_NAMES = [
    "ecostress_nighttime_median",
    "ecostress_nighttime_q15",
    "ecostress_nighttime_q85",
    "ecostress_nighttime_iqr",
    "ecostress_nighttime_count",
    "ecostress_nighttime_min",
    "ecostress_nighttime_std",
    "ecostress_predawn_composite",
]


# ── Reference grid ─────────────────────────────────────────────────────────

def _read_reference_grid() -> dict:
    """Return the 100 m DEM profile as the authoritative grid definition."""
    with rasterio.open(DEM_100M_PATH) as src:
        return dict(src.profile)


# ── ECOSTRESS helpers ──────────────────────────────────────────────────────

def _resample_to_100m(src_path: Path, dst_path: Path, ref: dict) -> Path:
    """Resample a single raster (70 m → 100 m) to match the reference grid.

    Uses bilinear interpolation for continuous LST values.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_nodata = src.nodata
        src_transform = src.transform
        src_crs = src.crs

    # Treat nodata as NaN for resampling, then restore
    if src_nodata is not None:
        mask = np.isclose(src_data, src_nodata) | np.isnan(src_data)
    else:
        mask = np.isnan(src_data)
    src_float = src_data.astype(np.float32)
    src_float[mask] = np.nan

    dst_data = np.full((ref["height"], ref["width"]), np.nan, dtype=np.float32)

    reproject(
        source=src_float,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref["transform"],
        dst_crs=ref["crs"],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    write_profile = ref.copy()
    write_profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")

    with rasterio.open(dst_path, "w", **write_profile) as dst:
        dst.write(dst_data, 1)

    return dst_path


def align_ecostress_composites(ref: dict) -> list[Path]:
    """Resample ECOSTRESS composites from 70 m to the 100 m reference grid."""
    aligned = []
    for name in COMPOSITE_NAMES:
        src_path = SATELLITE_DIR / f"{name}.tif"
        if not src_path.exists():
            log.debug("Skipping missing composite: %s", src_path.name)
            continue
        dst_path = SATELLITE_DIR / f"{name}_100m.tif"
        _resample_to_100m(src_path, dst_path, ref)
        aligned.append(dst_path)
        log.info("  Composite %s → %s", src_path.name, dst_path.name)
    return aligned


def align_ecostress_scenes(ref: dict) -> list[Path]:
    """Resample all filtered ECOSTRESS scenes from 70 m to 100 m.

    Writes to ``ecostress_aligned/`` so the 70 m originals are preserved.
    """
    ECOSTRESS_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    tifs = sorted(ECOSTRESS_FILTERED_DIR.glob("*.tif"))
    if not tifs:
        log.warning("No filtered ECOSTRESS scenes found in %s", ECOSTRESS_FILTERED_DIR)
        return []

    log.info("Aligning %d filtered scenes to 100 m grid ...", len(tifs))
    aligned = []
    for i, src_path in enumerate(tifs, 1):
        dst_path = ECOSTRESS_ALIGNED_DIR / src_path.name
        if dst_path.exists():
            aligned.append(dst_path)
            continue
        _resample_to_100m(src_path, dst_path, ref)
        aligned.append(dst_path)
        if i % 100 == 0:
            log.info("  ... %d / %d scenes aligned", i, len(tifs))

    log.info("Aligned %d scenes → %s", len(aligned), ECOSTRESS_ALIGNED_DIR)
    return aligned


def align_ecostress_masks(ref: dict) -> list[Path]:
    """Resample removal masks from 70 m to 100 m using nearest-neighbour.

    Masks are uint8 (1 = removed, 0 = kept) so nearest-neighbour preserves
    the binary nature of the data.
    """
    if not ECOSTRESS_MASK_DIR.exists():
        log.warning("No mask directory at %s — skipping mask alignment.", ECOSTRESS_MASK_DIR)
        return []

    ECOSTRESS_ALIGNED_MASK_DIR.mkdir(parents=True, exist_ok=True)

    tifs = sorted(ECOSTRESS_MASK_DIR.glob("*.tif"))
    if not tifs:
        log.warning("No mask TIFs found in %s", ECOSTRESS_MASK_DIR)
        return []

    log.info("Aligning %d removal masks to 100 m grid (nearest) ...", len(tifs))
    aligned = []
    for i, src_path in enumerate(tifs, 1):
        dst_path = ECOSTRESS_ALIGNED_MASK_DIR / src_path.name
        if dst_path.exists():
            aligned.append(dst_path)
            continue

        with rasterio.open(src_path) as src:
            src_data = src.read(1)
            src_transform = src.transform
            src_crs = src.crs
            src_nodata = src.nodata

        dst_data = np.zeros((ref["height"], ref["width"]), dtype=np.uint8)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            resampling=Resampling.nearest,
            src_nodata=src_nodata if src_nodata is not None else 255,
            dst_nodata=255,
        )

        write_profile = ref.copy()
        write_profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(dst_path, "w", **write_profile) as dst:
            dst.write(dst_data, 1)

        aligned.append(dst_path)
        if i % 100 == 0:
            log.info("  ... %d / %d masks aligned", i, len(tifs))

    log.info("Aligned %d masks → %s", len(aligned), ECOSTRESS_ALIGNED_MASK_DIR)
    return aligned


# ── gridMET helpers ────────────────────────────────────────────────────────

def align_gridmet(ref: dict) -> Path:
    """Reproject gridMET variables from WGS 84 ~4 km to the 100 m UTM grid.

    Continuous meteorological fields (temperature, humidity, radiation, wind)
    use **bilinear** interpolation for smooth transitions between coarse cells.
    Precipitation uses **nearest-neighbour** because it is spatially
    discontinuous and interpolating would create spurious non-zero values.

    Output is a Zarr store with the same variables and time dimension but on
    the 100 m spatial grid (y, x coordinates in UTM metres).
    """
    GRIDMET_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    out_zarr = GRIDMET_ALIGNED_DIR / "gridmet_100m_bilinear.zarr"

    if out_zarr.exists():
        log.info("gridMET 100 m Zarr already exists at %s — skipping.", out_zarr)
        return out_zarr

    ds = xr.open_zarr(GRIDMET_ZARR)
    log.info("Opened gridMET Zarr: %d times × %d lat × %d lon, vars=%s",
             ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"],
             list(ds.data_vars))

    dst_h, dst_w = ref["height"], ref["width"]
    dst_transform = ref["transform"]

    # Build source affine transform from the lat/lon coordinates
    lats = ds.lat.values  # descending (north → south)
    lons = ds.lon.values  # ascending (west → east)
    lat_sorted = np.sort(lats)  # ascending for bounds
    lon_sorted = np.sort(lons)

    # gridMET is cell-centred; half-step to edges
    dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 1 / 24
    dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 1 / 24

    src_left = float(lon_sorted[0]) - dlon / 2
    src_right = float(lon_sorted[-1]) + dlon / 2
    src_bottom = float(lat_sorted[0]) - dlat / 2
    src_top = float(lat_sorted[-1]) + dlat / 2

    src_h, src_w = len(lats), len(lons)
    src_transform = from_bounds(src_left, src_bottom, src_right, src_top, src_w, src_h)
    src_crs = "EPSG:4326"

    variables = list(ds.data_vars)
    times = ds.time.values

    # Pixel-centre coordinates from the reference transform
    xs = np.array([dst_transform.c + (col + 0.5) * dst_transform.a for col in range(dst_w)])
    ys = np.array([dst_transform.f + (row + 0.5) * dst_transform.e for row in range(dst_h)])

    # Process one variable at a time: reproject → write directly to Zarr →
    # free memory.  This keeps peak memory to one (time, y, x) array (~3 GB).
    # We write the first variable with mode="w" (creates store + coords),
    # then append subsequent variables with mode="a".
    first_var = True
    for var in variables:
        log.info("  Reprojecting %s (%d timesteps) ...", var, len(times))
        arr = ds[var].values  # (time, lat, lon)

        # Ensure lat is north→south (row 0 = north) to match transform
        if lats[0] < lats[-1]:
            arr = arr[:, ::-1, :]

        aligned = np.full((len(times), dst_h, dst_w), np.nan, dtype=np.float32)

        method = Resampling.nearest if var in _NEAREST_GRIDMET_VARS else Resampling.bilinear
        for t in range(len(times)):
            src_slice = arr[t].astype(np.float32)
            dst_slice = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
            reproject(
                source=src_slice,
                destination=dst_slice,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=CRS_UTM,
                resampling=method,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            aligned[t] = dst_slice

        ds_single = xr.Dataset(
            {var: (["time", "y", "x"], aligned)},
            coords={"time": times, "y": ys, "x": xs},
            attrs={
                "crs": CRS_UTM,
                "transform": list(dst_transform)[:6],
                "description": "gridMET variables reprojected to 100 m UTM grid "
                               "(bilinear for continuous fields, nearest for precipitation)",
            },
        )
        enc = {var: {"chunks": (90, dst_h, dst_w),
                     "compressor": {"id": "zlib", "level": 4}}}
        ds_single.to_zarr(
            str(out_zarr),
            mode="w" if first_var else "a",
            encoding=enc,
            zarr_format=2,
        )
        first_var = False
        del aligned, arr, ds_single
        log.info("    Written %s to Zarr", var)

    log.info("Wrote gridMET 100 m Zarr → %s", out_zarr)

    return out_zarr


# ── Main pipeline ──────────────────────────────────────────────────────────

def run(skip_scenes: bool = False) -> dict:
    """Align all data layers to the 100 m reference grid.

    Parameters
    ----------
    skip_scenes : bool
        If True, skip per-scene ECOSTRESS alignment (composites + gridMET only).

    Returns
    -------
    Dict with keys ``composites``, ``scenes``, ``gridmet`` mapping to output paths.
    """
    log.info("Reading reference grid from %s", DEM_100M_PATH)
    ref = _read_reference_grid()
    log.info("Reference grid: %d × %d, res=%s m, CRS=%s",
             ref["height"], ref["width"], ref["transform"].a, ref["crs"])

    outputs = {}

    # 1. ECOSTRESS composites (quick — handful of files)
    log.info("── Aligning ECOSTRESS composites (70 m → 100 m) ──")
    outputs["composites"] = align_ecostress_composites(ref)

    # 2. ECOSTRESS individual scenes (slow — hundreds of files)
    if skip_scenes:
        log.info("── Skipping per-scene ECOSTRESS alignment (--skip-scenes) ──")
        outputs["scenes"] = []
        outputs["masks"] = []
    else:
        log.info("── Aligning ECOSTRESS filtered scenes (70 m → 100 m) ──")
        outputs["scenes"] = align_ecostress_scenes(ref)

        # 2b. ECOSTRESS removal masks (nearest-neighbour for binary data)
        log.info("── Aligning ECOSTRESS removal masks (70 m → 100 m) ──")
        outputs["masks"] = align_ecostress_masks(ref)

    # 3. gridMET (WGS 84 ~4 km → 100 m UTM)
    log.info("── Aligning gridMET (WGS 84 ~4 km → 100 m UTM) ──")
    outputs["gridmet"] = align_gridmet(ref)

    log.info("── Alignment complete ──")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Reproject and resample all rasters to the 100 m UTM grid",
    )
    parser.add_argument(
        "--skip-scenes", action="store_true",
        help="Skip per-scene ECOSTRESS alignment (composites + gridMET only)",
    )
    args = parser.parse_args()
    run(skip_scenes=args.skip_scenes)


if __name__ == "__main__":
    main()
