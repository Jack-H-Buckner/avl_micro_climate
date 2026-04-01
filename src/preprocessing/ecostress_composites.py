"""Build ECOSTRESS nighttime LST composites at native 70m resolution.

Composites produced:
  - ecostress_nighttime_median.tif   — Median nighttime LST (°C)
  - ecostress_nighttime_q15.tif      — 15th percentile nighttime LST (°C)
  - ecostress_nighttime_q85.tif      — 85th percentile nighttime LST (°C)
  - ecostress_nighttime_iqr.tif      — Interquartile range of nighttime LST (°C)
  - ecostress_nighttime_count.tif    — Number of valid observations per pixel

All composites use predawn + evening scenes (nighttime window) and are
written at the native 70m ECOSTRESS grid in EPSG:32617.

Usage
-----
    python -m src.preprocessing.ecostress_composites
    python -m src.preprocessing.ecostress_composites --min-coverage 0.1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DIR, FIGURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ECOSTRESS_DIR = PROCESSED_DIR / "satellite" / "ecostress_filtered"
ECOSTRESS_NATIVE_DIR = PROCESSED_DIR / "satellite" / "ecostress_native"
SCENE_INVENTORY = PROCESSED_DIR / "satellite" / "ecostress_scenes.parquet"
OUT_DIR = PROCESSED_DIR / "satellite"

# Overpass classes considered "nighttime" for composites
NIGHTTIME_CLASSES = ("predawn", "evening")

# Minimum observations per pixel to compute quantiles
MIN_OBS = 5


def _get_reference_profile(scene_paths: list[Path]) -> dict:
    """Read the raster profile from the first available scene.

    All ECOSTRESS scenes share the same grid (same CRS, transform, shape),
    so any scene can serve as the reference.
    """
    for p in scene_paths:
        if p.exists():
            with rasterio.open(p) as src:
                return dict(src.profile)
    raise FileNotFoundError("No ECOSTRESS scene files found.")


def build_composites(
    min_coverage: float = 0.1,
) -> dict[str, Path]:
    """Stack nighttime ECOSTRESS scenes and compute pixel-wise quantiles.

    Parameters
    ----------
    min_coverage : float
        Minimum fraction of valid pixels for a scene to be included
        (relative to total_pixels in the inventory).

    Returns
    -------
    Dict mapping composite name to output path.
    """
    inventory = pd.read_parquet(SCENE_INVENTORY)
    total_pixels = inventory["total_pixels"].iloc[0]

    # Use filtered scenes if available, fall back to native
    scene_dir = ECOSTRESS_DIR if ECOSTRESS_DIR.exists() else ECOSTRESS_NATIVE_DIR
    log.info("Reading scenes from %s", scene_dir)

    # ── Select nighttime scenes with sufficient coverage ──────────────
    nighttime = inventory[
        inventory["overpass_class"].isin(NIGHTTIME_CLASSES)
        & (inventory["valid_pixels"] > total_pixels * min_coverage)
    ].copy()
    log.info(
        "Nighttime scenes with >%.0f%% coverage: %d (predawn=%d, evening=%d)",
        min_coverage * 100,
        len(nighttime),
        (nighttime["overpass_class"] == "predawn").sum(),
        (nighttime["overpass_class"] == "evening").sum(),
    )

    if nighttime.empty:
        log.warning("No qualifying nighttime scenes — cannot build composites.")
        return {}

    # ── Reference grid ────────────────────────────────────────────────
    scene_paths = [scene_dir / f for f in nighttime["filename"]]
    ref_profile = _get_reference_profile(scene_paths)
    h, w = ref_profile["height"], ref_profile["width"]
    n_scenes = len(nighttime)

    # ── Load all scenes into a 3D stack ───────────────────────────────
    # (n_scenes, h, w) — NaN where no data
    log.info("Allocating stack: %d scenes × %d × %d (%.1f GB)",
             n_scenes, h, w, n_scenes * h * w * 4 / 1e9)
    stack = np.full((n_scenes, h, w), np.nan, dtype=np.float32)

    loaded = 0
    for i, (_, row) in enumerate(nighttime.iterrows()):
        fpath = scene_dir / row["filename"]
        if not fpath.exists():
            continue

        with rasterio.open(fpath) as src:
            stack[i] = src.read(1).astype(np.float32)

        loaded += 1
        if loaded % 50 == 0:
            log.info("  Loaded %d / %d scenes ...", loaded, n_scenes)

    log.info("Finished loading %d scenes.", loaded)

    # ── Compute per-pixel quantiles ───────────────────────────────────
    log.info("Computing quantiles (this may take a minute) ...")

    count = np.sum(np.isfinite(stack), axis=0).astype(np.int32)
    has_enough = count >= MIN_OBS

    # np.nanquantile along axis 0
    median = np.nanquantile(stack, 0.50, axis=0).astype(np.float32)
    q15 = np.nanquantile(stack, 0.15, axis=0).astype(np.float32)
    q85 = np.nanquantile(stack, 0.85, axis=0).astype(np.float32)
    q25 = np.nanquantile(stack, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(stack, 0.75, axis=0).astype(np.float32)
    iqr = q75 - q25

    # Mask pixels with too few observations
    for arr in (median, q15, q85, iqr):
        arr[~has_enough] = np.nan

    nighttime_count = count.astype(np.float32)

    # Free the stack
    del stack

    # ── Write outputs ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_profile = ref_profile.copy()
    write_profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
    )

    outputs = {}
    composites = {
        "ecostress_nighttime_median": median,
        "ecostress_nighttime_q15": q15,
        "ecostress_nighttime_q85": q85,
        "ecostress_nighttime_iqr": iqr,
        "ecostress_nighttime_count": nighttime_count,
    }

    for name, data in composites.items():
        out_path = OUT_DIR / f"{name}.tif"
        with rasterio.open(out_path, "w", **write_profile) as dst:
            dst.write(data.astype(np.float32), 1)
        n_valid = int(np.count_nonzero(~np.isnan(data)))
        log.info("  %s → %s  (%d valid pixels)", name, out_path, n_valid)
        outputs[name] = out_path

    # ── Summary stats ─────────────────────────────────────────────────
    log.info("── Composite summary ──")
    log.info("  Median LST:  %.1f to %.1f °C  (spatial median %.1f)",
             np.nanmin(median), np.nanmax(median), np.nanmedian(median))
    log.info("  Q15 LST:     %.1f to %.1f °C  (spatial median %.1f)",
             np.nanmin(q15), np.nanmax(q15), np.nanmedian(q15))
    log.info("  Q85 LST:     %.1f to %.1f °C  (spatial median %.1f)",
             np.nanmin(q85), np.nanmax(q85), np.nanmedian(q85))
    log.info("  IQR:         %.1f to %.1f °C  (spatial median %.1f)",
             np.nanmin(iqr), np.nanmax(iqr), np.nanmedian(iqr))
    log.info("  Max obs per pixel: %d", count.max())

    return outputs


def run(min_coverage: float = 0.1) -> dict[str, Path]:
    """Build ECOSTRESS nighttime composites."""
    return build_composites(min_coverage=min_coverage)


def main():
    parser = argparse.ArgumentParser(
        description="Build ECOSTRESS nighttime LST composites",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.1,
        help="Minimum valid pixel fraction per scene (default: 0.1)",
    )
    args = parser.parse_args()
    run(min_coverage=args.min_coverage)


if __name__ == "__main__":
    main()
