"""Rebuild covariate rasters and frost predictions for the expanded study area.

Run after updating BBOX_WGS84 in config/settings.py.  Each step checks for
existing outputs and can be skipped with CLI flags.

Usage
-----
    python scripts/rebuild_expanded_area.py              # full rebuild
    python scripts/rebuild_expanded_area.py --skip-download  # covariates + predict only
    python scripts/rebuild_expanded_area.py --predict-only   # re-run GBM prediction only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BBOX_WGS84,
    COVARIATES_DIR,
    DEM_100M_PATH,
    ELEV_VALID_MAX,
    ELEV_VALID_MIN,
    RAW_NLCD_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


def step_download_dem():
    """Download and reproject DEM for the expanded bbox."""
    log.info("=== Step 1/6: Download DEM ===")
    from src.data.download_dem import run as download_dem
    download_dem()


def step_terrain_covariates():
    """Compute 8 terrain covariates from the DEM."""
    log.info("=== Step 2/6: Terrain covariates ===")
    from src.preprocessing.terrain import run as compute_terrain
    compute_terrain()


def step_download_nlcd():
    """Download NLCD products for the expanded bbox."""
    log.info("=== Step 3/6: Download NLCD ===")
    from src.data.download_nlcd import run as download_nlcd
    download_nlcd()


def step_nlcd_covariates():
    """Compute 8 land-surface covariates from NLCD."""
    log.info("=== Step 4/6: NLCD covariates ===")
    from src.preprocessing.nlcd_covariates import run as compute_nlcd
    compute_nlcd()


def step_sky_view_factor():
    """Compute sky view factor from the DEM."""
    log.info("=== Step 5/6: Sky view factor ===")
    from src.preprocessing.sky_view_factor import run as compute_svf
    compute_svf()


def step_precompute_predictions():
    """Run GBM prediction on the expanded grid and save last_frost_dates.npz."""
    log.info("=== Step 6/6: Precompute frost predictions ===")
    from scripts.precompute_last_frost import main as precompute
    # Clear sys.argv so argparse inside precompute doesn't choke on our flags
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    precompute()
    sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(description="Rebuild data for expanded study area")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip DEM and NLCD downloads (assume they exist)")
    parser.add_argument("--predict-only", action="store_true",
                        help="Only re-run GBM predictions (covariates must exist)")
    args = parser.parse_args()

    t0 = time.time()

    log.info("Expanded bounding box: W=%.4f S=%.4f E=%.4f N=%.4f",
             BBOX_WGS84["west"], BBOX_WGS84["south"],
             BBOX_WGS84["east"], BBOX_WGS84["north"])
    log.info("Elevation validity range: %d – %d m", ELEV_VALID_MIN, ELEV_VALID_MAX)

    if args.predict_only:
        step_precompute_predictions()
    elif args.skip_download:
        step_terrain_covariates()
        step_nlcd_covariates()
        step_sky_view_factor()
        step_precompute_predictions()
    else:
        step_download_dem()
        step_terrain_covariates()
        step_download_nlcd()
        step_nlcd_covariates()
        step_sky_view_factor()
        step_precompute_predictions()

    elapsed = time.time() - t0
    log.info("Done in %.0f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
