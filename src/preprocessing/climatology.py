"""Build a monthly Tmin climatology from the gridMET daily Zarr store.

For each grid cell and calendar month, computes the long-term (1991-2025)
mean and standard deviation of daily minimum temperature.  The resulting
climatology is saved as a compact Zarr store and used by the ECOSTRESS
filtering pipeline to reject pixels whose LST falls in the far left tail
of the climatological distribution (i.e., cloud-contaminated cold outliers).

Usage
-----
    python -m src.preprocessing.climatology
"""

import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import GRIDDED_TMIN_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
GRIDMET_TMIN_ZARR = GRIDDED_TMIN_DIR / "gridmet_tmin_daily.zarr"
OUTPUT_PATH = GRIDDED_TMIN_DIR / "gridmet_tmin_climatology.zarr"


def build_climatology() -> xr.Dataset:
    """Compute per-pixel, per-month mean and std of gridMET daily Tmin.

    Returns
    -------
    xr.Dataset with dimensions (month, lat, lon) and variables
    ``tmin_mean`` and ``tmin_std``.
    """
    log.info("Opening gridMET Tmin Zarr: %s", GRIDMET_TMIN_ZARR)
    ds = xr.open_zarr(GRIDMET_TMIN_ZARR)

    log.info(
        "  Time range: %s to %s  (%d days)",
        str(ds.time.values[0])[:10],
        str(ds.time.values[-1])[:10],
        ds.sizes["time"],
    )

    grouped = ds["tmin"].groupby("time.month")
    tmin_mean = grouped.mean(dim="time")
    tmin_std = grouped.std(dim="time")

    clim = xr.Dataset(
        {
            "tmin_mean": tmin_mean,
            "tmin_std": tmin_std,
        }
    )
    clim["tmin_mean"].attrs["units"] = "degC"
    clim["tmin_mean"].attrs["long_name"] = "Long-term monthly mean of daily Tmin"
    clim["tmin_std"].attrs["units"] = "degC"
    clim["tmin_std"].attrs["long_name"] = "Long-term monthly std dev of daily Tmin"

    return clim


def save_zarr(clim: xr.Dataset) -> Path:
    """Write the climatology dataset to Zarr."""
    GRIDDED_TMIN_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        import shutil
        shutil.rmtree(OUTPUT_PATH)

    clim.to_zarr(OUTPUT_PATH, mode="w")
    log.info("Saved climatology → %s", OUTPUT_PATH)
    return OUTPUT_PATH


def run() -> Path:
    """Build and save the monthly Tmin climatology."""
    clim = build_climatology()

    log.info("── Climatology summary ──")
    log.info("  Months: %s", list(clim.month.values))
    log.info("  Grid shape: lat=%d, lon=%d", clim.sizes["lat"], clim.sizes["lon"])
    log.info(
        "  Tmin mean range: %.1f to %.1f °C",
        float(clim.tmin_mean.min()),
        float(clim.tmin_mean.max()),
    )
    log.info(
        "  Tmin std range: %.1f to %.1f °C",
        float(clim.tmin_std.min()),
        float(clim.tmin_std.max()),
    )

    out = save_zarr(clim)
    clim.close()
    return out


if __name__ == "__main__":
    run()
