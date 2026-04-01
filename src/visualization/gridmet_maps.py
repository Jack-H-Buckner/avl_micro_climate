"""Plot monthly-mean heatmaps for each gridMET variable.

Reads the gridmet_frost_season.zarr archive, computes the long-term
monthly mean at each grid cell, and produces a panel figure per variable
with one heatmap per month.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import FIGURES_DIR, GRIDMET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ZARR_PATH = GRIDMET_DIR / "gridmet_frost_season.zarr"

# Frost-season months in calendar order (Sep → May)
FROST_MONTHS = [9, 10, 11, 12, 1, 2, 3, 4, 5]
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Display metadata: (variable, title, units, colormap)
VARIABLE_META = [
    ("tmmn",          "Min Temperature",             "°C",    "coolwarm"),
    ("tmmx",          "Max Temperature",             "°C",    "coolwarm"),
    ("vs",            "Wind Speed (10 m)",           "m/s",   "YlGnBu"),
    ("sph",           "Specific Humidity",           "kg/kg", "YlGnBu"),
    ("vpd",           "Vapor Pressure Deficit",      "kPa",   "YlOrRd"),
    ("srad",          "Shortwave Radiation",         "W/m²",  "YlOrRd"),
    ("pr",            "Precipitation",               "mm",    "Blues"),
    ("rmin",          "Min Relative Humidity",       "%",     "YlGnBu"),
    ("diurnal_range", "Diurnal Temperature Range",   "°C",    "inferno"),
]


def load_gridmet() -> xr.Dataset:
    """Load the gridMET frost-season Zarr archive."""
    ds = xr.open_zarr(ZARR_PATH)
    log.info("Loaded Zarr: %s", list(ds.data_vars))
    return ds


def compute_monthly_means(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Compute long-term monthly mean for a variable.

    Returns DataArray with dimensions (month, lat, lon).
    """
    da = ds[var]
    monthly = da.groupby("time.month").mean(dim="time")
    # Keep only frost-season months
    monthly = monthly.sel(month=[m for m in FROST_MONTHS if m in monthly.month.values])
    return monthly


def plot_variable(ds: xr.Dataset, var: str, title: str, units: str,
                  cmap: str, out_dir: Path, dpi: int = 150) -> Path:
    """Create a panel figure of monthly-mean heatmaps for one variable."""
    monthly = compute_monthly_means(ds, var)
    months = monthly.month.values
    n_months = len(months)

    ncols = 3
    nrows = int(np.ceil(n_months / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    fig.suptitle(f"gridMET Monthly Mean — {title}", fontsize=15, fontweight="bold")

    # Global color limits from data (2nd–98th percentile across all months)
    all_vals = monthly.values[np.isfinite(monthly.values)]
    if len(all_vals) > 0:
        vmin, vmax = np.percentile(all_vals, [2, 98])
    else:
        vmin, vmax = 0, 1

    axes_flat = np.array(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i >= n_months:
            ax.set_visible(False)
            continue

        m = int(months[i])
        data = monthly.sel(month=m)

        im = ax.pcolormesh(
            data.lon, data.lat, data.values,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
        )
        ax.set_title(MONTH_LABELS.get(m, str(m)), fontsize=12)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # Single colorbar for the figure
    fig.colorbar(im, ax=axes_flat[:n_months].tolist(), label=units,
                 fraction=0.02, pad=0.02)

    out_path = out_dir / f"gridmet_monthly_{var}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s → %s", var, out_path)
    return out_path


def run(dpi: int = 150) -> list[Path]:
    """Generate monthly-mean heatmaps for all gridMET variables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES_DIR / "gridmet"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_gridmet()
    paths = []
    for var, title, units, cmap in VARIABLE_META:
        if var not in ds.data_vars:
            log.warning("Skipping %s — not in dataset", var)
            continue
        p = plot_variable(ds, var, title, units, cmap, out_dir, dpi=dpi)
        paths.append(p)

    ds.close()
    log.info("Done — %d figures saved to %s", len(paths), out_dir)
    return paths


if __name__ == "__main__":
    run()
