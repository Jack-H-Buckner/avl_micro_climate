"""Heatmap visualizations for ECOSTRESS nighttime LST composites.

Generates a 2x3 panel figure showing:
  - Median nighttime LST
  - 15th percentile nighttime LST
  - 85th percentile nighttime LST
  - Interquartile range
  - Observation count per pixel

Usage
-----
    python -m src.visualization.ecostress_maps
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SATELLITE_DIR = PROCESSED_DIR / "satellite"

# (filename_stem, title, colormap, diverging_at_zero)
COMPOSITE_META = [
    ("ecostress_nighttime_median", "Median Nighttime LST (°C)",          "coolwarm",  True),
    ("ecostress_nighttime_q15",    "15th Percentile Nighttime LST (°C)", "coolwarm",  True),
    ("ecostress_nighttime_q85",    "85th Percentile Nighttime LST (°C)", "coolwarm",  True),
    ("ecostress_nighttime_iqr",    "Nighttime LST IQR (°C)",            "YlOrRd",    False),
    ("ecostress_nighttime_count",  "Valid Nighttime Observations",       "viridis",   False),
]


def _load_composite(name: str) -> tuple[np.ndarray, dict]:
    """Read an ECOSTRESS composite GeoTIFF."""
    path = SATELLITE_DIR / f"{name}.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        bounds = src.bounds
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, {"bounds": bounds}


def plot_composites(out_path: Path | None = None, dpi: int = 200) -> Path:
    """Create a 2x3 panel figure of the ECOSTRESS composites.

    Parameters
    ----------
    out_path : save location (default: FIGURES_DIR / ecostress_composites.png)
    dpi : figure resolution

    Returns
    -------
    Path to saved figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = FIGURES_DIR / "ecostress_composites.png"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        "ECOSTRESS Nighttime LST Composites — Asheville, NC (70 m)",
        fontsize=15, fontweight="bold",
    )

    # Hide the 6th (empty) subplot
    axes[1, 2].set_visible(False)

    for ax, (name, title, cmap, diverging) in zip(axes.flat, COMPOSITE_META):
        data, meta = _load_composite(name)
        bounds = meta["bounds"]
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        vmin, vmax = np.nanpercentile(data, [2, 98])

        if diverging:
            abs_max = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(
                data, extent=extent, origin="upper", cmap=cmap,
                vmin=vmin, vmax=vmax,
            )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12)
        ax.ticklabel_format(style="plain")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)

        # Annotate with summary stats
        if "count" not in name:
            median_val = np.nanmedian(data)
            ax.text(
                0.02, 0.02, f"median: {median_val:.1f}",
                transform=ax.transAxes, fontsize=8,
                color="white", backgroundcolor=(0, 0, 0, 0.5),
                verticalalignment="bottom",
            )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ECOSTRESS composite heatmaps → %s", out_path)
    return out_path


def run() -> Path:
    """Generate the ECOSTRESS composite heatmap figure."""
    return plot_composites()


if __name__ == "__main__":
    run()
