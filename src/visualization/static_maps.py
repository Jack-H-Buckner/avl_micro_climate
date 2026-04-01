"""Generate heatmap figures for terrain covariates."""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import COVARIATES_DIR, CRS_GEO, CRS_UTM, FIGURES_DIR, STATION_FROST_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Covariate display metadata: (filename_stem, title, colormap, diverging)
COVARIATE_META = [
    ("elevation",  "Elevation (m)",             "terrain",  False),
    ("slope",      "Slope (°)",                 "YlOrBr",   False),
    ("aspect_sin", "Aspect sin (N–S)",          "RdBu",     True),
    ("aspect_cos", "Aspect cos (E–W)",          "RdBu",     True),
    ("tpi_300m",   "TPI 300 m",                 "RdBu_r",   True),
    ("tpi_1000m",  "TPI 1000 m",               "RdBu_r",   True),
    ("curvature",  "Curvature",                 "PiYG",     True),
    ("twi",        "Topographic Wetness Index", "YlGnBu",   False),
]


def _load_stations() -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load station locations and transform to UTM easting/northing.

    Returns (eastings, northings, names) or None if no frost-date file exists.
    """
    if not STATION_FROST_PATH.exists():
        log.warning("No station frost dates file found — skipping station overlay.")
        return None

    df = pd.read_parquet(STATION_FROST_PATH)
    stations = df.drop_duplicates(subset="station_id")[["station_id", "lat", "lon", "name"]].copy()

    transformer = Transformer.from_crs(CRS_GEO, CRS_UTM, always_xy=True)
    eastings, northings = transformer.transform(stations["lon"].values, stations["lat"].values)

    return np.array(eastings), np.array(northings), stations["name"].tolist()


def _load_covariate(name: str) -> tuple[np.ndarray, dict]:
    """Read a covariate GeoTIFF and mask nodata to NaN."""
    path = COVARIATES_DIR / f"{name}.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, {"bounds": bounds}


def plot_covariates(out_path: Path | None = None, dpi: int = 200) -> Path:
    """Create a 2×4 panel figure of all 8 terrain covariates.

    Parameters
    ----------
    out_path : save location (default: FIGURES_DIR / covariates_heatmap.png)
    dpi : figure resolution

    Returns
    -------
    Path to saved figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = FIGURES_DIR / "covariates_heatmap.png"

    # Load station locations (UTM coords)
    station_data = _load_stations()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    fig.suptitle("Terrain Covariates — Asheville, NC (100 m)", fontsize=16, fontweight="bold")

    for ax, (name, title, cmap, diverging) in zip(axes.flat, COVARIATE_META):
        data, meta = _load_covariate(name)
        bounds = meta["bounds"]
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Clip outliers for display
        vmin_raw, vmax_raw = np.nanpercentile(data, [2, 98])

        if diverging:
            abs_max = max(abs(vmin_raw), abs(vmax_raw))
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(
                data, extent=extent, origin="upper", cmap=cmap,
                vmin=vmin_raw, vmax=vmax_raw,
            )

        # Overlay station locations
        if station_data is not None:
            eastings, northings, names = station_data
            ax.scatter(
                eastings, northings,
                c="black", edgecolors="white", s=30, linewidths=0.7,
                zorder=5,
            )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)
        ax.ticklabel_format(style="plain")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Easting (m)", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved covariate heatmap → %s", out_path)
    return out_path


def run() -> Path:
    """Generate the covariate heatmap figure."""
    return plot_covariates()


if __name__ == "__main__":
    run()
