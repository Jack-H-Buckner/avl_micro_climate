"""Global configuration: bounding box, CRS, resolution, file paths."""

from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Study area ──────────────────────────────────────────────────────────────
# Center: Asheville, NC
CENTER_LAT = 35.5951
CENTER_LON = -82.5515

# ~85 km × 85 km bounding box around Asheville (geographic, EPSG:4326)
BBOX_WGS84 = {
    "west": -83.055207,
    "south": 35.205304,
    "east": -82.281531,
    "north": 35.960311,
}

# ── Elevation validity range ───────────────────────────────────────────────
# GBM was trained on ~550–2037 m; pixels outside this range are flagged as
# out-of-sample.  A small buffer avoids hard edges.
ELEV_VALID_MIN = 500   # metres
ELEV_VALID_MAX = 2100  # metres

# ── Coordinate reference systems ────────────────────────────────────────────
CRS_GEO = "EPSG:4326"
CRS_UTM = "EPSG:32617"  # UTM Zone 17N

# ── Resolution ──────────────────────────────────────────────────────────────
TARGET_RESOLUTION = 100  # metres

# ── Directory layout ────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

RAW_DEM_DIR = RAW_DIR / "dem"
RAW_STATIONS_DIR = RAW_DIR / "stations"
RAW_GRIDDED_DIR = RAW_DIR / "gridded"
RAW_SATELLITE_DIR = RAW_DIR / "satellite"

COVARIATES_DIR = PROCESSED_DIR / "covariates"
GRIDDED_TMIN_DIR = PROCESSED_DIR / "gridded_tmin"
GRIDMET_DIR = PROCESSED_DIR / "gridded"
RAW_GRIDMET_DIR = RAW_GRIDDED_DIR / "gridmet"
MODEL_DIAG_DIR = OUTPUT_DIR / "model_diagnostics"

RAW_NLCD_DIR = RAW_DIR / "nlcd"
RAW_HLS_DIR = RAW_DIR / "hls"
NDVI_DIR = PROCESSED_DIR / "ndvi"

# ── Key file paths ──────────────────────────────────────────────────────────
DEM_100M_PATH = PROCESSED_DIR / "dem_100m.tif"
DEM_10M_PATH = PROCESSED_DIR / "dem_10m.tif"
STATION_FROST_PATH = PROCESSED_DIR / "station_frost_dates.parquet"

# ── Figures ─────────────────────────────────────────────────────────────────
FIGURES_DIR = PROJECT_ROOT / "figures"
