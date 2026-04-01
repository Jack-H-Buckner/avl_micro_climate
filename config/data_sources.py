"""URLs, product IDs, and metadata for each data source."""

# ── USGS 3DEP 1/3 arc-second DEM ───────────────────────────────────────────
# The National Map API endpoint for discovering 3DEP products.
TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

# Product parameters for 1/3 arc-second (~10 m) DEM
DEM_3DEP_PARAMS = {
    "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
    "prodFormats": "GeoTIFF",
    "outputFormat": "JSON",
    "max": 50,
}

# ── NLCD 2021 ──────────────────────────────────────────────────────────────
# MRLC NLCD 2021 products — download via MRLC viewer or direct S3 links.
# These cover CONUS at 30 m resolution.
NLCD_PRODUCTS = {
    "land_cover": {
        "coverage_id": "mrlc_download__NLCD_2021_Land_Cover_L48",
    },
    "impervious": {
        "coverage_id": "mrlc_download__NLCD_2021_Impervious_L48",
    },
    "tree_canopy": {
        "coverage_id": "mrlc_download__nlcd_tcc_conus_2021_v2021-4",
    },
}

# ── HLS Vegetation Index products ──────────────────────────────────────────
# NASA Harmonized Landsat-Sentinel-2 NDVI at 30 m via earthaccess / LP DAAC.
HLS_VI_PRODUCTS = {
    "landsat": {"short_name": "HLSL30_VI", "version": "2.0"},
    "sentinel": {"short_name": "HLSS30_VI", "version": "2.0"},
}

# Frost season months for NDVI downloads (Sep–May)
FROST_SEASON_MONTHS = list(range(9, 13)) + list(range(1, 6))  # [9,10,11,12,1,2,3,4,5]
FROST_SEASON_START_YEAR = 2018
