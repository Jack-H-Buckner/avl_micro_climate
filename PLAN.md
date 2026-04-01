# Asheville Frost Date Modeling App — Project Plan

## Project overview

This project builds a high-resolution (100m) frost date prediction system for the Asheville, NC metro area (Buncombe County and surroundings). The system models the climatological onset of first fall frost and last spring frost using a regression kriging framework that fuses high-spatial-resolution terrain data with temporally dense temperature observations and gridded climate products.

The long-term goal is to link climatological frost maps to seasonal forecasts (NOAA CFSv2, CPC outlooks) to predict interannual variation in frost timing.

### Study area

- **Center:** Asheville, NC (35.5951°N, 82.5515°W)
- **Extent:** Buncombe County + buffer (~40km × 40km bounding box)
- **Elevation range:** ~550m (French Broad River valley) to ~2,037m (Mount Mitchell)
- **Key terrain features:** French Broad River valley (cold air drainage corridor), Swannanoa Valley, Blue Ridge escarpment, numerous hollows and ridgelines
- **EPSG:** 32617 (UTM Zone 17N) for all raster work

### Target output

- Gridded 100m resolution maps of:
  - Mean first fall frost date (Tmin ≤ 0°C)
  - Mean last spring frost date (Tmin ≤ 0°C)
  - Standard deviation of frost dates (interannual variability)
  - Probability of frost by calendar date (e.g., "probability of frost on October 15")
- Interactive web app for exploring frost risk by location

---

## Architecture

```
asheville-frost/
├── PLAN.md                     # This file
├── README.md                   # Setup and usage instructions
├── pyproject.toml              # Python project config (uv/pip)
├── .env.example                # Template for API keys (NASA Earthdata, etc.)
│
├── config/
│   ├── settings.py             # Global config (bounding box, CRS, resolution, paths)
│   └── data_sources.py         # URLs, product IDs, and metadata for each data source
│
├── data/                       # Gitignored — all downloaded and intermediate data
│   ├── raw/                    # Original downloads
│   │   ├── dem/                # USGS 3DEP 1/3 arc-second tiles
│   │   ├── stations/           # GHCN-Daily and ECONet station data
│   │   ├── gridded/            # Gridded climate data
│   │   │   ├── prism/          # PRISM daily Tmin (if used as alternative)
│   │   │   └── gridmet/        # gridMET annual NetCDF files (tmmn, tmmx, vs, sph, etc.)
│   │   └── satellite/          # Satellite LST data
│   │       ├── modis/          # MOD11A1 / MYD11A1 nighttime LST
│   │       └── ecostress/      # ECO_L2T_LSTE COGs
│   ├── processed/              # Cleaned and aligned data
│   │   ├── dem_100m.tif        # DEM reprojected and resampled to 100m UTM grid
│   │   ├── covariates/         # Terrain-derived rasters at 100m
│   │   ├── station_frost_dates.parquet  # Computed frost dates per station per year
│   │   ├── gridded_tmin/       # Daily Tmin grids clipped and aligned to study area
│   │   ├── gridded/            # Processed gridMET data
│   │   │   └── gridmet_frost_season.zarr  # All variables, clipped, frost season dates
│   │   └── satellite/          # Processed satellite LST
│   │       ├── modis_lst_night.zarr            # MODIS nighttime LST time series
│   │       ├── ecostress_scenes.parquet        # Scene inventory with overpass times
│   │       ├── ecostress_predawn_composite.tif # Mean predawn LST at 100m
│   │       ├── ecostress_nighttime_min.tif     # Min observed nighttime LST at 100m
│   │       ├── ecostress_nighttime_std.tif     # Std dev of nighttime LST at 100m
│   │       └── ecostress_native/               # Individual scenes at native 70m
│   └── output/                 # Final model products
│       ├── frost_first_fall_mean.tif
│       ├── frost_last_spring_mean.tif
│       ├── frost_probability_by_doy.nc
│       └── model_diagnostics/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_dem.py         # Fetch USGS 3DEP tiles, mosaic, clip
│   │   ├── download_stations.py    # Fetch GHCN-Daily and ECONet data via API
│   │   ├── download_gridded.py     # Fetch PRISM daily Tmin (legacy/alternative)
│   │   ├── download_gridmet.py     # Fetch gridMET variables via pygridmet (primary)
│   │   ├── download_nlcd.py        # Fetch NLCD land cover, impervious, tree canopy
│   │   ├── download_hls_ndvi.py    # Fetch HLS-VI NDVI scenes, build biweekly composites
│   │   ├── download_modis.py       # Fetch MODIS LST via AppEEARS API
│   │   ├── download_ecostress.py   # Fetch ECOSTRESS LST via earthaccess / AppEEARS
│   │   └── validate.py             # Data quality checks and completeness reports
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── terrain.py              # Compute covariates from DEM (slope, aspect, TPI, SVF)
│   │   ├── land_surface.py         # Aggregate NLCD to 100m, compute distance-to-water
│   │   ├── ndvi.py                 # Build biweekly HLS NDVI composites, match to scenes
│   │   ├── frost_dates.py          # Extract first/last frost dates from station records
│   │   ├── align_grids.py          # Reproject, clip, resample all rasters to common grid
│   │   └── prepare_training.py     # Build pixel-level training dataset with all covariates
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── regression_kriging.py   # Core RK model: OLS/GLS regression + variogram + kriging
│   │   ├── random_forest.py        # RF benchmark model for comparison
│   │   ├── cross_validation.py     # Spatial leave-one-out and k-fold CV
│   │   └── predict.py              # Apply fitted model to full 100m grid
│   │
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── frost_maps.py           # Generate frost date maps from daily predictions
│   │   ├── uncertainty.py          # Kriging variance → confidence intervals
│   │   └── export.py               # Export to GeoTIFF, NetCDF, vector formats
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── static_maps.py          # Matplotlib/cartopy publication maps
│       └── interactive.py          # Folium or Plotly web maps
│
├── app/                            # Web app (Phase 2)
│   ├── app.py                      # Streamlit or FastAPI entry point
│   ├── components/                 # UI components
│   └── static/                     # Pre-rendered map tiles
│
├── notebooks/                      # Exploratory analysis
│   ├── 01_explore_stations.ipynb
│   ├── 02_terrain_covariates.ipynb
│   ├── 03_model_fitting.ipynb
│   └── 04_validation.ipynb
│
├── tests/
│   ├── test_terrain.py
│   ├── test_frost_dates.py
│   └── test_model.py
│
└── scripts/
    ├── run_pipeline.py             # End-to-end pipeline runner
    └── update_current_season.py    # Fetch latest data and update predictions
```

---

## Phase 1: Data acquisition and preprocessing

### Task 1.1 — DEM acquisition and terrain covariates

**Source:** USGS 3DEP 1/3 arc-second (~10m) DEM

**Steps:**
1. Download 3DEP tiles covering the study area bounding box from the USGS National Map API (`https://tnmaccess.nationalmap.gov/api/v1/products`)
2. Mosaic tiles into a single raster
3. Reproject to UTM Zone 17N (EPSG:32617)
4. Resample to 100m target grid using bilinear interpolation
5. Compute terrain covariates at 100m:

| Covariate | Description | Python tool | Why it matters for frost |
|-----------|-------------|-------------|------------------------|
| `elevation` | Height above sea level (m) | Direct from DEM | Lapse rate drives ~6.5°C/km cooling |
| `slope` | Surface gradient (degrees) | `richdem` or `xdem` | Steep slopes shed cold air; flat areas pool it |
| `aspect_sin` | sin(aspect) — N/S component | `numpy` from aspect | North-facing slopes cool faster |
| `aspect_cos` | cos(aspect) — E/W component | `numpy` from aspect | Morning sun exposure on east slopes |
| `tpi_300m` | Topographic Position Index (300m radius) | `richdem` | Negative = valley/hollow (cold pool), positive = ridge |
| `tpi_1000m` | TPI at 1000m radius | `richdem` | Broader landscape position |
| `curvature` | Profile curvature | `richdem` | Concave terrain concentrates cold air |
| `twi` | Topographic Wetness Index | `pysheds` or `richdem` | Wetter valleys have different thermal properties |

**Output:** `data/processed/covariates/*.tif` — one GeoTIFF per covariate, all on the same 100m grid.

**Key libraries:** `rasterio`, `richdem`, `pyproj`, `numpy`

### Task 1.2 — Station data acquisition

**Sources:**
- **GHCN-Daily:** Long-record stations with daily Tmin/Tmax. Access via NOAA NCEI API (`https://www.ncei.noaa.gov/cdo-web/api/v2/`) or direct FTP download.
- **NC ECONet:** 45 research-grade stations with hourly/1-min data. Access via NC State Climate Office CLOUDS API (`https://api.climate.ncsu.edu/`).
- **COOP network:** Additional volunteer observer stations in western NC (available within GHCN-Daily).

**Steps:**
1. Query all stations within bounding box + 20km buffer (buffer provides edge stability for kriging)
2. Download daily Tmin records for period of interest (1990–present for climatology)
3. Apply quality control:
   - Remove flagged observations (GHCN-D QC flags)
   - Check for minimum 80% completeness in frost season (Sep–May) per station-year
   - Screen for obviously erroneous values (Tmin > Tmax, Tmin < -30°C, etc.)
4. Record station metadata: lat, lon, elevation, network, period of record

**Output:** `data/processed/station_frost_dates.parquet` with columns:
```
station_id | lat | lon | elevation | year | first_fall_frost_doy | last_spring_frost_doy | network
```

**Key libraries:** `requests`, `pandas`, `geopandas`

### Task 1.3 — Gridded temperature data (for daily Tmin fields)

**Primary source:** PRISM daily Tmin at 4km resolution (free tier). Accessed via PRISM FTP or the `prism` Python package. Period: 1981–present.

**Alternative/supplement:** nClimGrid-Daily at ~5km (1951–present) or gridMET at ~4km (1979–present). Both available via OpenDAP or bulk download.

**Steps:**
1. Download daily Tmin grids for frost season months (September–May) across the study period
2. Clip to study area bounding box
3. Reproject to UTM Zone 17N
4. Store as a time-indexed NetCDF or Zarr archive for efficient access

**Output:** `data/processed/gridded_tmin/prism_tmin_daily.zarr`

**Key libraries:** `xarray`, `rioxarray`, `zarr`, `dask` (for lazy loading large time series)

### Task 1.4 — Satellite LST data

Satellite land surface temperature provides independent spatial information about thermal patterns that complements station observations and gridded products. Two sensors are used here: MODIS for consistent daily coverage at moderate resolution, and ECOSTRESS for the highest available thermal resolution from space.

**Prerequisites:** A free NASA Earthdata Login account is required for both data sources. Register at `https://urs.earthdata.nasa.gov/`.

#### Task 1.4a — MODIS LST (1km, daily)

**Source:** MOD11A1 (Terra, ~10:30 AM/PM local) and MYD11A1 (Aqua, ~1:30 AM/PM local) daily LST at 1km resolution. Available 2000–present.

**Access:** NASA AppEEARS API (`https://appeears.earthdatacloud.nasa.gov/api/`) or Earthdata Search. AppEEARS handles subsetting, reprojection, and format conversion.

**Steps:**
1. Submit an AppEEARS area request for the study bounding box:
   - Products: `MOD11A1.061` (Terra) and `MYD11A1.061` (Aqua)
   - Layers: `LST_Night_1km`, `QC_Night`, `LST_Day_1km`, `QC_Day`
   - Date range: frost seasons (Sep–May) for the study period (2000–present)
   - Output projection: UTM Zone 17N (EPSG:32617)
   - Output format: GeoTIFF
2. Filter by quality flags — retain only clear-sky pixels (QC bits 0-1 = 00)
3. Convert from Kelvin (scaled) to °C
4. Nighttime passes are primary interest for frost work:
   - Terra nighttime: ~10:30 PM local (captures early evening cooling)
   - Aqua nighttime: ~1:30 AM local (closer to predawn minimum, more useful)
5. Clip to study area and align to the 100m target grid (nearest neighbor resample for QC, bilinear for LST)
6. Store as time-indexed Zarr archive

**Output:** `data/processed/satellite/modis_lst_night.zarr`

**Role in the model:**
- Use as an additional covariate in regression kriging (mean nighttime LST captures spatial patterns not fully explained by terrain alone — urban heat islands, water body effects, land cover thermal properties)
- Use for validation of the frost model's spatial predictions
- 1km resolution means MODIS adds broad spatial context but not fine-scale detail

#### Task 1.4b — ECOSTRESS LST (70m, variable revisit)

**Source:** ECO_L2T_LSTE Version 2 (tiled product) — atmospherically corrected land surface temperature and emissivity at 70m resolution. Available 2018–present.

**Instrument details:**
- ECOSTRESS is mounted on the International Space Station (ISS) at ~400km altitude
- ISS has a non-sun-synchronous, precessing orbit at 51.6° inclination
- Unlike polar-orbiting satellites (Landsat, MODIS), ECOSTRESS observes the same location at **different times of day** with each overpass — it samples the full diurnal cycle over a period of weeks
- Revisit interval: 1–5 days depending on latitude and ISS orbital geometry
- Coverage: 53.6°N to 53.6°S latitude (Asheville at 35.6°N is well covered)
- Swath width: ~384 km
- Five TIR spectral bands (8–12.5 μm), LST retrieved via Temperature/Emissivity Separation (TES) algorithm
- Collection 2 (current) includes improved radiometric calibration and cloud masking
- Tiled products are distributed as Cloud-Optimized GeoTIFFs (COGs)

**Why ECOSTRESS matters for this project:**
- 70m resolution can resolve individual hollows, ridgelines, stream corridors, and neighborhood-scale thermal variation in Asheville — 10× finer than Landsat thermal, 15× finer than MODIS
- Nighttime and predawn overpasses capture the surface temperatures most relevant to frost formation
- The variable overpass time is both a strength (samples different phases of nocturnal cooling) and a limitation (you cannot control when predawn observations occur)
- Best used as a **spatial calibration and validation layer** rather than a primary temporal data source, given its irregular temporal sampling

**Access (three options, from simplest to most flexible):**

1. **AppEEARS (recommended for initial work):**
   - Submit an area request at `https://appeears.earthdatacloud.nasa.gov/`
   - Product: `ECO_L2T_LSTE.002`
   - Layers: `LST`, `LST_err`, `QC`, `cloud`
   - Draw bounding box around Buncombe County
   - Select frost season date ranges
   - AppEEARS handles subsetting, reprojection, and delivers GeoTIFFs

2. **NASA Earthdata Search (for browsing individual scenes):**
   - Search at `https://search.earthdata.nasa.gov/`
   - Filter by collection: "ECOSTRESS Tiled Land Surface Temperature"
   - Useful for inspecting specific overpasses and checking data availability before bulk requests

3. **Programmatic access via CMR API + `earthaccess` Python library:**
   ```python
   import earthaccess
   
   earthaccess.login()
   results = earthaccess.search_data(
       short_name="ECO_L2T_LSTE",
       version="002",
       bounding_box=(-82.8, 35.3, -82.2, 35.9),  # Asheville area
       temporal=("2023-10-01", "2024-04-30"),       # Frost season
   )
   files = earthaccess.download(results, "data/raw/satellite/ecostress/")
   ```
   NASA also provides a full tutorial repository: `https://github.com/nasa/ECOSTRESS-Data-Resources`

**Processing steps:**
1. Download all available ECOSTRESS scenes over the study area for frost seasons (Oct–Apr)
2. Extract overpass time from granule metadata — each scene includes UTC acquisition time
3. Classify scenes by local overpass time:
   - **Predawn (2:00–6:00 AM local):** Highest priority for frost work — closest to Tmin
   - **Evening (8:00 PM–2:00 AM local):** Useful — captures early radiative cooling phase
   - **Daytime (10:00 AM–4:00 PM local):** Lower priority for frost but useful for land cover thermal characterization
4. Apply quality filtering:
   - Remove cloud-contaminated pixels using the `cloud` layer
   - Filter by `QC` flags for LST retrieval quality
   - Check `LST_err` layer — discard pixels with error > 2K
5. Convert LST from Kelvin to °C
6. Reproject to UTM Zone 17N and align to the 100m target grid
   - ECOSTRESS native resolution (70m) is finer than target grid (100m), so aggregate using mean within each 100m cell
   - Preserve the native 70m data as well for fine-scale analysis
7. Compute per-pixel summary statistics from the nighttime/predawn scenes:
   - Mean nighttime LST (composite of all clear-sky predawn scenes)
   - Minimum observed LST (captures coldest conditions)
   - Standard deviation (spatial variability indicator)
8. Store individual scenes and composites

**Expected data volume:**
- Individual scenes over Buncombe County: ~50–100 MB per scene (COG format)
- Expect ~20–40 clear-sky scenes per frost season, of which ~5–15 will be nighttime/predawn
- Total for one frost season: ~2–5 GB raw, much less after clipping to study area

**Output:**
```
data/raw/satellite/ecostress/           # Raw downloaded COGs
data/processed/satellite/
    ecostress_scenes.parquet            # Scene inventory: datetime, overpass_time_local, cloud_fraction
    ecostress_predawn_composite.tif     # Mean predawn LST at 100m (frost-relevant composite)
    ecostress_nighttime_min.tif         # Minimum observed nighttime LST at 100m
    ecostress_nighttime_std.tif         # Std dev of nighttime LST at 100m
    ecostress_native/                   # Individual scenes at 70m for fine-scale analysis
```

**Role in the model:**
- **Spatial validation:** Compare regression kriging predictions against ECOSTRESS-observed thermal patterns — do predicted cold pools match where ECOSTRESS shows the coldest nighttime LST?
- **Additional covariate:** The predawn composite LST can be added as a covariate in the regression model — it captures thermal properties (urban heat island, water body buffering, land cover effects) not fully represented by terrain alone
- **Fine-scale calibration:** Where ECOSTRESS predawn observations exist on frost nights, they provide 70m ground truth for checking whether the 100m model captures sub-kilometer thermal gradients
- **Note on LST vs air temperature:** ECOSTRESS measures skin (surface) temperature, not 2m air temperature. On clear calm nights, skin temperature can be 2–5°C colder than air temperature due to radiative cooling. This offset should be accounted for when comparing ECOSTRESS LST with station Tmin or using it as a covariate

**Key libraries:** `earthaccess`, `rasterio`, `rioxarray`, `xarray`, `pandas`

### Task 1.5 — Auxiliary meteorological variables from gridMET

**Source:** gridMET — daily gridded surface meteorological data at ~4km (1/24th degree) resolution covering CONUS from 1979 to present, updated daily. Produced by the Climatology Lab at UC Merced. gridMET blends PRISM's high-resolution spatial patterns with the temporal completeness and additional variables from the NLDAS-2 reanalysis.

**Why gridMET over PRISM alone:** PRISM provides only temperature and precipitation. gridMET provides the full suite of meteorological variables that control the rate and magnitude of nocturnal surface cooling — wind speed, humidity, solar radiation, and vapor pressure deficit — all on the same 4km grid. These variables explain *why* some nights produce much stronger cold air pooling and frost than others, even at similar minimum temperatures.

**Variables to download:**

| Variable | gridMET code | Units | File pattern | Physical role in nighttime cooling |
|----------|-------------|-------|-------------|-----------------------------------|
| Min temperature | `tmmn` | K | `tmmn_YYYY.nc` | Baseline nighttime temperature — primary predictor |
| Max temperature | `tmmx` | K | `tmmx_YYYY.nc` | Daytime heating sets initial conditions for cooling; diurnal range (`tmmx - tmmn`) signals clear-sky radiative nights |
| Wind speed (10m) | `vs` | m/s | `vs_YYYY.nc` | Calm nights allow cold air drainage and inversions; wind mixes the boundary layer and reduces spatial variation |
| Specific humidity | `sph` | kg/kg | `sph_YYYY.nc` | Dry air is more transparent to longwave radiation → faster surface cooling; humid air traps outgoing radiation |
| Vapor pressure deficit | `vpd` | kPa | `vpd_YYYY.nc` | Integrates temperature and humidity into a single dryness metric; high VPD = strong radiative cooling potential |
| Downward shortwave radiation | `srad` | W/m² | `srad_YYYY.nc` | Previous day's solar heating determines how much thermal energy surfaces must radiate away at night |
| Precipitation | `pr` | mm | `pr_YYYY.nc` | Wet surfaces have different thermal properties; recent rain changes soil heat capacity and evaporative cooling |
| Min relative humidity | `rmin` | % | `rmin_YYYY.nc` | Low nighttime humidity amplifies radiative cooling |

**Day definition note:** gridMET defines a "day" as the 24 hours ending at 12:00 UTC (7:00 AM Eastern Standard Time). This means:
- `tmmn` for date D captures the minimum temperature through the morning of day D — aligns well with nighttime ECOSTRESS passes
- `tmmx` for date D captures the maximum temperature from the afternoon of day D-1 through midday of day D — for matching with a nighttime ECOSTRESS pass on the night of D-1 to D, use `tmmx` from date D-1 (the preceding afternoon's maximum)
- `srad` for date D reflects the previous day's solar input — same date-shifting logic applies
- Wind, humidity, and precipitation should be matched carefully to the overnight period

**Access methods:**

1. **`pygridmet` Python library (recommended):**
   ```python
   import pygridmet as gridmet

   # Get gridded data for study area bounding box
   bbox = (-82.8, 35.3, -82.2, 35.9)  # (west, south, east, north)
   dates = ("2023-10-01", "2024-04-30")

   variables = ["tmmn", "tmmx", "vs", "sph", "vpd", "srad", "pr", "rmin"]
   data = gridmet.get_bygeom(bbox, dates, variables=variables)
   # Returns an xarray.Dataset with all variables on the 4km grid
   ```
   No authentication required. Handles OpenDAP subsetting automatically.

2. **Direct NetCDF download:**
   ```
   http://www.northwestknowledge.net/metdata/data/{variable}_{year}.nc
   ```
   Files are annual, one per variable. Download, then clip to study area with `xarray`.

3. **Google Earth Engine:**
   Collection: `IDAHO_EPSCOR/GRIDMET` — useful if you're already working in GEE for other data.

**Processing steps:**
1. Download all variables for frost season months (Sep–May) across the ECOSTRESS period (2018–present)
2. Clip to study area bounding box + buffer
3. Convert temperature from Kelvin to °C
4. Compute derived variables:
   - `diurnal_range = tmmx - tmmn` (°C) — strong predictor of radiative cooling intensity
   - `tmmx_prev = tmmx` shifted by one day (to match preceding afternoon for nighttime passes)
   - `srad_prev = srad` shifted by one day (preceding day's solar input)
5. For each ECOSTRESS scene date, extract the matching gridMET variables:
   - Each 100m pixel inherits values from its enclosing 4km gridMET cell (same as for `gridded_tmin`)
   - All 100m pixels within one 4km cell share the same meteorological context — the terrain covariates explain the within-cell variation
6. Store as a time-indexed Zarr archive for efficient date-based lookups

**Output:**
```
data/raw/gridded/gridmet/               # Raw annual NetCDF files per variable
data/processed/gridded/
    gridmet_frost_season.zarr            # All variables, clipped to study area, frost season dates
```

**Expected data volume:**
- Raw: ~200 MB per variable per year (full CONUS) — but only a tiny subset is needed after clipping
- Processed (clipped to study area): ~5–10 MB per variable per frost season
- Total for all variables, all years: < 1 GB

**Key libraries:** `pygridmet`, `xarray`, `rioxarray`, `zarr`, `dask`

### Task 1.6 — Land surface covariates (NLCD, NDVI, sky view factor)

These covariates capture how the physical properties of the land surface — vegetation, built structures, sky exposure — modify nighttime cooling beyond what terrain shape alone explains. They divide into two categories based on temporal behavior.

#### Static covariates (compute once, apply to all scenes)

These change on multi-year timescales. Compute once from the most recent available product and treat as constant across all ECOSTRESS scenes.

**1. NLCD land cover, impervious surface, and tree canopy cover**

**Source:** USGS National Land Cover Database (NLCD) 2021 release, 30m resolution. Free, public domain. Download from `https://www.mrlc.gov/data`.

**Products to download:**
| Product | Resolution | Type | What it captures |
|---------|-----------|------|-----------------|
| Land cover | 30m | Categorical (16 classes) | Surface type: forest, developed, agriculture, water, etc. |
| Impervious surface | 30m | Continuous (0–100%) | Fraction of hard surfaces (roads, buildings, parking lots) |
| Tree canopy cover | 30m | Continuous (0–100%) | Fraction of canopy overhead — controls sky view and longwave trapping |

**Processing steps:**
1. Download NLCD tiles covering the study area from MRLC
2. Clip to study area bounding box
3. Reproject to UTM Zone 17N (EPSG:32617)
4. Aggregate from 30m to 100m target grid:
   - **Impervious surface:** mean of 30m pixels within each 100m cell (gives fractional imperviousness)
   - **Tree canopy cover:** mean of 30m pixels within each 100m cell (gives fractional canopy)
   - **Land cover (categorical):** compute fractional composition within each 100m cell, then derive binary indicators:
     ```python
     # For each 100m cell, count the 30m pixels (roughly 9-11 pixels) in each class
     'is_forest': fraction of cell classified as deciduous/evergreen/mixed forest (NLCD 41/42/43) > 0.5
     'is_developed': fraction classified as any developed class (NLCD 21/22/23/24) > 0.5
     'is_agriculture': fraction classified as pasture/cultivated (NLCD 81/82) > 0.5
     'is_water': fraction classified as open water (NLCD 11) > 0.5
     ```
5. Compute distance-to-water feature:
   - Extract water pixels from NLCD land cover
   - Compute Euclidean distance from each 100m cell to nearest water pixel
   - Output as `dist_to_water_m.tif`

**Physical rationale:**
- **Impervious surface:** Asphalt and concrete store daytime heat and release it at night (urban heat island). Downtown Asheville at 40–60% impervious will be several degrees warmer than surrounding rural areas at 0–5% impervious on a calm clear night.
- **Tree canopy:** Forest canopy acts as a thermal blanket — it blocks outgoing longwave radiation and reduces sky view factor. A forested hollow won't get as cold as an open field in the same terrain position. Canopy also intercepts frost formation (frost on leaves above, not on ground).
- **Land cover classes:** Broad surface type affects thermal inertia, roughness, and moisture availability. Agricultural fields cool fastest after harvest (bare soil, high SVF); wetlands stay warmer (thermal inertia of water).
- **Distance to water:** Rivers and reservoirs moderate adjacent air temperatures, reducing frost risk within ~100–500m.

**Output:**
```
data/processed/covariates/
    impervious_pct.tif          # 0-100, continuous
    tree_canopy_pct.tif         # 0-100, continuous
    is_forest.tif               # Binary
    is_developed.tif            # Binary
    is_agriculture.tif          # Binary
    is_water.tif                # Binary
    dist_to_water_m.tif         # Continuous, meters
```

**Key libraries:** `rasterio`, `numpy`, `scipy.ndimage` (for distance transform)

**2. Sky view factor (SVF)**

**Source:** Derived from the 10m DEM (already downloaded in Task 1.1).

**What it is:** SVF quantifies the fraction of the sky hemisphere visible from each point on the surface, ranging from 0 (completely enclosed — deep canyon) to 1 (flat open plain, full sky exposure). It directly controls the rate of longwave radiative cooling at night — a pixel with SVF = 0.4 (in a narrow valley) loses heat to the sky half as fast as a pixel with SVF = 0.85 (on an open ridge).

**Why it's better than TPI for radiative cooling:** TPI tells you whether you're in a valley or on a ridge relative to surroundings, which mainly relates to cold air drainage. SVF directly measures the radiative geometry — how much sky the surface "sees" for emitting longwave radiation. A pixel on a south-facing slope with TPI = 0 (mid-slope) could still have low SVF if it's backed by a high ridge to the north. Both are useful, but SVF is more physically direct for the radiative cooling mechanism.

**Computation:**
```python
# Pseudocode — compute SVF from 10m DEM
# For each pixel, cast rays at multiple azimuths and compute horizon angles
import richdem  # or use SAGA GIS, GRASS GIS, or custom implementation

for azimuth in range(0, 360, 5):  # 72 directions
    horizon_angle[azimuth] = compute_horizon_angle(dem_10m, azimuth)

# SVF = 1 - mean(sin²(horizon_angle)) integrated over all azimuths
svf = 1 - np.mean(np.sin(horizon_angles)**2, axis=0)
```

- Compute on the 10m DEM (captures fine-scale valley geometry)
- Aggregate to 100m using mean (the average sky openness within each 100m cell)
- Search radius for horizon calculation: 500–1000m (captures nearby ridgelines but not distant mountains)
- This is computationally moderate — ~5–15 minutes on a laptop for Buncombe County at 10m

**Output:** `data/processed/covariates/sky_view_factor.tif` (0–1, continuous)

**Key libraries:** `richdem`, `numpy`, or SAGA GIS via command line (`saga_cmd ta_lighting 3`)

#### Seasonally varying covariate (matched per ECOSTRESS scene)

**3. NDVI from Harmonized Landsat-Sentinel-2 (HLS)**

**Source:** NASA HLS-VI (Vegetation Index) products — NDVI at 30m resolution, derived from harmonized Landsat 8/9 + Sentinel-2A/B/C surface reflectance. Global observations every 2–3 days. Available from 2013 to present via LP DAAC.

**Temporal validity:** NDVI changes on weekly to biweekly timescales during the growing season, and more slowly during dormancy. For the frost season (Oct–Apr), vegetation transitions through senescence (Oct–Nov) and green-up (Mar–Apr), with relatively stable dormant values in winter (Dec–Feb). A biweekly composite is appropriate — assigning each ECOSTRESS scene the nearest-in-time cloud-free NDVI composite introduces negligible error.

**Why it matters for frost:**
- Green vegetation transpires, adding moisture to the surface energy balance and modifying overnight cooling rates
- Dormant/senescent vegetation and bare soil have lower thermal inertia and cool faster
- The transition periods (fall senescence, spring green-up) are exactly when frost timing matters most — a field that's still green in early October cools differently from the same field in late November when it's brown
- NDVI also correlates with surface roughness and canopy structure, capturing information beyond the binary NLCD forest/non-forest classification

**Access:** Via AppEEARS or `earthaccess`:
- Products: `HLSL30VI.002` (Landsat) and `HLSS30VI.002` (Sentinel-2)
- Layer: `NDVI`
- These are analysis-ready — already atmospherically corrected, BRDF-normalized, and harmonized between sensors

**Processing steps:**
1. Download all HLS-VI NDVI scenes over the study area for frost season months (Sep–May), 2018–present
2. Filter by quality flags — remove cloud, cloud shadow, snow/ice pixels
3. Create biweekly (14-day) maximum-value composites:
   - For each 14-day window, take the maximum NDVI at each pixel across all clear observations
   - Maximum-value compositing is standard practice — it minimizes residual cloud/atmosphere contamination
   - This yields ~15 composites per frost season (mid-Sep through mid-Apr)
4. For each ECOSTRESS scene, assign the nearest-in-time biweekly NDVI composite:
   ```python
   # Match ECOSTRESS scene date to nearest NDVI composite center date
   composite_dates = [sept_15, sept_29, oct_13, oct_27, ...]  # biweekly centers
   nearest_ndvi = composites[argmin(|scene_date - composite_dates|)]
   ```
5. Aggregate NDVI from 30m to 100m using mean
6. For pixels with no valid NDVI (persistent cloud during a composite period), interpolate temporally from adjacent composites

**Output:**
```
data/processed/ndvi/
    hls_ndvi_composites.zarr    # Biweekly NDVI composites at 100m, indexed by composite center date
    ndvi_lookup.parquet         # Mapping: ECOSTRESS scene_id → nearest NDVI composite date
```

**Expected data volume:**
- ~15 composites per frost season × 6+ seasons × ~10 MB per composite = ~1 GB total
- Very manageable on a laptop

**Key libraries:** `earthaccess`, `rasterio`, `rioxarray`, `xarray`, `numpy`

---

## Phase 2: Model development — Regression kriging

### Task 2.1 — Prepare training dataset from ECOSTRESS scenes

The modeling unit is an individual ECOSTRESS pixel observation — not a station-year average. Each clear-sky nighttime/predawn ECOSTRESS scene provides tens of thousands of 100m training samples, each paired with terrain covariates and the coarse gridded Tmin from the same night.

**Step 1: Build scene inventory**
```python
# For each ECOSTRESS scene in the frost season archive:
scene_record = {
    'scene_id': granule_id,
    'utc_datetime': acquisition_time,            # From granule metadata
    'local_datetime': utc_to_eastern(acquisition_time),
    'local_hour': local_datetime.hour + local_datetime.minute / 60,
    'sunrise_time': compute_sunrise(date, lat=35.595, lon=-82.551),  # Using astropy or pvlib
    'hours_until_sunrise': (sunrise_time - local_datetime).total_seconds() / 3600,
    'date': acquisition_time.date(),
    'cloud_fraction': fraction of cloudy pixels in study area,
}
```

Filter to retain scenes where:
- `local_hour` is between 20:00 (8 PM) and sunrise (nighttime/predawn window)
- `cloud_fraction` < 0.3 (mostly clear over study area)
- Quality flags indicate reliable LST retrieval

**Step 2: For each retained scene, extract pixel-level training samples**
```python
# Pseudocode for one scene
# First, load the matching gridMET data for this scene's date
scene_date = scene.date
gridmet_day = load_gridmet(scene_date)  # tmmn, tmmx, vs, sph, vpd, srad, pr, rmin

# For daytime variables, use the preceding day (afternoon before the night)
gridmet_prev = load_gridmet(scene_date - 1_day)  # tmmx, srad from prior afternoon

for pixel in ecostress_scene.valid_pixels:
    row = {
        # Response variable
        'lst': pixel.lst_celsius,                # ECOSTRESS LST (°C) at 100m

        # Coarse gridded predictors — matched to same night, from 4km gridMET cells
        'gridded_tmin': sample_nearest_4km(gridmet_day['tmmn'], pixel.lon, pixel.lat),
        'gridded_tmax': sample_nearest_4km(gridmet_prev['tmmx'], pixel.lon, pixel.lat),
        'diurnal_range': gridded_tmax - gridded_tmin,   # Derived: clear-sky indicator
        'wind_speed': sample_nearest_4km(gridmet_day['vs'], pixel.lon, pixel.lat),
        'specific_humidity': sample_nearest_4km(gridmet_day['sph'], pixel.lon, pixel.lat),
        'vpd': sample_nearest_4km(gridmet_day['vpd'], pixel.lon, pixel.lat),
        'srad_prev': sample_nearest_4km(gridmet_prev['srad'], pixel.lon, pixel.lat),
        'precip': sample_nearest_4km(gridmet_day['pr'], pixel.lon, pixel.lat),
        'rh_min': sample_nearest_4km(gridmet_day['rmin'], pixel.lon, pixel.lat),

        # Terrain covariates at 100m (static — precomputed)
        'elevation': sample_raster('elevation.tif', pixel.lon, pixel.lat),
        'slope': sample_raster('slope.tif', pixel.lon, pixel.lat),
        'aspect_sin': sample_raster('aspect_sin.tif', pixel.lon, pixel.lat),
        'aspect_cos': sample_raster('aspect_cos.tif', pixel.lon, pixel.lat),
        'tpi_300m': sample_raster('tpi_300m.tif', pixel.lon, pixel.lat),
        'tpi_1000m': sample_raster('tpi_1000m.tif', pixel.lon, pixel.lat),
        'curvature': sample_raster('curvature.tif', pixel.lon, pixel.lat),

        # Land surface covariates at 100m (static — precomputed from NLCD and DEM)
        'impervious_pct': sample_raster('impervious_pct.tif', pixel.lon, pixel.lat),
        'tree_canopy_pct': sample_raster('tree_canopy_pct.tif', pixel.lon, pixel.lat),
        'sky_view_factor': sample_raster('sky_view_factor.tif', pixel.lon, pixel.lat),
        'dist_to_water_m': sample_raster('dist_to_water_m.tif', pixel.lon, pixel.lat),

        # Vegetation state (seasonally varying — matched to nearest biweekly composite)
        'ndvi': sample_raster(
            nearest_ndvi_composite(scene.date),   # HLS biweekly NDVI at 100m
            pixel.lon, pixel.lat
        ),

        # Temporal predictor
        'hours_until_sunrise': scene.hours_until_sunrise,

        # Metadata for grouping/filtering
        'scene_id': scene.scene_id,
        'date': scene.date,
        'pixel_x': pixel.col,
        'pixel_y': pixel.row,
    }
    training_data.append(row)
```

**Step 3: Assign each pixel to its nearest 4km gridMET cell**

Each 100m pixel inherits all gridMET variables from whichever 4km cell it falls within. This means ~1,600 ECOSTRESS pixels share the same meteorological context (a 4km cell contains 40×40 = 1,600 cells at 100m). The terrain covariates and `hours_until_sunrise` then explain the *within-cell variation* — this is exactly the downscaling signal we want to learn. The gridMET variables explain the *between-night variation* — why the same terrain produces different LST patterns on different nights.

**Data volume considerations:**
- Each clear-sky ECOSTRESS scene over Buncombe County yields ~30,000–50,000 valid 100m pixels
- With ~5–15 usable nighttime scenes per frost season, and multiple frost seasons (2018–present), expect ~500,000–2,000,000 total training samples
- This is a large dataset by regression standards but easily fits in memory as a Parquet file (~50–200 MB)
- The large sample size is an advantage — it provides robust coefficient estimates and fine-grained spatial learning

**Output:** `data/processed/training/ecostress_training_samples.parquet`

**Key libraries:** `rasterio`, `rioxarray`, `pandas`, `pvlib` or `astropy` (for sunrise calculation), `pyproj`

### Task 2.2 — Fit regression kriging model

**The model:**
```
LST_ecostress ~ gridded_tmin + gridded_tmax + diurnal_range + wind_speed
                + specific_humidity + vpd + srad_prev + precip + rh_min
                + elevation + slope + aspect_sin + aspect_cos
                + tpi_300m + tpi_1000m + curvature
                + impervious_pct + tree_canopy_pct + sky_view_factor
                + dist_to_water_m + ndvi
                + hours_until_sunrise
```

This model learns how 70m land surface temperature relates to the coarse 4km meteorological context, local terrain, land surface properties, and vegetation state. The gridMET variables explain between-night variability (why tonight is different from last night), the terrain and land surface covariates explain within-cell variability (why this hollow is colder than that ridgetop on the same night), NDVI captures seasonal surface changes, and `hours_until_sunrise` explains where the observation falls in the nocturnal cooling cycle.

**Expected coefficient signs and interpretation:**

*Meteorological predictors (vary nightly, constant within each 4km cell):*
| Predictor | Expected sign | Physical interpretation |
|-----------|--------------|----------------------|
| `gridded_tmin` | Positive (strong, near 1.0) | Baseline: colder nights → colder surfaces |
| `gridded_tmax` | Positive (weak) | Warmer daytime → more stored heat to radiate; or use `diurnal_range` instead |
| `diurnal_range` | Negative | Large range = clear/dry conditions = stronger radiative cooling = colder surfaces relative to Tmin |
| `wind_speed` | Positive | Wind mixes boundary layer, reduces cooling, warms valley bottoms relative to ridges |
| `specific_humidity` | Positive | Humid air traps longwave radiation, slows surface cooling |
| `vpd` | Negative | High VPD = dry air = faster radiative cooling (correlated with humidity — check VIF) |
| `srad_prev` | Negative (weak) | More solar heating → more thermal energy to lose overnight; or positive if warm surfaces stay warmer |
| `precip` | Positive (weak) | Wet surfaces cool differently; cloudy/precipitating nights are warmer |
| `rh_min` | Positive | Higher minimum RH → more atmospheric moisture → less radiative cooling |

*Terrain covariates (static, vary at 100m within each 4km cell):*
| Predictor | Expected sign | Physical interpretation |
|-----------|--------------|----------------------|
| `elevation` | Negative | Higher ground is colder (lapse rate) — but may be *positive* during inversions |
| `slope` | Positive (weak) | Steeper slopes shed cold air, stay slightly warmer |
| `aspect_sin` | Weak/variable | North vs south orientation effect on overnight cooling |
| `aspect_cos` | Weak/variable | East vs west orientation |
| `tpi_300m` | Positive | Valley bottoms (negative TPI) are colder than ridges — cold air pooling |
| `tpi_1000m` | Positive | Broad landscape position effect |
| `curvature` | Positive | Concave terrain pools cold air |

*Land surface covariates (static or slow-varying, at 100m):*
| Predictor | Expected sign | Physical interpretation |
|-----------|--------------|----------------------|
| `impervious_pct` | Positive | Urban heat island — impervious surfaces store and release heat |
| `tree_canopy_pct` | Positive | Canopy traps longwave radiation, reduces sky exposure, slows cooling |
| `sky_view_factor` | Negative | More sky exposure → faster radiative cooling → colder surface |
| `dist_to_water_m` | Negative (weak) | Closer to water → moderating thermal influence → warmer |
| `ndvi` | Positive (weak, seasonal) | Green vegetation modifies surface energy balance; dormant vegetation has lower thermal inertia |

*Temporal predictor:*
| Predictor | Expected sign | Physical interpretation |
|-----------|--------------|----------------------|
| `hours_until_sunrise` | Negative | More hours until sunrise → earlier in cooling cycle → warmer surface |

**Step 1: Exploratory analysis**
- Scatter plots of LST vs each predictor, colored by `hours_until_sunrise`
- Check for nonlinearity (especially in `elevation` — inversions create a nonlinear lapse rate)
- Correlation matrix — check for multicollinearity among predictors (VIF)
  - `specific_humidity`, `vpd`, `rh_min`, and `diurnal_range` will be correlated — may need to select a subset or use PCA
  - `gridded_tmin` and `gridded_tmax` are correlated — `diurnal_range` may be more informative than including both
- Distribution of `hours_until_sunrise` across scenes — is it well-sampled?
- Distribution of meteorological conditions across scenes — are calm/clear nights well-represented?

**Step 2: OLS regression**
- Fit using `statsmodels` OLS for diagnostic output (p-values, VIF, influence plots)
- Start with a core model and build up:
  - **Core:** `LST ~ gridded_tmin + elevation + tpi_300m + hours_until_sunrise`
  - **Add meteorology:** `+ wind_speed + diurnal_range + specific_humidity`
  - **Add remaining terrain:** `+ slope + aspect_sin + aspect_cos + tpi_1000m + curvature`
  - **Add remaining meteorology:** `+ srad_prev + precip + rh_min`
- Evaluate R², adjusted R², AIC/BIC at each step — do the additional variables improve the model?
- Drop variables with high VIF (> 10) or non-significant coefficients
- Consider interaction terms — these are where the meteorological and terrain variables combine most powerfully:
  - `wind_speed × tpi_300m` — cold air pooling effect suppressed on windy nights (key interaction)
  - `diurnal_range × tpi_300m` — large diurnal range amplifies topographic cold pooling
  - `wind_speed × elevation` — lapse rate is steeper on calm nights
  - `gridded_tmin × tpi_300m` — cold air pooling may intensify on the coldest nights
  - `hours_until_sunrise × tpi_300m` — valley bottoms may cool faster as the night progresses

**Step 3: Account for within-scene spatial autocorrelation**
- Residuals from pixels within the same scene will be spatially autocorrelated
- Two approaches to handle this:
  - (a) **Clustered standard errors** by scene — corrects inference (p-values, CIs) but doesn't change predictions
  - (b) **Random effects by scene** — use a mixed model: `LST ~ fixed_effects + (1 | scene_id)` via `statsmodels` MixedLM or `pymer4`
  - (c) **Kriging of residuals** — fit variogram to residuals within each scene, krige onto full grid
- Option (c) is the full regression kriging approach and gives the best spatial predictions
- However, because the training data is pixels (not sparse stations), the kriging step may be computationally expensive — consider subsampling residuals for variogram fitting

**Step 4: Variogram estimation on residuals**
- Pool residuals across scenes (after removing scene-level mean residual)
- Compute empirical variogram using `pykrige` or `gstools`
- Fit theoretical variogram model (exponential or Matérn recommended for temperature fields)
- The variogram range will indicate the spatial scale of unexplained variation — expect 1–5 km
- If the variogram range is short (< 1 km), kriging adds less value and the regression alone may suffice

**Step 5: Kriging of residuals (optional, based on Step 4)**
- If significant spatial autocorrelation remains in residuals:
  - Subsample residuals (e.g., every 5th pixel) to reduce kriging computation
  - Krige residuals onto the full 100m grid
  - Add kriged residuals to regression prediction

**Key libraries:** `statsmodels`, `scikit-learn`, `pykrige` or `gstools`, `scipy`, `pymer4` (optional for mixed models)

### Task 2.3 — Random forest benchmark

Run in parallel using the same training dataset:
```python
from sklearn.ensemble import RandomForestRegressor

features = [
    # Meteorological context (4km, varies nightly)
    'gridded_tmin', 'gridded_tmax', 'diurnal_range',
    'wind_speed', 'specific_humidity', 'vpd',
    'srad_prev', 'precip', 'rh_min',
    # Terrain (100m, static)
    'elevation', 'slope', 'aspect_sin', 'aspect_cos',
    'tpi_300m', 'tpi_1000m', 'curvature',
    # Land surface (100m, static or slow-varying)
    'impervious_pct', 'tree_canopy_pct', 'sky_view_factor',
    'dist_to_water_m', 'ndvi',
    # Temporal
    'hours_until_sunrise',
]

rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=10)
rf.fit(X_train[features], y_train['lst'])
```

- RF naturally handles nonlinear relationships (inversions, wind thresholds) and interaction effects (wind × terrain) without explicit specification
- Feature importance ranking — which variables matter most? Expect `gridded_tmin` and `elevation` to dominate, but `wind_speed` and `diurnal_range` may be surprisingly important
- Partial dependence plots for key variables:
  - `elevation` — does the model learn inversion behavior (warming with elevation at valley floor)?
  - `wind_speed` — is there a threshold below which cold pooling activates?
  - `hours_until_sunrise` — what cooling curve shape does the model learn?
  - `wind_speed` × `tpi_300m` — 2D partial dependence to visualize the calm-night cold pooling interaction
- RF handles correlated predictors (`vpd`, `specific_humidity`, `rh_min`) without VIF issues — include all and let importance ranking sort them out
- Optionally krige RF residuals too (RF + kriging hybrid)
- Compare RMSE, MAE, R² against regression kriging

### Task 2.4 — Cross-validation

**Scene-based holdout (primary CV strategy):**
- Hold out entire scenes (not individual pixels) to avoid spatial leakage
- Train on all pixels from N-1 scenes, predict on all pixels from the held-out scene
- This is the honest test: can the model predict LST spatial patterns on a night it hasn't seen?
- Report per-scene RMSE and overall pooled RMSE

**Temporal holdout:**
- Train on frost seasons 2018–2022, test on 2023–2024 (or similar split)
- Tests whether the learned downscaling relationships are stable across years

**Spatial block holdout (within-scene):**
- Divide the study area into spatial blocks (e.g., 2km × 2km)
- Within each scene, hold out blocks and predict
- Tests whether the model generalizes spatially within a scene

**Metrics:**
- RMSE (°C) — target < 2°C for LST prediction
- MAE (°C)
- R²
- Bias (systematic over/under prediction)
- Per-scene RMSE distribution (are some scenes much worse? Why? — cloud contamination, extreme inversions?)
- Residual maps: do errors cluster in specific terrain types (e.g., dense forest, water bodies)?

**Validation against station data:**
- For scenes acquired close to station observation times, compare:
  - ECOSTRESS LST at station pixel vs station-measured air temperature
  - Model-predicted LST at station pixel vs station-measured air temperature
- Quantify the LST-to-air-temperature offset and its variability

**Key libraries:** `scikit-learn`, custom spatial CV implementation

---

## Phase 3: Prediction and output

### Task 3.1 — Predict nightly Tmin at 100m resolution

At prediction time, the trained model downscales any night's coarse Tmin field to 100m:

```
predicted_LST(x,y) = β₀ + β₁·gridded_tmin(x,y) + β₂·elevation(x,y) + β₃·slope(x,y)
                     + β₄·aspect_sin(x,y) + β₅·aspect_cos(x,y) + β₆·tpi_300m(x,y)
                     + β₇·tpi_1000m(x,y) + β₈·curvature(x,y) + β₉·hours_until_sunrise
                     + kriged_residual(x,y)  [if kriging is used]
```

**For frost date mapping, set `hours_until_sunrise = 0`** (i.e., predict LST at sunrise, the approximate time of the daily minimum). This gives you the coldest expected surface temperature for that night at each 100m pixel.

**Prediction workflow for a single night:**
1. Load the PRISM (or nClimGrid/gridMET) daily Tmin grid for that date
2. For each 100m pixel, look up the Tmin from its enclosing 4km cell
3. Apply regression coefficients with terrain covariates (precomputed, static)
4. Set `hours_until_sunrise = 0` for frost-relevant prediction
5. Optionally add kriged residual surface
6. Threshold at 0°C to classify frost/no-frost

**Compute notes:**
- Terrain covariates are static — compute once and reuse for every night
- The only input that changes nightly is `gridded_tmin` (one 4km raster per night)
- Prediction for one night: ~1 second (matrix multiply on 66,000 pixels)
- Full frost season (210 nights × 30 years): ~1–2 hours total
- Kriging adds ~1–5 minutes per night if used

### Task 3.2 — Frost date extraction

For each 100m pixel, scan the predicted nightly Tmin series to extract:

```python
for pixel in grid:
    for year in years:
        nightly_tmin = predicted_tmin[pixel, frost_season_nights(year)]

        # First fall frost: first night in Sep–Dec where predicted LST ≤ 0°C
        fall_frost_nights = [d for d in sep_dec_nights if nightly_tmin[d] <= 0]
        first_fall_frost_doy[pixel, year] = fall_frost_nights[0] if fall_frost_nights else NaN

        # Last spring frost: last night in Jan–May where predicted LST ≤ 0°C
        spring_frost_nights = [d for d in jan_may_nights if nightly_tmin[d] <= 0]
        last_spring_frost_doy[pixel, year] = spring_frost_nights[-1] if spring_frost_nights else NaN
```

**Output maps (GeoTIFF at 100m):**
- `frost_first_fall_mean.tif` — mean first fall frost DOY (averaged across years)
- `frost_first_fall_std.tif` — standard deviation (interannual variability)
- `frost_last_spring_mean.tif` — mean last spring frost DOY
- `frost_last_spring_std.tif` — standard deviation
- `frost_free_season_mean.tif` — mean number of frost-free days

### Task 3.3 — Frost probability maps

For each calendar day in the frost season:
```
P(frost on day d at pixel x,y) = fraction of years where predicted_LST(x,y,d) ≤ 0°C
```

Alternatively, assuming normal distribution of frost dates per pixel:
```
P(frost on day d) = Φ((d - mean_frost_doy) / sd_frost_doy)
```

**Output:** `frost_probability_by_doy.nc` — a 3D NetCDF (x, y, day_of_year) with frost probabilities.

### Task 3.4 — Uncertainty quantification

Sources of uncertainty to propagate:
1. **Regression coefficient uncertainty** — from OLS standard errors, propagate to prediction intervals
2. **Kriging variance** — if kriging is used, provides spatially varying prediction uncertainty
3. **Input uncertainty** — PRISM Tmin has its own interpolation error, higher in data-sparse mountain areas
4. **LST-to-air-temperature offset** — the model predicts surface temperature, but frost depends on air temperature at plant height; this offset introduces ~1–3°C systematic uncertainty

Convert to:
- 90% prediction interval on frost dates at each pixel
- Map of prediction confidence (higher near terrain types well-represented in ECOSTRESS training data)

### Task 3.5 — Validation against station frost records

The ultimate validation: compare model-predicted frost dates against observed frost dates at GHCN-Daily and ECONet stations.

1. For each station, extract the model-predicted frost dates at that pixel
2. Compare to observed frost dates from station Tmin records
3. Report RMSE, MAE, bias in frost date prediction (in days)
4. Stratify by station elevation, terrain position, and urban/rural setting
5. Map residuals — are there systematic spatial patterns in frost date errors?

This closes the loop: ECOSTRESS teaches the model about fine-scale thermal patterns, and station records confirm whether those patterns translate correctly to frost timing.

---

## Phase 4: Seasonal forecast integration (future)

### Task 4.1 — CFSv2 temperature anomaly ingestion

- Access CFSv2 seasonal forecasts from NOAA (available on AWS Open Data)
- Extract predicted temperature anomalies for the Asheville region for the upcoming frost season
- Anomalies are at ~56km resolution — one or a few grid cells cover the study area

### Task 4.2 — Anomaly-adjusted frost dates

Simple approach:
```
adjusted_frost_doy(x,y) = climatological_frost_doy(x,y) + f(T_anomaly)
```

Where `f(T_anomaly)` converts a seasonal temperature anomaly (°C) into a frost date shift (days). This relationship can be estimated empirically from the historical record:
- In years with +1°C anomaly, how many days later does first frost arrive?
- Typical sensitivity: ~3–5 days per °C for first fall frost

### Task 4.3 — CPC outlook integration

CPC provides tercile probabilities (above/below/near normal temperature). These can be translated into probabilistic frost date shifts.

---

## Dependencies

### Python environment

```
python >= 3.11
numpy
pandas
geopandas
xarray
rioxarray
rasterio
pyproj
richdem
pykrige        # or gstools for variogram/kriging
scikit-learn
statsmodels
scipy
matplotlib
cartopy
folium         # interactive maps
requests
zarr
dask
shapely
earthaccess    # NASA Earthdata authentication and ECOSTRESS/MODIS data access
pvlib          # Solar position and sunrise/sunset calculations
pygridmet      # gridMET meteorological data access via OpenDAP
```

### External data access

- **USGS National Map API:** No key required for 3DEP DEM downloads
- **NOAA NCEI CDO API:** Free API token required — register at `https://www.ncdc.noaa.gov/cdo-web/token`
- **NC State CLOUDS API:** No key required for public station data
- **PRISM:** No key required for 4km daily data (FTP download)
- **NASA Earthdata:** Free account required for MODIS/ECOSTRESS — register at `https://urs.earthdata.nasa.gov/`

---

## Implementation order

Work through these tasks sequentially. Each task should be a working, testable unit before moving to the next.

1. **`config/settings.py`** — Define bounding box, CRS, resolution, file paths
2. **`src/data/download_dem.py`** — Get the DEM and verify coverage
3. **`src/preprocessing/terrain.py`** — Compute all terrain covariates including sky view factor, visually inspect maps
4. **`src/data/download_nlcd.py`** — Fetch NLCD land cover, impervious surface, tree canopy cover
5. **`src/preprocessing/land_surface.py`** — Aggregate NLCD to 100m, compute distance-to-water, binary land cover indicators
6. **`src/data/download_ecostress.py`** — Fetch ECOSTRESS scenes, build scene inventory with overpass times
7. **`src/data/download_gridmet.py`** — Fetch gridMET variables (tmmn, tmmx, vs, sph, vpd, srad, pr, rmin) for dates matching ECOSTRESS scenes
8. **`src/data/download_hls_ndvi.py`** — Fetch HLS-VI NDVI, build biweekly composites
9. **`src/data/download_stations.py`** — Get station data for validation
10. **`src/preprocessing/frost_dates.py`** — Compute observed frost dates from station Tmin records
11. **`src/preprocessing/prepare_training.py`** — Build pixel-level training dataset (ECOSTRESS LST + gridMET + terrain + land surface + NDVI + hours_until_sunrise)
12. **`src/model/regression_kriging.py`** — Fit OLS regression + variogram + kriging on ECOSTRESS training data
13. **`src/model/random_forest.py`** — Fit RF benchmark on same training data
14. **`src/model/cross_validation.py`** — Scene-based holdout CV for both models
15. **`src/model/predict.py`** — Apply fitted model to every frost season night (2018–present) at 100m
16. **`src/postprocessing/frost_maps.py`** — Extract frost dates and probability maps from nightly predictions
17. **`src/postprocessing/validate.py`** — Compare predicted frost dates against station observations
18. **`src/visualization/static_maps.py`** — Publication-quality figures
19. **`src/visualization/interactive.py`** — Web-based map explorer
20. **`app/app.py`** — Streamlit app wrapping the interactive map

---

## Key assumptions and decisions

- **Frost threshold:** LST ≤ 0°C (32°F) at the surface. This is a conservative threshold — surface frost can form even when air temperature is slightly above 0°C due to radiative cooling of surfaces below air temperature. Could add a "hard frost" threshold at ≤ -2°C later.
- **Frost season:** September 1 – May 31. First fall frost = first day in Sep–Dec with predicted LST ≤ 0°C. Last spring frost = last day in Jan–May with predicted LST ≤ 0°C.
- **Model training period:** 2018–present (limited by ECOSTRESS availability). Predictions can be generated for any date with available PRISM Tmin, but the model's learned relationships are based on the ECOSTRESS era.
- **Working resolution:** 100m. Covariates computed from 10m DEM, aggregated to 100m (mean for elevation, circular mean for aspect, min for TPI to capture valley bottoms). ECOSTRESS native 70m data aggregated to 100m for model fitting.
- **CRS:** EPSG:32617 (UTM Zone 17N) for all spatial operations. Input data in geographic coordinates (EPSG:4326) will be reprojected.
- **LST vs air temperature:** The model predicts land surface temperature (skin temperature), not 2m air temperature. On clear calm nights, skin temperature is typically 2–5°C colder than air temperature. This means the model's frost predictions are conservative (more frost than station-measured air temperature would indicate). Station Tmin records are used for *validation*, not training. The `gridded_tmin` predictor is air temperature from PRISM, but the response variable is ECOSTRESS surface temperature — the regression coefficients implicitly learn the LST-to-Tmin offset as part of the intercept and `gridded_tmin` coefficient.
- **Prediction at sunrise:** For frost date mapping, predictions are made with `hours_until_sunrise = 0`, representing the approximate time of the daily surface temperature minimum. This gives the coldest expected surface temperature for each night.
- **Coarse-to-fine assumption:** The model assumes that the spatial relationship between 4km gridded Tmin and 100m surface temperature (as modulated by terrain) is stable across nights and years. This is physically reasonable — the terrain doesn't change, and the dominant cooling mechanisms (radiative cooling, cold air drainage, lapse rate) operate consistently.

---

## Notes for Claude Code / VS Code

- Each `src/` module should be independently runnable with a `if __name__ == '__main__':` block for testing
- Use `pathlib.Path` throughout, referencing paths from `config/settings.py`
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Log progress with `logging` module (not print statements)
- Save intermediate results to `data/processed/` so steps can be re-run independently
- All raster I/O through `rasterio` (writing) and `rioxarray` (reading/analysis)
- Pin dependency versions in `pyproject.toml`