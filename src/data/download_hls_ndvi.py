"""Download HLS Vegetation Index (NDVI) scenes via earthaccess.

Fetches HLSL30_VI (Landsat) and HLSS30_VI (Sentinel-2) NDVI layers from
NASA LP DAAC for frost season months (Sep–May) over the study area.

Scenes are saved as individual GeoTIFFs in data/raw/hls/, with quality
flag rasters alongside for cloud/shadow masking during compositing.

Prerequisites:
  - NASA Earthdata account (https://urs.earthdata.nasa.gov/)
  - Credentials stored via `earthaccess.login(persist=True)` or in
    ~/.netrc as: machine urs.earthdata.nasa.gov login <user> password <pw>
"""

import calendar
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84, RAW_HLS_DIR
from config.data_sources import HLS_VI_PRODUCTS, FROST_SEASON_MONTHS, FROST_SEASON_START_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _frost_season_date_ranges(start_year: int, end_year: int) -> list[tuple[str, str]]:
    """Generate (start_date, end_date) pairs for each frost season month.

    Returns date ranges for Sep–Dec of each year from start_year, and
    Jan–May of each year from start_year+1 through end_year.
    """
    ranges = []
    for year in range(start_year, end_year + 1):
        for month in FROST_SEASON_MONTHS:
            if month >= 9:
                yr = year
            else:
                yr = year + 1
                if yr > end_year:
                    continue

            last_day = calendar.monthrange(yr, month)[1]
            start = f"{yr}-{month:02d}-01"
            end = f"{yr}-{month:02d}-{last_day:02d}"
            ranges.append((start, end))
    return ranges


def _search_and_download(short_name: str, version: str,
                          date_range: tuple[str, str]) -> list[Path]:
    """Search for and download HLS-VI granules for one product and date range."""
    import earthaccess

    bbox = (BBOX_WGS84["west"], BBOX_WGS84["south"],
            BBOX_WGS84["east"], BBOX_WGS84["north"])

    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        bounding_box=bbox,
        temporal=date_range,
        count=500,
    )

    if not results:
        log.info("  No granules for %s %s–%s", short_name, *date_range)
        return []

    log.info("  Found %d granules for %s %s–%s", len(results), short_name, *date_range)

    # Filter data links to only NDVI and Fmask bands (skip EVI, SAVI, etc.)
    # This avoids downloading ~9x more data than needed.
    keep_bands = {"NDVI", "Fmask"}
    filtered_links = []
    for granule in results:
        links = granule.data_links()
        for link in links:
            if any(band in link for band in keep_bands):
                filtered_links.append(link)

    if not filtered_links:
        log.info("  No NDVI/Fmask links found in granules")
        return []

    log.info("  Downloading %d files (NDVI + Fmask only)", len(filtered_links))
    downloaded = earthaccess.download(filtered_links, str(RAW_HLS_DIR))
    return [Path(p) for p in downloaded]


def run(end_year: int | None = None) -> list[Path]:
    """Download all HLS-VI NDVI scenes for frost season months.

    Parameters
    ----------
    end_year : last year to download (default: current year)
    """
    import earthaccess

    if end_year is None:
        end_year = datetime.now().year

    RAW_HLS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Authenticating with NASA Earthdata …")
    earthaccess.login(persist=True)

    date_ranges = _frost_season_date_ranges(FROST_SEASON_START_YEAR, end_year)
    log.info("Will search %d month windows from %d to %d",
             len(date_ranges), FROST_SEASON_START_YEAR, end_year)

    all_paths: list[Path] = []

    for product_name, product_info in HLS_VI_PRODUCTS.items():
        short_name = product_info["short_name"]
        version = product_info["version"]
        log.info("── Downloading %s (%s v%s) ──", product_name, short_name, version)
        for dr in date_ranges:
            try:
                paths = _search_and_download(short_name, version, dr)
                all_paths.extend(paths)
            except Exception as e:
                log.warning("  Failed for %s %s–%s: %s", short_name, dr[0], dr[1], e)
                continue

    log.info("Downloaded %d total HLS-VI files → %s", len(all_paths), RAW_HLS_DIR)
    return all_paths


if __name__ == "__main__":
    run()
