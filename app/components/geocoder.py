"""Address geocoding via Nominatim (free, no API key)."""

import sys
from pathlib import Path

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BBOX_WGS84


def geocode_address(address_string):
    """Geocode an address string to (lat, lon, display_name).

    Appends ", Asheville, NC" if the input lacks a state/city hint.
    Returns None if geocoding fails or the result is outside the study area.
    """
    geolocator = Nominatim(user_agent="asheville-frost-app", timeout=5)

    # If the address doesn't mention NC or Asheville, add context
    lower = address_string.lower()
    if "nc" not in lower and "north carolina" not in lower and "asheville" not in lower:
        address_string = f"{address_string}, Asheville, NC"

    try:
        location = geolocator.geocode(address_string)
    except (GeocoderTimedOut, GeocoderServiceError):
        return None

    if location is None:
        return None

    lat, lon = location.latitude, location.longitude

    # Validate within study area bounding box
    if not (BBOX_WGS84["south"] <= lat <= BBOX_WGS84["north"]
            and BBOX_WGS84["west"] <= lon <= BBOX_WGS84["east"]):
        return None

    return lat, lon, location.address
