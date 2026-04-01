"""Pixel-level time series extraction for the interactive app."""

import numpy as np
from pyproj import Transformer

from .data_loader import extract_frost_timeseries_at_pixel


def latlon_to_pixel(lat, lon, src_profile):
    """Convert WGS84 lat/lon to (row, col) on the UTM grid."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    x, y = transformer.transform(lon, lat)
    transform = src_profile["transform"]
    col, row = ~transform * (x, y)
    return int(round(row)), int(round(col))


def extract_timeseries(data, mode, lat, lon, src_profile, valid_mask):
    """Extract frost time series at a lat/lon location.

    Snaps to nearest valid pixel if needed (within 5px).
    Returns list of (week_num, date_label, cumulative_prob) or None.
    """
    row, col = latlon_to_pixel(lat, lon, src_profile)

    grid_h, grid_w = valid_mask.shape
    if not (0 <= row < grid_h and 0 <= col < grid_w):
        return None

    # Snap to nearest valid pixel with frost signal if needed
    if not valid_mask[row, col]:
        search_radius = 5
        best_dist = float("inf")
        best_r, best_c = row, col
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    if valid_mask[nr, nc]:
                        dist = dr * dr + dc * dc
                        if dist < best_dist:
                            best_dist = dist
                            best_r, best_c = nr, nc
        if best_dist == float("inf"):
            return None
        row, col = best_r, best_c

    return extract_frost_timeseries_at_pixel(data, mode, row, col)
