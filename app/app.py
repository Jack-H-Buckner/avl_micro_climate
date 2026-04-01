"""Asheville Frost Climatology Explorer — interactive Streamlit app.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.data_loader import (
    load_last_frost_data,
    last_frost_date_for_threshold,
    reproject_to_wgs84,
    reproject_elev_mask_to_wgs84,
    fractional_week_to_date_label,
)
from components.frost_analysis import extract_timeseries
from components.map_builder import build_frost_map
from components.geocoder import geocode_address
from components.plotting import plot_frost_timeseries

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Asheville Frost Explorer",
    page_icon="",
    layout="wide",
)

# ── Cached data loading ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading last-frost climatology...")
def _load_data():
    data, valid_mask, profile, mode, elev_valid_mask = load_last_frost_data()
    return data, valid_mask, profile, mode, elev_valid_mask


data, valid_mask, profile, mode, elev_valid_mask = _load_data()

# ── Session state defaults ───────────────────────────────────────────────────

if "selected_location" not in st.session_state:
    st.session_state.selected_location = None
if "location_label" not in st.session_state:
    st.session_state.location_label = None

# ── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Frost Climatology Explorer")
    st.markdown(
        "Explore the estimated **last frost date** at different "
        "confidence levels across the Asheville, NC region at 100m resolution."
    )

    st.divider()

    threshold_pct = st.slider(
        "Confidence level (%)",
        min_value=5,
        max_value=100,
        value=50,
        step=5,
        format="%d%%",
        help="The map shows the date by which there is this probability "
             "that the last frost has already occurred.",
    )
    threshold = threshold_pct / 100.0

    st.divider()
    st.subheader("Look up an address")
    address_input = st.text_input(
        "Enter an address",
        placeholder="e.g. 61 Pack Square, Asheville, NC",
    )
    geocode_btn = st.button("Look up", type="primary", use_container_width=True)

    if geocode_btn and address_input.strip():
        result = geocode_address(address_input.strip())
        if result is not None:
            lat, lon, display_name = result
            st.session_state.selected_location = (lat, lon)
            st.session_state.location_label = display_name
            st.success(f"Found: {display_name}")
        else:
            st.error(
                "Address not found within the study area. "
                "Try including 'Asheville, NC' or check the address."
            )

    st.divider()
    st.markdown("**Or click the map** to select a location.")

    if st.session_state.selected_location is not None:
        lat, lon = st.session_state.selected_location
        st.info(f"Selected: {lat:.4f}N, {lon:.4f}W")
        if st.button("Clear selection"):
            st.session_state.selected_location = None
            st.session_state.location_label = None
            st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────

last_frost = last_frost_date_for_threshold(data, mode, threshold)
last_frost_wgs84, bounds = reproject_to_wgs84(last_frost, profile)

# Reproject elevation validity mask for the gray overlay
elev_invalid_wgs84 = None
if elev_valid_mask is not None:
    elev_valid_wgs84, _ = reproject_elev_mask_to_wgs84(elev_valid_mask, profile)
    elev_invalid_wgs84 = ~elev_valid_wgs84

m = build_frost_map(
    last_frost_wgs84,
    bounds,
    threshold,
    marker_location=st.session_state.selected_location,
    marker_label=st.session_state.location_label,
    elev_invalid_mask=elev_invalid_wgs84,
)

map_data = st_folium(
    m,
    width=None,
    height=600,
    returned_objects=["last_clicked"],
)

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    click_lat = clicked["lat"]
    click_lng = clicked["lng"]
    if st.session_state.selected_location != (click_lat, click_lng):
        st.session_state.selected_location = (click_lat, click_lng)
        st.session_state.location_label = None
        st.rerun()

# ── Time series chart ────────────────────────────────────────────────────────

if st.session_state.selected_location is not None:
    lat, lon = st.session_state.selected_location
    timeseries = extract_timeseries(data, mode, lat, lon, profile, valid_mask)

    if timeseries is not None:
        label = st.session_state.location_label or f"{lat:.4f}N, {lon:.4f}W"

        last_frost_wk = None
        for wn, dlabel, cp in timeseries:
            if cp is not None and cp >= threshold:
                last_frost_wk = wn
                break

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = plot_frost_timeseries(timeseries, threshold, location_label=label)
            st.pyplot(fig)
        with col2:
            st.markdown("### Summary")
            if last_frost_wk is not None:
                date_str = fractional_week_to_date_label(last_frost_wk)
                st.metric(
                    label=f"Last frost date ({threshold_pct}% confidence)",
                    value=f"~{date_str}",
                )
            else:
                st.warning(
                    f"Last frost date not reached at {threshold_pct}% confidence."
                )
    else:
        st.warning("Selected location is outside the study area grid.")
