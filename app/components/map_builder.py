"""Build interactive folium map with frost-free week raster overlay."""

import io

import branca.colormap as cm
import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data_loader import fractional_week_to_date_label


# Week range for the spring colorbar (roughly Jan through May)
VMIN, VMAX = 18, 39


CMAP_NAME = "YlOrRd"  # light yellow → orange → red; stays readable over maps


def _array_to_rgba(array, vmin=VMIN, vmax=VMAX, cmap_name=CMAP_NAME, alpha=180):
    """Convert a 2D float array to an RGBA uint8 image.

    NaN pixels are fully transparent. Valid pixels get a semi-transparent
    alpha so underlying map features (roads, terrain) show through.

    Parameters
    ----------
    alpha : int 0-255, transparency for valid pixels (180 ≈ 70% opaque)
    """
    cmap = plt.cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    normed = norm(array)
    rgba = cmap(normed)  # (H, W, 4) float in [0, 1]
    rgba = (rgba * 255).astype(np.uint8)

    # Set semi-transparent alpha on valid pixels
    valid_mask = np.isfinite(array)
    rgba[valid_mask, 3] = alpha

    # Set NaN pixels to fully transparent
    rgba[~valid_mask] = [0, 0, 0, 0]

    return rgba


def _rgba_to_png_bytes(rgba):
    """Encode an RGBA uint8 array to PNG bytes for folium ImageOverlay."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_frost_map(
    frost_free_array,
    bounds,
    threshold,
    center=(35.5951, -82.5515),
    zoom_start=10,
    marker_location=None,
    marker_label=None,
    elev_invalid_mask=None,
):
    """Build a folium Map with the frost-free week overlay.

    Parameters
    ----------
    frost_free_array : 2D array (WGS84), week numbers where frost drops below threshold
    bounds : [[south, west], [north, east]]
    threshold : float, the probability threshold (for legend title)
    center : (lat, lon) map center
    zoom_start : initial zoom level
    marker_location : optional (lat, lon) to place a marker
    marker_label : optional label for the marker popup
    elev_invalid_mask : optional (H, W) bool, True where elevation is OUT of range

    Returns
    -------
    folium.Map
    """
    # Compute dynamic color range from actual data (fractional weeks)
    # Clamp to valid frost-season range first (bilinear interp can create
    # outliers from blending with sentinel values at edges)
    frost_free_clamped = np.where(
        np.isfinite(frost_free_array),
        np.clip(frost_free_array, 1.0, 39.0),
        np.nan,
    )
    valid_vals = frost_free_clamped[np.isfinite(frost_free_clamped)]
    if len(valid_vals) > 0:
        vmin = float(np.percentile(valid_vals, 2))
        vmax = float(np.percentile(valid_vals, 98))
        if vmax - vmin < 1.0:
            vmin = max(1.0, vmin - 0.5)
            vmax = min(39.0, vmax + 0.5)
    else:
        vmin, vmax = float(VMIN), float(VMAX)
    frost_free_array = frost_free_clamped

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=None,
    )

    # Base tile layers — Positron (no labels) as default
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Light (default)",
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    # Frost-free week overlay
    rgba = _array_to_rgba(frost_free_array, vmin=vmin, vmax=vmax)
    png_buf = _rgba_to_png_bytes(rgba)

    import base64
    png_b64 = base64.b64encode(png_buf.read()).decode("utf-8")
    png_data_url = f"data:image/png;base64,{png_b64}"

    folium.raster_layers.ImageOverlay(
        image=png_data_url,
        bounds=bounds,
        opacity=1.0,
        name=f"Last frost date ({threshold:.0%} prob)",
        interactive=False,
        zindex=400,
    ).add_to(m)

    # Gray overlay for out-of-sample elevation pixels
    if elev_invalid_mask is not None and np.any(elev_invalid_mask):
        gray_rgba = np.zeros((*elev_invalid_mask.shape, 4), dtype=np.uint8)
        gray_rgba[elev_invalid_mask] = [160, 160, 160, 180]  # semi-transparent gray
        gray_png = _rgba_to_png_bytes(gray_rgba)

        import base64 as _b64
        gray_b64 = _b64.b64encode(gray_png.read()).decode("utf-8")
        gray_url = f"data:image/png;base64,{gray_b64}"

        folium.raster_layers.ImageOverlay(
            image=gray_url,
            bounds=bounds,
            opacity=1.0,
            name="Out-of-sample elevation",
            interactive=False,
            zindex=450,
        ).add_to(m)

    # Roads & labels overlays — injected via JS into a high-z custom pane
    # so they render on top of the ImageOverlay
    from branca.element import MacroElement, Template as BrancaTemplate
    roads_labels_js = MacroElement()
    roads_labels_js._template = BrancaTemplate("""
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};

            // Create custom panes with high z-index
            var roadsPane = map.createPane('roadsPane');
            roadsPane.style.zIndex = 650;
            roadsPane.style.pointerEvents = 'none';

            var labelsPane = map.createPane('labelsPane');
            labelsPane.style.zIndex = 660;
            labelsPane.style.pointerEvents = 'none';

            // Roads-only overlay (CartoDB dark lines — transparent background)
            var roads = L.tileLayer(
                'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png',
                {attribution: 'CartoDB', pane: 'roadsPane', opacity: 0.4}
            );
            roads.addTo(map);

            // Labels-only overlay (transparent background)
            var labels = L.tileLayer(
                'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png',
                {attribution: 'CartoDB', pane: 'labelsPane', opacity: 1.0}
            );
            labels.addTo(map);
        })();
        {% endmacro %}
    """)
    m.add_child(roads_labels_js)

    # Colorbar legend as a Leaflet control (renders inside the map div)
    legend_inner_html = _build_legend_inner_html(vmin, vmax, threshold)
    from branca.element import MacroElement, Template
    legend_control = MacroElement()
    legend_control._template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var legend = L.control({position: 'bottomleft'});
            legend.onAdd = function(map) {
                var div = L.DomUtil.create('div', 'frost-legend');
                div.innerHTML = `""" + legend_inner_html.replace("\n", "").replace("`", "\\`") + """`;
                return div;
            };
            legend.addTo({{ this._parent.get_name() }});
        })();
        {% endmacro %}
    """)
    m.add_child(legend_control)

    # Optional marker for selected location
    if marker_location is not None:
        popup_text = marker_label or f"{marker_location[0]:.4f}, {marker_location[1]:.4f}"
        folium.Marker(
            location=marker_location,
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="red", icon="home"),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def _build_legend_inner_html(vmin, vmax, threshold):
    """Build inner HTML for a Leaflet control color bar legend.

    vmin/vmax are fractional week numbers.
    """
    # Use 20 color stops for a smooth gradient
    n_colors = 20
    colors = []
    for i in range(n_colors):
        frac = i / max(n_colors - 1, 1)
        r, g, b = [int(c * 255) for c in plt.cm.get_cmap(CMAP_NAME)(frac)[:3]]
        colors.append(f"rgb({r},{g},{b})")

    gradient_stops = ", ".join(
        f"{c} {i * 100 / (n_colors - 1):.1f}%" for i, c in enumerate(colors)
    )

    # Date tick labels: ~6-8 evenly spaced
    span = vmax - vmin
    n_ticks = 7
    tick_html = ""
    for i in range(n_ticks):
        fw = vmin + i * span / (n_ticks - 1)
        label = fractional_week_to_date_label(fw)
        pct = i * 100 / (n_ticks - 1)
        tick_html += (
            f'<div style="position:absolute; left:{pct:.1f}%; '
            f'transform:translateX(-50%); text-align:center; '
            f'font-size:11px; font-weight:600; color:#222;">'
            f'{label}</div>'
        )

    thresh_pct = int(round(threshold * 100))
    gray_note = (
        '<div style="display:flex; align-items:center; margin-top:8px; '
        'font-size:11px; color:#555;">'
        '<div style="width:18px; height:12px; background:rgb(160,160,160); '
        'border-radius:2px; margin-right:6px; flex-shrink:0;"></div>'
        'Elevation outside training range (out of sample)</div>'
    )
    return (
        f'<div style="background:rgba(255,255,255,0.95); border-radius:8px; '
        f'padding:10px 16px 36px 16px; box-shadow:0 2px 8px rgba(0,0,0,0.3); '
        f'min-width:350px; max-width:500px;">'
        f'<div style="font-size:12px; font-weight:700; color:#222; '
        f'margin-bottom:6px; text-align:center;">'
        f'Estimated last frost date ({thresh_pct}% confidence)</div>'
        f'<div style="height:16px; border-radius:4px; '
        f'background:linear-gradient(to right, {gradient_stops});"></div>'
        f'<div style="position:relative; height:22px; margin-top:4px;">'
        f'{tick_html}</div>{gray_note}</div>'
    )
