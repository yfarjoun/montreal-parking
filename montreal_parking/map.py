"""Build the interactive map with external GeoJSON layers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from montreal_parking.constants import (
    COLOR_FREE,
    COLOR_NO_DATA,
    COLOR_RESTRICTED,
    COLOR_TIME_LIMITED,
    CRS_WGS84,
    MAP_CENTER,
    MAP_ZOOM_BOROUGH,
    MAP_ZOOM_DEFAULT,
    OUTPUT_DIR,
    SIMPLIFY_TOLERANCE,
    TILES_ATTR,
    TILES_URL,
)

# Layer configuration: (display name, category key, color, default-on)
_LAYER_CONFIG: list[tuple[str, str, str, bool]] = [
    ("Free Parking", "free", COLOR_FREE, True),
    ("Time-Limited", "time_limited", COLOR_TIME_LIMITED, True),
    ("Restricted", "restricted", COLOR_RESTRICTED, False),
    ("No Data", "no_data", COLOR_NO_DATA, False),
]


def _export_category_geojson(
    intervals_wgs: gpd.GeoDataFrame,
    category: str,
    dest: Path,
) -> bool:
    """Export one category to a simplified GeoJSON file. Returns True if non-empty."""
    subset = intervals_wgs[intervals_wgs["category"] == category].copy()
    if subset.empty:
        return False

    subset["popup_html"] = subset.apply(
        lambda row: (
            f"<b>{row['street_name']}</b><br>"
            f"Category: {category}<br>"
            f"Length: {row['length_m']:.0f}m<br>"
            f"<small>{html.escape(str(row['descriptions'])[:200])}</small>"
        ),
        axis=1,
    )
    subset = subset[["geometry", "popup_html"]].copy()
    subset["geometry"] = subset["geometry"].simplify(SIMPLIFY_TOLERANCE)
    subset.to_file(dest, driver="GeoJSON")
    print(f"    {category}: {len(subset)} features -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def _build_pole_geojson(
    signs_df: pd.DataFrame,
    dest: Path,
    label_prefix: str = "Pole",
) -> bool:
    """Export pole markers as GeoJSON with popup HTML. Returns True if non-empty."""
    if signs_df.empty:
        return False

    sign_html: dict[Any, str] = (
        signs_df.groupby("POTEAU_ID_POT")
        .apply(  # type: ignore[call-overload]
            lambda g: "<br>".join(
                f"[{row['FLECHE_PAN']}] {row['sign_category']}: "
                + html.escape(str(row["DESCRIPTION_RPA"]).replace("\\", "\\\\"))
                for _, row in g.iterrows()
            ),
            include_groups=False,
        )
        .to_dict()
    )

    unique = signs_df.drop_duplicates(subset="POTEAU_ID_POT")
    features = []
    for _, row in unique.iterrows():
        pid = row["POTEAU_ID_POT"]
        lat, lon = float(row["Latitude"]), float(row["Longitude"])
        sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        popup = (
            f"<b>{label_prefix} {pid}</b><br>"
            f"{sign_html.get(pid, '')}<br>"
            f'<a href="{sv_url}" target="_blank">Street View</a>'
        )
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"popup_html": popup},
        })

    with open(dest, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"    poles ({label_prefix}): {len(features)} features -> {dest}")
    return True


def _build_html_shell(
    layers: list[dict[str, Any]],
    center: list[float],
    zoom: int,
) -> str:
    """Generate a lightweight HTML page that loads GeoJSON via fetch()."""
    # Build JS layer definitions
    layer_js_parts = []
    overlay_js_parts = []
    for layer in layers:
        var = layer["var"]
        file = layer["file"]
        name = layer["name"]
        color = layer["color"]
        default_on = layer["default_on"]
        is_point = layer.get("is_point", False)

        if is_point:
            layer_js_parts.append(f"""
    // {name}
    var {var} = L.layerGroup();
    fetch('data/{file}')
      .then(r => r.json())
      .then(data => {{
        L.geoJSON(data, {{
          pointToLayer: function(f, ll) {{
            return L.circleMarker(ll, {{radius: 3, color: '{color}', fillColor: '{color}', fillOpacity: 0.7}});
          }},
          onEachFeature: function(f, layer) {{
            if (f.properties && f.properties.popup_html) layer.bindPopup(f.properties.popup_html, {{maxWidth: 400}});
          }}
        }}).addTo({var});
      }});""")
        else:
            layer_js_parts.append(f"""
    // {name}
    var {var} = L.layerGroup();
    fetch('data/{file}')
      .then(r => r.json())
      .then(data => {{
        L.geoJSON(data, {{
          style: {{color: '{color}', weight: 5, opacity: 0.8}},
          onEachFeature: function(f, layer) {{
            if (f.properties && f.properties.popup_html) layer.bindPopup(f.properties.popup_html, {{maxWidth: 400}});
          }}
        }}).addTo({var});
      }});""")

        if default_on:
            overlay_js_parts.append(f'    "{name}": {var},')
        else:
            overlay_js_parts.append(f'    "{name}": {var},')

    layers_init = "\n".join(layer_js_parts)
    # Default-on layers get added to the map immediately
    default_adds = "\n".join(
        f"    {layer['var']}.addTo(map);"
        for layer in layers
        if layer["default_on"]
    )
    overlays_obj = "\n".join(overlay_js_parts)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Montreal Free Parking Finder</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.locatecontrol@0.82.0/dist/L.Control.Locate.min.css"/>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.locatecontrol@0.82.0/dist/L.Control.Locate.min.js"></script>
  <script>
    var map = L.map('map').setView({center}, {zoom});
    L.tileLayer('{TILES_URL}', {{
      attribution: '{TILES_ATTR}',
      maxZoom: 19,
      subdomains: 'abcd'
    }}).addTo(map);

    // GPS locate button
    L.control.locate({{
      position: 'topleft',
      setView: 'once',
      flyTo: true,
      keepCurrentZoomLevel: true,
      strings: {{ title: "Show my location" }}
    }}).addTo(map);

{layers_init}

{default_adds}

    L.control.layers(null, {{
{overlays_obj}
    }}, {{collapsed: false}}).addTo(map);
  </script>
</body>
</html>"""


def build_map(
    intervals_gdf: gpd.GeoDataFrame,
    signs_df: pd.DataFrame,
    unsnapped_signs: pd.DataFrame | None = None,
    *,
    borough: str | None = None,
) -> None:
    """Export GeoJSON data files and a lightweight HTML map shell.

    Outputs to OUTPUT_DIR/index.html and OUTPUT_DIR/data/*.geojson.
    """
    intervals_wgs = intervals_gdf.to_crs(CRS_WGS84)

    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine map center and zoom
    if borough and not intervals_wgs.empty:
        # Use non-no_data intervals for centering (no_data gap-fills can span far)
        signaled = intervals_wgs[intervals_wgs["category"] != "no_data"]
        if signaled.empty:
            signaled = intervals_wgs
        bounds = signaled.total_bounds  # [minx, miny, maxx, maxy]
        center = [float((bounds[1] + bounds[3]) / 2), float((bounds[0] + bounds[2]) / 2)]
        zoom = MAP_ZOOM_BOROUGH
    else:
        center = MAP_CENTER
        zoom = MAP_ZOOM_DEFAULT

    # Export category GeoJSON files
    layers: list[dict[str, Any]] = []
    for display_name, category, color, default_on in _LAYER_CONFIG:
        dest = data_dir / f"{category}.geojson"
        exported = _export_category_geojson(intervals_wgs, category, dest)
        if exported:
            layers.append({
                "var": f"layer_{category}",
                "file": f"{category}.geojson",
                "name": display_name,
                "color": color,
                "default_on": default_on,
            })

    # Export poles GeoJSON (off by default)
    poles_dest = data_dir / "poles.geojson"
    if _build_pole_geojson(signs_df, poles_dest, label_prefix="Pole"):
        layers.append({
            "var": "layer_poles",
            "file": "poles.geojson",
            "name": "Sign Poles",
            "color": "#333",
            "default_on": False,
            "is_point": True,
        })

    # Export unsnapped poles (off by default)
    if unsnapped_signs is not None and not unsnapped_signs.empty:
        unsnapped_dest = data_dir / "unmatched_poles.geojson"
        if _build_pole_geojson(unsnapped_signs, unsnapped_dest, label_prefix="Unmatched"):
            layers.append({
                "var": "layer_unmatched",
                "file": "unmatched_poles.geojson",
                "name": "Unmatched Poles",
                "color": "#e67e22",
                "default_on": False,
                "is_point": True,
            })

    # Write HTML shell
    html_content = _build_html_shell(layers, center, zoom)
    index_path = OUTPUT_DIR / "index.html"
    with open(index_path, "w") as f:
        f.write(html_content)
    print(f"    HTML shell: {index_path} ({len(html_content) / 1e3:.0f} KB)")
