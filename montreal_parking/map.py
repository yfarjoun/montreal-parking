"""Build the interactive map with external GeoJSON layers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from montreal_parking import __version__
from montreal_parking.constants import (
    COLOR_FREE,
    COLOR_NO_DATA,
    COLOR_PAID,
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
    IntervalCategory,
)

FREE = IntervalCategory.FREE
TIME_LIMITED = IntervalCategory.TIME_LIMITED
PAID = IntervalCategory.PAID
RESTRICTED = IntervalCategory.RESTRICTED
NO_DATA = IntervalCategory.NO_DATA

# Layer configuration: (display name, category key, color, default-on)
_LAYER_CONFIG: list[tuple[str, IntervalCategory, str, bool]] = [
    ("Free Parking", FREE, COLOR_FREE, True),
    ("Time-Limited", TIME_LIMITED, COLOR_TIME_LIMITED, True),
    ("Paid Parking", PAID, COLOR_PAID, True),
    ("Restricted", RESTRICTED, COLOR_RESTRICTED, False),
    ("No Data", NO_DATA, COLOR_NO_DATA, False),
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

    def _popup(row: Any) -> str:
        lines = [
            f"<b>{row['street_name']}</b><br>",
            f"Category: {category}<br>",
        ]
        if "rate" in row.index and pd.notna(row.get("rate")):
            lines.append(f"Rate: ${row['rate']:.2f}/hr<br>")
        lines.append(f"Length: {row['length_m']:.0f}m<br>")
        lines.append(f"<small>{html.escape(str(row['descriptions'])[:200])}</small>")
        return "".join(lines)

    subset["popup_html"] = subset.apply(_popup, axis=1)
    subset = subset[["geometry", "popup_html"]].copy()
    subset["geometry"] = subset["geometry"].simplify(SIMPLIFY_TOLERANCE)
    subset.to_file(dest, driver="GeoJSON")
    print(f"    {category}: {len(subset)} features -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def _build_pole_geojson(
    signs_df: pd.DataFrame,
    dest: Path,
    label_prefix: str = "Pole",
    *,
    group_by_side: bool = False,
) -> bool:
    """Export pole markers as GeoJSON with popup HTML. Returns True if non-empty."""
    if signs_df.empty:
        return False

    arrow_display = {0: "", 2: "\u2190 ", 3: "\u2192 "}

    has_side = group_by_side and "side" in signs_df.columns
    group_cols: list[str] = ["POTEAU_ID_POT", "side"] if has_side else ["POTEAU_ID_POT"]

    sign_html: dict[Any, str] = (
        signs_df.groupby(group_cols)
        .apply(  # type: ignore[call-overload]
            lambda g: "<br>".join(
                f"{arrow_display.get(row['FLECHE_PAN'], '')} {row['sign_category']}: "
                + html.escape(str(row["DESCRIPTION_RPA"]))
                for _, row in g.iterrows()
            ),
            include_groups=False,
        )
        .to_dict()
    )

    unique = signs_df.drop_duplicates(subset=group_cols)
    features = []
    for _, row in unique.iterrows():
        pid = row["POTEAU_ID_POT"]
        key = (pid, row["side"]) if has_side else pid
        lat, lon = float(row["Latitude"]), float(row["Longitude"])
        sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        side_label = f" ({row['side']})" if has_side else ""
        popup = (
            f"<b>{label_prefix} {pid}{side_label}</b><br>"
            f"{sign_html.get(key, '')}<br>"
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


def _offset_deux_cotes_copies(
    copies: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute offset Lat/Lon for DEUX COTES copies on the opposite side of the road.

    Interpolates along the road at the copy's projection distance, then offsets
    6m perpendicular to the road in the direction of the copy's (flipped) side.
    """
    from montreal_parking.constants import CRS_MTM8, CRS_WGS84

    roads_mtm = roads_gdf.to_crs(CRS_MTM8)
    road_geoms = roads_mtm.set_index("ID_TRC")["geometry"]

    result = copies.copy()
    lats = np.empty(len(result))
    lons = np.empty(len(result))

    for i, (_, row) in enumerate(result.iterrows()):
        id_trc = row["ID_TRC"]
        if id_trc not in road_geoms.index:
            lats[i], lons[i] = row["Latitude"], row["Longitude"]
            continue
        road = road_geoms.loc[id_trc]
        proj = min(row["projection_distance"], road.length)
        road_pt = shapely.line_interpolate_point(road, proj)

        # Tangent vector
        p1 = shapely.line_interpolate_point(road, max(0, proj - 1))
        p2 = shapely.line_interpolate_point(road, min(road.length, proj + 1))
        tx = shapely.get_x(p2) - shapely.get_x(p1)
        ty = shapely.get_y(p2) - shapely.get_y(p1)
        length = np.sqrt(tx * tx + ty * ty)
        if length < 0.01:
            lats[i], lons[i] = row["Latitude"], row["Longitude"]
            continue

        # Perpendicular unit vector (left = +normal, right = -normal)
        nx, ny = -ty / length, tx / length
        offset = 3.0 if row["side"] == "left" else -3.0
        offset_x = shapely.get_x(road_pt) + nx * offset
        offset_y = shapely.get_y(road_pt) + ny * offset

        # Convert back to WGS84
        pt = gpd.GeoSeries(
            [shapely.Point(offset_x, offset_y)], crs=CRS_MTM8
        ).to_crs(CRS_WGS84).iloc[0]
        lons[i], lats[i] = pt.x, pt.y

    result["Latitude"] = lats
    result["Longitude"] = lons
    return result


def _build_html_shell(
    layers: list[dict[str, Any]],
    center: list[float],
    zoom: int,
    *,
    data_date: str = "",
) -> str:
    """Generate a lightweight HTML page that loads GeoJSON via fetch()."""
    # Build per-layer JS: fetch GeoJSON, store raw data, render on viewport change
    layer_fetch_parts = []
    overlay_js_parts = []
    layer_keys = []
    for layer in layers:
        var = layer["var"]
        file = layer["file"]
        name = layer["name"]
        color = layer["color"]
        is_point = layer.get("is_point", False)

        layer_keys.append(var)

        opts = (
            f"pointToLayer: function(f, ll) {{"
            f" return L.circleMarker(ll, {{radius:3,color:'{color}',"
            f"fillColor:'{color}',fillOpacity:0.7}}); }}"
            if is_point
            else f"style: {{color:'{color}',weight:5,opacity:0.8,lineCap:'butt'}}"
        )

        layer_fetch_parts.append(f"""
    // {name}
    layerData['{var}'] = null;
    layerGroups['{var}'] = L.layerGroup();
    fetch('data/{file}')
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        layerData['{var}'] = data;
        layerOpts['{var}'] = {{ {opts},
          onEachFeature: function(f, layer) {{
            if (f.properties && f.properties.popup_html)
              layer.bindPopup(f.properties.popup_html, {{maxWidth:400}});
          }}
        }};
        renderLayer('{var}');
      }});""")

        overlay_js_parts.append(f'    "{name}": layerGroups["{var}"],')

    layers_init = "\n".join(layer_fetch_parts)
    default_adds = "\n".join(
        f"    layerGroups['{layer['var']}'].addTo(map);"
        for layer in layers
        if layer["default_on"]
    )
    overlays_obj = "\n".join(overlay_js_parts)

    # Build JS object mapping layer display name → {color, isPoint}
    legend_entries = ", ".join(
        f'"{layer["name"]}": {{color:"{layer["color"]}", isPoint:{str(layer.get("is_point", False)).lower()}}}'
        for layer in layers
    )
    legend_map_js = f"var legendInfo = {{{legend_entries}}};"

    # Category colors for driving mode (line layers only)
    _cc_entries = ", ".join(
        f'"{layer["var"]}": "{layer["color"]}"'
        for layer in layers
        if not layer.get("is_point")
    )
    category_colors_js = f"var categoryColors = {{{_cc_entries}}};"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Montreal Free Parking Finder</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.locatecontrol@0.82.0/dist/L.Control.Locate.min.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.css"/>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    .legend-swatch {{
      display: inline-block;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .legend-swatch-line {{
      width: 18px;
      height: 4px;
      border-radius: 2px;
    }}
    .legend-swatch-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .info-control {{
      background: white;
      padding: 6px 10px;
      border-radius: 5px;
      box-shadow: 0 1px 5px rgba(0,0,0,0.3);
      font-size: 11px;
      line-height: 1.6;
      color: #555;
      max-width: 280px;
    }}
    .info-control a {{
      color: #3498db;
    }}
    .info-toggle {{
      display: none;
      background: white;
      border: none;
      border-radius: 50%;
      width: 28px;
      height: 28px;
      font-size: 16px;
      line-height: 28px;
      text-align: center;
      cursor: pointer;
      box-shadow: 0 1px 5px rgba(0,0,0,0.3);
      color: #555;
      padding: 0;
    }}
    @media (max-width: 767px) {{
      .info-toggle {{ display: block; }}
      .info-control {{ display: none; }}
      .info-expanded .info-control {{ display: block; }}
      .info-expanded .info-toggle {{ display: none; }}
    }}
    .driving-bar {{
      display: none;
      position: fixed;
      top: 0;
      width: 16px;
      height: 100%;
      z-index: 1000;
      opacity: 0.85;
      transition: background-color 0.5s;
      background-color: #888;
    }}
    .driving-bar-left {{ left: 0; border-right: 2px solid rgba(0,0,0,0.15); }}
    .driving-bar-right {{ right: 0; border-left: 2px solid rgba(0,0,0,0.15); }}
    .driving-active {{
      background-color: #3498db !important;
      color: white !important;
    }}
    .driving-arrow {{
      background: none !important;
      border: none !important;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.locatecontrol@0.82.0/dist/L.Control.Locate.min.js"></script>
  <script src="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.js"></script>
  <script>
    var map = L.map('map', {{preferCanvas: true}}).setView({center}, {zoom});
    L.tileLayer('{TILES_URL}', {{
      attribution: '{TILES_ATTR}',
      maxZoom: 19,
      subdomains: 'abcd'
    }}).addTo(map);

    // GPS locate button
    var locateControl = L.control.locate({{
      position: 'topleft',
      setView: 'once',
      flyTo: true,
      keepCurrentZoomLevel: true,
      strings: {{ title: "Show my location" }}
    }}).addTo(map);

    // Address search bar (Nominatim geocoder, biased to Montreal)
    L.Control.geocoder({{
      defaultMarkGeocode: false,
      placeholder: 'Search address...',
      geocoder: L.Control.Geocoder.nominatim({{
        geocodingQueryParams: {{ viewbox: '-73.97,45.40,-73.47,45.70', bounded: 1 }}
      }})
    }}).on('markgeocode', function(e) {{
      var bb = e.geocode.bbox;
      map.fitBounds(bb || e.geocode.center.toBounds(200));
      L.marker(e.geocode.center).addTo(map)
        .bindPopup(e.geocode.name).openPopup();
    }}).addTo(map);

    // --- Viewport-filtered rendering ---
    var layerData = {{}};    // raw GeoJSON per layer key
    var layerOpts = {{}};    // L.geoJSON options per layer key
    var layerGroups = {{}};  // L.layerGroup shown on map

    function featureBBox(f) {{
      // Quick bbox from coordinates (works for points, lines, polygons)
      var coords = f.geometry.coordinates;
      if (f.geometry.type === 'Point') return [coords[1], coords[0], coords[1], coords[0]];
      var flat = coords;
      // Flatten one level for LineString; two for Polygon/MultiLineString
      if (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiLineString')
        flat = [].concat.apply([], coords);
      if (f.geometry.type === 'MultiPolygon')
        flat = [].concat.apply([], [].concat.apply([], coords));
      var minLat = 90, maxLat = -90, minLng = 180, maxLng = -180;
      for (var i = 0; i < flat.length; i++) {{
        var c = flat[i];
        if (c[0] < minLng) minLng = c[0];
        if (c[0] > maxLng) maxLng = c[0];
        if (c[1] < minLat) minLat = c[1];
        if (c[1] > maxLat) maxLat = c[1];
      }}
      return [minLat, minLng, maxLat, maxLng];
    }}

    function renderLayer(key) {{
      var group = layerGroups[key];
      if (!group || !layerData[key]) return;
      // Only render if this layer is on the map
      if (!map.hasLayer(group)) return;
      group.clearLayers();
      var b = map.getBounds();
      var visible = {{type: 'FeatureCollection', features: []}};
      var features = layerData[key].features;
      for (var i = 0; i < features.length; i++) {{
        var bb = featureBBox(features[i]);
        // Check if feature bbox intersects map bounds
        if (bb[2] >= b.getSouth() && bb[0] <= b.getNorth()
            && bb[3] >= b.getWest() && bb[1] <= b.getEast()) {{
          visible.features.push(features[i]);
        }}
      }}
      if (visible.features.length > 0) {{
        L.geoJSON(visible, layerOpts[key]).addTo(group);
      }}
    }}

    var skipNextRender = false;

    function renderAll() {{
      // Skip one render cycle after popup opens (auto-pan would destroy it)
      if (skipNextRender) {{ skipNextRender = false; return; }}
      for (var key in layerData) {{ renderLayer(key); }}
    }}

    map.on('popupopen', function() {{ skipNextRender = true; }});
    map.on('moveend', renderAll);
    map.on('overlayadd', function(e) {{
      // Find the key for the layer that was just toggled on
      for (var key in layerGroups) {{
        if (layerGroups[key] === e.layer) {{ renderLayer(key); break; }}
      }}
    }});

{layers_init}

{default_adds}

    {legend_map_js}
    L.control.layers(null, {{
{overlays_obj}
    }}, {{collapsed: window.innerWidth < 768}}).addTo(map);

    // Collapse layer control when tapping elsewhere on mobile
    map.on('click', function() {{
      var el = document.querySelector('.leaflet-control-layers');
      if (el) el.classList.remove('leaflet-control-layers-expanded');
    }});

    // Inject colored swatches into layer control labels
    document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function(label) {{
      var span = label.querySelector('span');
      if (!span) return;
      var name = span.textContent.trim();
      var info = legendInfo[name];
      if (!info) return;
      var swatch = document.createElement('span');
      swatch.className = 'legend-swatch ' + (info.isPoint ? 'legend-swatch-dot' : 'legend-swatch-line');
      swatch.style.backgroundColor = info.color;
      span.parentNode.insertBefore(swatch, span);
    }});

    // --- Info box (disclaimer + GitHub link) ---
    var infoControl = L.control({{position: 'bottomleft'}});
    infoControl.onAdd = function() {{
      var container = L.DomUtil.create('div', 'info-wrapper');
      var btn = L.DomUtil.create('button', 'info-toggle', container);
      btn.textContent = '\u24d8';
      btn.title = 'Info';
      var div = L.DomUtil.create('div', 'info-control', container);
      div.innerHTML =
        '<b>Montreal Free Parking Finder</b><br>' +
        'Hobby project \u2014 not official. Data may be inaccurate.<br>' +
        '<a href="stats.html">Statistics</a>' +
        ' · <a href="https://github.com/yfarjoun/montreal-parking" target="_blank">GitHub</a>' +
        ' · <a href="https://github.com/yfarjoun/montreal-parking/issues" target="_blank">Report a bug</a>' +
        '<br><span style="color:#aaa;font-size:10px">' +
        'v{__version__}{" \u00b7 data: " + data_date if data_date else ""}</span>';
      btn.addEventListener('click', function(e) {{
        e.stopPropagation();
        container.classList.add('info-expanded');
      }});
      L.DomEvent.disableClickPropagation(container);
      document.addEventListener('click', function() {{
        container.classList.remove('info-expanded');
      }});
      return container;
    }};
    infoControl.addTo(map);

    // --- Droppable pin + shareable URL ---
    var popupJustOpened = false;
    map.on('popupopen', function() {{ popupJustOpened = true; }});

    var droppedPin = null;

    // Restore pin and view from URL hash: #lat,lng,zoom or #lat,lng
    function restoreFromHash() {{
      var h = window.location.hash.replace('#', '');
      if (!h) return;
      var parts = h.split(',').map(Number);
      if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {{
        var lat = parts[0], lng = parts[1], z = parts[2] || {zoom};
        map.setView([lat, lng], z);
        placePin(lat, lng);
      }}
    }}

    function placePin(lat, lng) {{
      if (droppedPin) map.removeLayer(droppedPin);
      var shareUrl = window.location.origin + window.location.pathname
        + '#' + lat.toFixed(5) + ',' + lng.toFixed(5) + ',' + map.getZoom();
      var popupHtml = '<b>Dropped pin</b><br>'
        + '<a href="' + shareUrl + '" onclick="navigator.clipboard'
        + '.writeText(this.href);this.textContent=\\'Copied!\\';'
        + 'return false;">Copy share link</a>';
      droppedPin = L.marker([lat, lng]).addTo(map)
        .bindPopup(popupHtml).openPopup();
      window.location.hash = lat.toFixed(5) + ','
        + lng.toFixed(5) + ',' + map.getZoom();
    }}

    map.on('click', function(e) {{
      if (popupJustOpened) {{ popupJustOpened = false; return; }}
      placePin(e.latlng.lat, e.latlng.lng);
    }});

    restoreFromHash();

    // --- Driving Mode ---
    {category_colors_js}

    var drivingMode = false;
    var drivingWatchId = null;
    var drivingHeading = 0;
    var drivingPositions = [];
    var compassHeading = null;
    var drivingMarker = null;
    var sideBarLastUpdate = 0;
    var longPressTriggered = false;
    var longPressTimer = null;

    // Side bar elements
    var leftBar = document.createElement('div');
    leftBar.className = 'driving-bar driving-bar-left';
    document.body.appendChild(leftBar);
    var rightBar = document.createElement('div');
    rightBar.className = 'driving-bar driving-bar-right';
    document.body.appendChild(rightBar);

    // --- Geo utilities ---
    function haversineDist(lat1, lng1, lat2, lng2) {{
      var R = 6371000;
      var dLat = (lat2 - lat1) * Math.PI / 180;
      var dLng = (lng2 - lng1) * Math.PI / 180;
      var a = Math.sin(dLat/2) * Math.sin(dLat/2)
            + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180)
              * Math.sin(dLng/2) * Math.sin(dLng/2);
      return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }}

    function bearingBetween(lat1, lng1, lat2, lng2) {{
      var dLng = (lng2 - lng1) * Math.PI / 180;
      var la1 = lat1 * Math.PI / 180, la2 = lat2 * Math.PI / 180;
      var y = Math.sin(dLng) * Math.cos(la2);
      var x = Math.cos(la1) * Math.sin(la2)
            - Math.sin(la1) * Math.cos(la2) * Math.cos(dLng);
      return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
    }}

    function sideOfHeading(heading, lat, lng, pLat, pLng) {{
      var hRad = heading * Math.PI / 180;
      // Cross product: positive = left, negative = right
      var cross = Math.sin(hRad) * (pLat - lat) - Math.cos(hRad) * (pLng - lng);
      return cross > 0 ? 'left' : 'right';
    }}

    // --- Compass ---
    function onCompass(e) {{
      if (e.webkitCompassHeading !== undefined) compassHeading = e.webkitCompassHeading;
      else if (e.alpha !== null) compassHeading = (360 - e.alpha) % 360;
    }}

    function startCompass() {{
      if (typeof DeviceOrientationEvent !== 'undefined'
          && typeof DeviceOrientationEvent.requestPermission === 'function') {{
        DeviceOrientationEvent.requestPermission().then(function(s) {{
          if (s === 'granted') window.addEventListener('deviceorientation', onCompass);
        }}).catch(function() {{}});
      }} else if (typeof DeviceOrientationEvent !== 'undefined') {{
        window.addEventListener('deviceorientation', onCompass);
      }}
    }}

    function stopCompass() {{
      window.removeEventListener('deviceorientation', onCompass);
      compassHeading = null;
    }}

    // --- Directional marker ---
    var arrowIcon = L.divIcon({{
      className: 'driving-arrow',
      html: '<svg viewBox="0 0 24 24" width="32" height="32" class="driving-arrow-svg">'
          + '<polygon points="12,2 4,22 12,17 20,22" fill="#3498db" stroke="#fff" stroke-width="1.5"/></svg>',
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    }});

    function updateDrivingMarker(lat, lng, heading) {{
      if (!drivingMarker) {{
        drivingMarker = L.marker([lat, lng], {{icon: arrowIcon, zIndexOffset: 1000}}).addTo(map);
      }} else {{
        drivingMarker.setLatLng([lat, lng]);
      }}
      var el = drivingMarker.getElement();
      if (el) {{
        var svg = el.querySelector('.driving-arrow-svg');
        if (svg) svg.style.transform = 'rotate(' + heading + 'deg)';
      }}
    }}

    // --- Side bar update ---
    function updateSideBars(lat, lng, heading) {{
      var closestLeft = {{dist: Infinity, color: null}};
      var closestRight = {{dist: Infinity, color: null}};
      var sr = 0.0005; // ~50m in degrees
      for (var key in layerData) {{
        if (!layerData[key] || !categoryColors[key]) continue;
        var color = categoryColors[key];
        var features = layerData[key].features;
        for (var i = 0; i < features.length; i++) {{
          var f = features[i];
          var coordSets = f.geometry.type === 'MultiLineString' ? f.geometry.coordinates
                        : f.geometry.type === 'LineString' ? [f.geometry.coordinates] : null;
          if (!coordSets) continue;
          for (var s = 0; s < coordSets.length; s++) {{
            var coords = coordSets[s];
            // Quick bbox pre-filter
            var inRange = false;
            for (var j = 0; j < coords.length; j++) {{
              if (Math.abs(coords[j][0] - lng) < sr && Math.abs(coords[j][1] - lat) < sr) {{
                inRange = true; break;
              }}
            }}
            if (!inRange) continue;
            // Find nearest vertex, classify side
            for (var j = 0; j < coords.length; j++) {{
              var d = haversineDist(lat, lng, coords[j][1], coords[j][0]);
              if (d > 50) continue;
              var side = sideOfHeading(heading, lat, lng, coords[j][1], coords[j][0]);
              if (side === 'left' && d < closestLeft.dist) {{
                closestLeft = {{dist: d, color: color}};
              }} else if (side === 'right' && d < closestRight.dist) {{
                closestRight = {{dist: d, color: color}};
              }}
            }}
          }}
        }}
      }}
      leftBar.style.backgroundColor = closestLeft.color || '#888';
      rightBar.style.backgroundColor = closestRight.color || '#888';
    }}

    // --- Enter / exit driving mode ---
    function enterDrivingMode() {{
      drivingMode = true;
      leftBar.style.display = 'block';
      rightBar.style.display = 'block';
      drivingPositions = [];
      locateControl.stop();
      map.setZoom(17);
      if (!navigator.geolocation) return;
      drivingWatchId = navigator.geolocation.watchPosition(
        onDrivingPosition,
        function() {{}},
        {{enableHighAccuracy: true, maximumAge: 2000, timeout: 10000}}
      );
      startCompass();
      var btn = document.querySelector('.leaflet-control-locate a');
      if (btn) btn.classList.add('driving-active');
    }}

    function exitDrivingMode() {{
      drivingMode = false;
      leftBar.style.display = 'none';
      rightBar.style.display = 'none';
      if (drivingWatchId !== null) {{
        navigator.geolocation.clearWatch(drivingWatchId);
        drivingWatchId = null;
      }}
      stopCompass();
      if (drivingMarker) {{
        map.removeLayer(drivingMarker);
        drivingMarker = null;
      }}
      var btn = document.querySelector('.leaflet-control-locate a');
      if (btn) btn.classList.remove('driving-active');
    }}

    function onDrivingPosition(pos) {{
      var lat = pos.coords.latitude, lng = pos.coords.longitude;
      drivingPositions.push({{lat: lat, lng: lng, time: Date.now()}});
      if (drivingPositions.length > 5) drivingPositions.shift();
      // Heading from GPS when moving, compass when stationary
      if (drivingPositions.length >= 2) {{
        var p1 = drivingPositions[drivingPositions.length - 2];
        var p2 = drivingPositions[drivingPositions.length - 1];
        if (haversineDist(p1.lat, p1.lng, p2.lat, p2.lng) > 3) {{
          drivingHeading = bearingBetween(p1.lat, p1.lng, p2.lat, p2.lng);
        }} else if (compassHeading !== null) {{
          drivingHeading = compassHeading;
        }}
      }} else if (compassHeading !== null) {{
        drivingHeading = compassHeading;
      }}
      map.setView([lat, lng], Math.max(map.getZoom(), 17), {{animate: true}});
      updateDrivingMarker(lat, lng, drivingHeading);
      // Throttle side bar updates to every 2s
      var now = Date.now();
      if (now - sideBarLastUpdate > 2000) {{
        sideBarLastUpdate = now;
        updateSideBars(lat, lng, drivingHeading);
      }}
    }}

    // --- Long-press on locate button ---
    (function() {{
      var btn = document.querySelector('.leaflet-control-locate a');
      if (!btn) return;
      btn.addEventListener('mousedown', function() {{
        longPressTimer = setTimeout(function() {{
          longPressTimer = null; longPressTriggered = true;
          if (drivingMode) exitDrivingMode(); else enterDrivingMode();
        }}, 800);
      }});
      btn.addEventListener('mouseup', function() {{
        if (longPressTimer) {{ clearTimeout(longPressTimer); longPressTimer = null; }}
      }});
      btn.addEventListener('mouseleave', function() {{
        if (longPressTimer) {{ clearTimeout(longPressTimer); longPressTimer = null; }}
      }});
      btn.addEventListener('contextmenu', function(e) {{ e.preventDefault(); }});
      btn.addEventListener('touchstart', function(e) {{
        e.preventDefault();
        longPressTimer = setTimeout(function() {{
          longPressTimer = null; longPressTriggered = true;
          if (drivingMode) exitDrivingMode(); else enterDrivingMode();
        }}, 800);
      }});
      btn.addEventListener('touchend', function() {{
        if (longPressTimer) {{ clearTimeout(longPressTimer); longPressTimer = null; }}
      }});
      btn.addEventListener('touchmove', function() {{
        if (longPressTimer) {{ clearTimeout(longPressTimer); longPressTimer = null; }}
      }});
      btn.addEventListener('click', function(e) {{
        if (longPressTriggered) {{
          e.stopImmediatePropagation(); e.preventDefault();
          longPressTriggered = false;
        }}
      }}, true);
    }})();
  </script>
  <script data-goatcounter="https://yfarjoun.goatcounter.com/count"
          async src="//gc.zgo.at/count.js"></script>
</body>
</html>"""


def build_map(
    intervals_gdf: gpd.GeoDataFrame,
    signs_df: pd.DataFrame,
    unsnapped_signs: pd.DataFrame | None = None,
    roads_gdf: gpd.GeoDataFrame | None = None,
    *,
    borough: str | None = None,
    debug: bool = False,
    data_date: str = "",
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
        signaled = intervals_wgs[intervals_wgs["category"] != NO_DATA]
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

    # Export poles GeoJSON (off by default), excluding DEUX COTES copies
    original_signs = signs_df
    if "is_deux_cotes_copy" in signs_df.columns:
        original_signs = signs_df[~signs_df["is_deux_cotes_copy"]]
    poles_dest = data_dir / "poles.geojson"
    if _build_pole_geojson(original_signs, poles_dest, label_prefix="Pole"):
        layers.append({
            "var": "layer_poles",
            "file": "poles.geojson",
            "name": "Sign Poles",
            "color": "#333",
            "default_on": False,
            "is_point": True,
        })

    # Export DEUX COTES copies (debug only)
    if debug and "is_deux_cotes_copy" in signs_df.columns and roads_gdf is not None:
        copies = signs_df[signs_df["is_deux_cotes_copy"]]
        if not copies.empty:
            copies_offset = _offset_deux_cotes_copies(copies, roads_gdf)
            copies_dest = data_dir / "deux_cotes_copies.geojson"
            if _build_pole_geojson(copies_offset, copies_dest, label_prefix="DC Copy", group_by_side=True):
                layers.append({
                    "var": "layer_dc_copies",
                    "file": "deux_cotes_copies.geojson",
                    "name": "DEUX COTES Copies",
                    "color": "#e056fd",
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
    html_content = _build_html_shell(layers, center, zoom, data_date=data_date)
    index_path = OUTPUT_DIR / "index.html"
    with open(index_path, "w") as f:
        f.write(html_content)
    print(f"    HTML shell: {index_path} ({len(html_content) / 1e3:.0f} KB)")
