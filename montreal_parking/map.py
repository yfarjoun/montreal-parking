"""Build the interactive Folium map."""

from __future__ import annotations

import html
from typing import Any

import folium
import geopandas as gpd
import pandas as pd

from montreal_parking.constants import (
    COLOR_FREE,
    COLOR_NO_DATA,
    COLOR_RESTRICTED,
    COLOR_TIME_LIMITED,
    CRS_WGS84,
    GOOGLE_TILES,
)


def _style_function(category: str) -> Any:
    """Return a Folium style function for a given category."""
    color_map = {
        "free": COLOR_FREE,
        "time_limited": COLOR_TIME_LIMITED,
        "restricted": COLOR_RESTRICTED,
        "no_data": COLOR_NO_DATA,
    }
    color = color_map.get(category, COLOR_NO_DATA)

    def style(_feature: Any) -> dict[str, Any]:
        return {
            "color": color,
            "weight": 5,
            "opacity": 0.8,
        }

    return style


def _build_pole_sign_map(df: pd.DataFrame) -> dict[Any, str]:
    """Build a mapping from pole ID to HTML sign descriptions."""
    result: dict[Any, str] = (
        df.groupby("POTEAU_ID_POT")
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
    return result


def _add_pole_markers(
    fg: folium.FeatureGroup,
    poles_df: pd.DataFrame,
    sign_map: dict[Any, str],
    label_prefix: str = "Pole",
    color: str = "#333",
    radius: int = 3,
    fill_color: str | None = None,
) -> None:
    """Add circle markers for poles to a FeatureGroup."""
    unique = poles_df.drop_duplicates(subset="POTEAU_ID_POT")
    for _, row in unique.iterrows():
        pid = row["POTEAU_ID_POT"]
        lat, lon = row["Latitude"], row["Longitude"]
        sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        popup_text = (
            f"<b>{label_prefix} {pid}</b><br>"
            f"{sign_map.get(pid, '')}<br>"
            f'<a href="{sv_url}" target="_blank">Street View</a>'
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color or color,
            fill_opacity=0.7 if fill_color is None else 0.9,
            popup=folium.Popup(popup_text, max_width=400),
        ).add_to(fg)


def build_map(
    intervals_gdf: gpd.GeoDataFrame,
    signs_df: pd.DataFrame,
    unsnapped_signs: pd.DataFrame | None = None,
) -> folium.Map:
    """Create an interactive Folium map showing parking zones and poles."""
    intervals_wgs = intervals_gdf.to_crs(CRS_WGS84)

    m = folium.Map(
        location=[45.5225, -73.5700],
        zoom_start=14,
        tiles=GOOGLE_TILES,
        attr="Google Maps",
    )

    # Add interval layers by category
    layer_config = [
        ("Free Parking", "free", True),
        ("Time-Limited", "time_limited", True),
        ("Restricted", "restricted", False),
        ("No Data", "no_data", False),
    ]

    for layer_name, category, show in layer_config:
        subset = intervals_wgs[intervals_wgs["category"] == category].copy()
        if subset.empty:
            continue

        cat_label = category
        subset["popup_html"] = subset.apply(
            lambda row, _cat=cat_label: (
                f"<b>{row['street_name']}</b><br>"
                f"Category: {_cat}<br>"
                f"Length: {row['length_m']:.0f}m<br>"
                f"<small>{html.escape(str(row['descriptions'])[:200]).replace(chr(92), chr(92)*2)}</small>"
            ),
            axis=1,
        )

        fg = folium.FeatureGroup(name=layer_name, show=show)
        folium.GeoJson(
            subset[["geometry", "popup_html"]],
            style_function=_style_function(category),
            popup=folium.GeoJsonPopup(fields=["popup_html"], labels=False),
        ).add_to(fg)
        fg.add_to(m)

    # Add poles layer (off by default)
    poles_fg = folium.FeatureGroup(name="Sign Poles", show=False)
    _add_pole_markers(poles_fg, signs_df, _build_pole_sign_map(signs_df))
    poles_fg.add_to(m)

    # Add unsnapped poles layer (orange markers, off by default)
    if unsnapped_signs is not None and not unsnapped_signs.empty:
        unsnapped_fg = folium.FeatureGroup(name="Unmatched Poles", show=False)
        _add_pole_markers(
            unsnapped_fg, unsnapped_signs, _build_pole_sign_map(unsnapped_signs),
            label_prefix="Unmatched Pole", color="#e67e22", radius=5, fill_color="#e67e22",
        )
        unsnapped_fg.add_to(m)

    folium.LayerControl().add_to(m)
    return m
