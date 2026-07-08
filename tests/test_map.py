"""Tests for map GeoJSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from montreal_parking.constants import CRS_WGS84, IntervalCategory
from montreal_parking.map import _export_cleaning_geojson


def _intervals(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, crs=CRS_WGS84)


def test_cleaning_geojson_includes_only_parsed(tmp_path: Path) -> None:
    gdf = _intervals([
        {
            "street_name": "Rue Test",
            "category": IntervalCategory.FREE,
            "cleaning": [{
                "weekdays": [1], "start_min": 480, "end_min": 540,
                "month_start": (4, 1), "month_end": (12, 1),
            }],
            "cleaning_text": "Cleaning: Tue 8:00–9:00 · Apr 1 – Dec 1",
            "geometry": LineString([(-73.57, 45.50), (-73.569, 45.50)]),
        },
        {
            "street_name": "Rue Sans",
            "category": IntervalCategory.FREE,
            "cleaning": [],
            "cleaning_text": "",
            "geometry": LineString([(-73.57, 45.51), (-73.569, 45.51)]),
        },
    ])
    dest = tmp_path / "cleaning.geojson"
    assert _export_cleaning_geojson(gdf, dest) is True

    data = json.loads(dest.read_text())
    assert len(data["features"]) == 1
    props = data["features"][0]["properties"]
    assert "Cleaning: Tue" in props["popup_html"]
    assert props["cleaning"][0]["weekdays"] == [1]
    assert props["cleaning"][0]["month_start"] == [4, 1]  # tuple → JSON array


def test_cleaning_geojson_empty_returns_false(tmp_path: Path) -> None:
    gdf = _intervals([{
        "street_name": "Rue Sans",
        "category": IntervalCategory.FREE,
        "cleaning": [],
        "cleaning_text": "",
        "geometry": LineString([(-73.57, 45.51), (-73.569, 45.51)]),
    }])
    dest = tmp_path / "cleaning.geojson"
    assert _export_cleaning_geojson(gdf, dest) is False
    assert not dest.exists()
