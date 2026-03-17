"""Tests for vectorized snap projection and side determination.

Uses synthetic road geometries and pole positions to verify that the
vectorized _compute_projection_and_side produces correct results.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from montreal_parking.constants import CRS_MTM8
from montreal_parking.snap import _compute_projection_and_side


def _make_road_geoms(
    x_start: float = 300000,
    y_start: float = 5045000,
    length: float = 200,
    id_trc: int = 1,
) -> pd.Series:  # type: ignore[type-arg]
    """Return a Series mapping ID_TRC -> road geometry (west-to-east)."""
    geom = LineString([(x_start, y_start), (x_start + length, y_start)])
    return pd.Series({id_trc: geom})


def _make_joined(
    poles: list[dict[str, object]],
) -> gpd.GeoDataFrame:
    """Build a minimal GeoDataFrame mimicking the sjoin_nearest output."""
    rows = []
    for p in poles:
        rows.append({
            "POTEAU_ID_POT": p["id"],
            "ID_TRC": p["id_trc"],
            "geometry": Point(float(p["x"]), float(p["y"])),
        })
    return gpd.GeoDataFrame(rows, crs=CRS_MTM8)


class TestComputeProjectionAndSide:
    """Tests for the vectorized _compute_projection_and_side helper."""

    def test_projection_distance_midpoint(self) -> None:
        """A pole at the midpoint of the road should project to half the road length."""
        road_geoms = _make_road_geoms(length=200)
        joined = _make_joined(
            [{"id": 1, "id_trc": 1, "x": 300100, "y": 5045005}],
        )
        proj, _ = _compute_projection_and_side(joined, road_geoms)
        assert abs(proj[0] - 100.0) < 1.0

    def test_projection_distance_start(self) -> None:
        """A pole near the start of the road should project close to 0."""
        road_geoms = _make_road_geoms(length=200)
        joined = _make_joined(
            [{"id": 1, "id_trc": 1, "x": 300005, "y": 5045005}],
        )
        proj, _ = _compute_projection_and_side(joined, road_geoms)
        assert proj[0] < 10.0

    def test_side_left_for_north_of_we_road(self) -> None:
        """A pole north of a west-to-east road should be on the left side."""
        road_geoms = _make_road_geoms(length=200)
        joined = _make_joined(
            [{"id": 1, "id_trc": 1, "x": 300100, "y": 5045010}],  # 10m north
        )
        _, sides = _compute_projection_and_side(joined, road_geoms)
        assert sides[0] == "left"

    def test_side_right_for_south_of_we_road(self) -> None:
        """A pole south of a west-to-east road should be on the right side."""
        road_geoms = _make_road_geoms(length=200)
        joined = _make_joined(
            [{"id": 1, "id_trc": 1, "x": 300100, "y": 5044990}],  # 10m south
        )
        _, sides = _compute_projection_and_side(joined, road_geoms)
        assert sides[0] == "right"

    def test_multiple_poles_vectorized(self) -> None:
        """Multiple poles should all get correct projection and side in one call."""
        road_geoms = _make_road_geoms(length=200)
        joined = _make_joined(
            [
                {"id": 1, "id_trc": 1, "x": 300050, "y": 5045010},  # left, ~50m
                {"id": 2, "id_trc": 1, "x": 300150, "y": 5044990},  # right, ~150m
            ],
        )
        proj, sides = _compute_projection_and_side(joined, road_geoms)
        assert len(proj) == 2
        assert abs(proj[0] - 50.0) < 1.0
        assert abs(proj[1] - 150.0) < 1.0
        assert sides[0] == "left"
        assert sides[1] == "right"

    def test_pole_beyond_road_end_clamps(self) -> None:
        """A pole past the end of the road should project to road length."""
        road_geoms = _make_road_geoms(length=100)
        joined = _make_joined(
            [{"id": 1, "id_trc": 1, "x": 300120, "y": 5045005}],  # 20m past end
        )
        proj, _ = _compute_projection_and_side(joined, road_geoms)
        assert proj[0] == 100.0  # clamped to road length

    def test_consistent_with_scalar(self) -> None:
        """Vectorized result should match a manual scalar computation."""
        road_geoms = _make_road_geoms(length=200)
        x, y = 300080, 5045007
        joined = _make_joined(
            [{"id": 1, "id_trc": 1, "x": x, "y": y}],
        )
        proj, sides = _compute_projection_and_side(joined, road_geoms)

        # Scalar check
        road = road_geoms.iloc[0]
        pole = Point(x, y)
        expected_proj = road.project(pole)
        assert abs(proj[0] - expected_proj) < 0.01

        # Side: north of W-E road = left
        assert sides[0] == "left"
