"""Tests for interval reconstruction and pole snapping.

Uses synthetic road geometries and sign data to test the spatial logic
without downloading real Montreal data.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from main import (
    CRS_MTM8,
    reconstruct_intervals,
    snap_poles_to_roads,
)


def _make_road(
    x_start: float = 300000,
    y_start: float = 5045000,
    length: float = 200,
    id_trc: int = 1,
    name: str = "Rue Test",
) -> gpd.GeoDataFrame:
    """Create a simple west-to-east road in MTM8 coords."""
    geom = LineString([(x_start, y_start), (x_start + length, y_start)])
    return gpd.GeoDataFrame(
        [{"ID_TRC": id_trc, "NOM_VOIE": name, "geometry": geom}],
        crs=CRS_MTM8,
    )


def _make_signs_df(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Build a signs DataFrame from a list of row dicts.

    Each row needs at minimum: POTEAU_ID_POT, Latitude, Longitude,
    DESCRIPTION_RPA, FLECHE_PAN, sign_category, is_restrictive.
    """
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Snap poles to roads
# ---------------------------------------------------------------------------


class TestSnapPolesToRoads:
    """Tests for snapping pole locations to road geometries."""

    def test_pole_snaps_to_nearest_road(self) -> None:
        """A pole 5m north of a road should snap to that road."""
        roads = _make_road()
        # Pole at road midpoint, 5m north (in WGS84 approx)
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 1,
            "Latitude": 45.5225,
            "Longitude": -73.5700,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 0,
            "sign_category": "no_parking",
            "is_restrictive": True,
            "NOM_ARROND": "Le Plateau-Mont-Royal",
        }])
        # Use WGS84 roads for this test
        roads_wgs = roads.to_crs("EPSG:4326")
        # Place pole near the road
        road_centroid = roads_wgs.geometry.iloc[0].interpolate(0.5, normalized=True)
        signs["Longitude"] = road_centroid.x
        signs["Latitude"] = road_centroid.y + 0.00004  # ~4.5m north

        snapped, unsnapped = snap_poles_to_roads(signs, roads_wgs)
        assert len(snapped) == 1
        assert snapped.iloc[0]["ID_TRC"] == 1
        assert unsnapped.empty

    def test_pole_too_far_is_unsnapped(self) -> None:
        """A pole >20m from any road should end up in unsnapped."""
        roads = _make_road()
        roads_wgs = roads.to_crs("EPSG:4326")
        road_centroid = roads_wgs.geometry.iloc[0].interpolate(0.5, normalized=True)
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 99,
            "Latitude": road_centroid.y + 0.005,  # ~500m north
            "Longitude": road_centroid.x,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 0,
            "sign_category": "no_parking",
            "is_restrictive": True,
            "NOM_ARROND": "Le Plateau-Mont-Royal",
        }])
        snapped, unsnapped = snap_poles_to_roads(signs, roads_wgs)
        assert snapped.empty
        assert len(unsnapped) == 1

    def test_side_determination(self) -> None:
        """Poles north of a W-E road should be 'left', south should be 'right'."""
        roads = _make_road()
        roads_wgs = roads.to_crs("EPSG:4326")
        midpoint = roads_wgs.geometry.iloc[0].interpolate(0.5, normalized=True)

        north_pole = {
            "POTEAU_ID_POT": 10,
            "Latitude": midpoint.y + 0.00004,
            "Longitude": midpoint.x,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": "unrestricted",
            "is_restrictive": False,
            "NOM_ARROND": "Test",
        }
        south_pole = {
            "POTEAU_ID_POT": 20,
            "Latitude": midpoint.y - 0.00004,
            "Longitude": midpoint.x + 0.0001,  # slight offset so they don't overlap
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": "unrestricted",
            "is_restrictive": False,
            "NOM_ARROND": "Test",
        }
        signs = _make_signs_df([north_pole, south_pole])
        snapped, _ = snap_poles_to_roads(signs, roads_wgs)

        sides = snapped.set_index("POTEAU_ID_POT")["side"]
        assert sides[10] == "left"
        assert sides[20] == "right"


# ---------------------------------------------------------------------------
# Reconstruct intervals
# ---------------------------------------------------------------------------


def _make_snapped_signs(
    poles: list[dict[str, object]],
) -> pd.DataFrame:
    """Build a snapped-signs DataFrame suitable for reconstruct_intervals.

    Each pole dict needs: POTEAU_ID_POT, projection_distance, side, ID_TRC,
    DESCRIPTION_RPA, FLECHE_PAN, sign_category, is_restrictive.
    """
    return pd.DataFrame(poles)


class TestReconstructIntervals:
    """Tests for interval reconstruction from snapped signs."""

    def test_single_pole_creates_two_intervals(self) -> None:
        """One pole on a road should create intervals before and after it."""
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 0,  # both directions
            "sign_category": "no_parking",
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        # Should have intervals on the right side (from signs) + no_data on left
        right_intervals = intervals[intervals["side"] == "right"]
        assert len(right_intervals) == 2  # before + after the pole

    def test_two_restrictive_poles_creates_restricted_interval(self) -> None:
        """Two no-parking poles pointing inward → restricted interval between them."""
        roads = _make_road(length=100)
        signs = _make_snapped_signs([
            {
                "POTEAU_ID_POT": 1,
                "projection_distance": 20.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 2,  # arrow 2 on right side = forward
                "sign_category": "no_parking",
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
            {
                "POTEAU_ID_POT": 2,
                "projection_distance": 80.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 3,  # arrow 3 on right side = backward
                "sign_category": "no_parking",
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
        ])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        middle = right[
            (right["start_dist"] >= 19) & (right["end_dist"] <= 81)
        ]
        assert len(middle) == 1
        assert middle.iloc[0]["category"] == "restricted"

    def test_free_interval_between_outward_arrows(self) -> None:
        """Two poles with arrows pointing away from the gap → free between them.

        This is the Poitevin poles 44998/180512 scenario.
        """
        roads = _make_road(length=100)
        signs = _make_snapped_signs([
            {
                "POTEAU_ID_POT": 44998,
                "projection_distance": 30.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 3,  # arrow 3 on right side = backward (away from gap)
                "sign_category": "no_parking",
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
            {
                "POTEAU_ID_POT": 180512,
                "projection_distance": 70.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 2,  # arrow 2 on right side = forward (away from gap)
                "sign_category": "no_parking",
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
        ])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        middle = right[
            (right["start_dist"] >= 29) & (right["end_dist"] <= 71)
        ]
        assert len(middle) == 1
        assert middle.iloc[0]["category"] == "free"

    def test_arrow_direction_flips_on_left_side(self) -> None:
        """On the left side, arrow 2=backward and arrow 3=forward (opposite of right)."""
        roads = _make_road(length=100)
        # Arrow 3 on left side = forward, arrow 2 = backward
        # So arrow 3 pointing forward from pole 1, arrow 2 pointing backward from pole 2
        # means both point INTO the gap → restricted
        signs = _make_snapped_signs([
            {
                "POTEAU_ID_POT": 1,
                "projection_distance": 20.0,
                "side": "left",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 3,  # left side: 3=forward
                "sign_category": "no_parking",
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
            {
                "POTEAU_ID_POT": 2,
                "projection_distance": 80.0,
                "side": "left",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 2,  # left side: 2=backward
                "sign_category": "no_parking",
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
        ])
        intervals = reconstruct_intervals(signs, roads)
        left = intervals[intervals["side"] == "left"]
        middle = left[
            (left["start_dist"] >= 19) & (left["end_dist"] <= 81)
        ]
        assert len(middle) == 1
        assert middle.iloc[0]["category"] == "restricted"

    def test_unsigned_side_gets_no_data(self) -> None:
        """When poles exist only on one side, the other side should be 'no_data'."""
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": "unrestricted",
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        left = intervals[intervals["side"] == "left"]
        assert len(left) == 1
        assert left.iloc[0]["category"] == "no_data"

    def test_deux_cotes_applies_to_both_sides(self) -> None:
        """A DEUX COTES sign should create intervals on both sides of the road.

        snap_poles_to_roads duplicates DEUX COTES signs to both sides,
        so reconstruct_intervals receives them on both sides already.
        """
        roads = _make_road(length=100)
        # Simulate what snap_poles_to_roads produces after DEUX COTES duplication
        base = {
            "POTEAU_ID_POT": 21799,
            "projection_distance": 50.0,
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P DEUX COTES",
            "FLECHE_PAN": 0,
            "sign_category": "no_parking",
            "is_restrictive": True,
            "NOM_VOIE": "Rue Le Jeune",
        }
        signs = _make_snapped_signs([
            {**base, "side": "right"},
            {**base, "side": "left"},
        ])
        intervals = reconstruct_intervals(signs, roads)
        # Both sides should have restricted intervals
        for side in ("left", "right"):
            side_intervals = intervals[intervals["side"] == side]
            assert not side_intervals.empty, f"No intervals on {side} side"
            cats = side_intervals["category"].unique()
            assert "restricted" in cats or "no_data" not in cats

    def test_bidirectional_arrow_applies_both_ways(self) -> None:
        """Arrow code 0 means the sign applies in both directions."""
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 0,
            "sign_category": "no_parking",
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        # Both before and after the pole should be restricted (arrow 0 = both)
        assert all(right["category"] == "restricted")
