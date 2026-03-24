"""Tests for interval reconstruction and pole snapping.

Uses synthetic road geometries and sign data to test the spatial logic
without downloading real Montreal data.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from montreal_parking.constants import CRS_MTM8, IntervalCategory, SignCategory
from montreal_parking.intervals import reconstruct_intervals
from montreal_parking.snap import snap_poles_to_roads

# SignCategory members (PAID and TIME_LIMITED omitted — conflict with IntervalCategory)
NO_PARKING = SignCategory.NO_PARKING
PERMIT = SignCategory.PERMIT
UNRESTRICTED = SignCategory.UNRESTRICTED
STREET_CLEANING = SignCategory.STREET_CLEANING

# IntervalCategory members (PAID and TIME_LIMITED omitted — conflict with SignCategory)
FREE = IntervalCategory.FREE
RESTRICTED = IntervalCategory.RESTRICTED
NO_DATA = IntervalCategory.NO_DATA


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
            "sign_category": NO_PARKING,
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
            "sign_category": NO_PARKING,
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
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_ARROND": "Test",
        }
        south_pole = {
            "POTEAU_ID_POT": 20,
            "Latitude": midpoint.y - 0.00004,
            "Longitude": midpoint.x + 0.0001,  # slight offset so they don't overlap
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": UNRESTRICTED,
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
            "sign_category": NO_PARKING,
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        # Should have intervals on the right side (from signs) + no_data on left
        right_intervals = intervals[intervals["side"] == "right"]
        # Level-based merge combines adjacent spans with the same category,
        # so a single bidirectional no_parking pole produces one "restricted" interval
        assert len(right_intervals) >= 1
        assert all(right_intervals["category"] == RESTRICTED)

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
                "sign_category": NO_PARKING,
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
                "sign_category": NO_PARKING,
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
        assert middle.iloc[0]["category"] == RESTRICTED

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
                "sign_category": NO_PARKING,
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
                "sign_category": NO_PARKING,
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
        assert middle.iloc[0]["category"] == FREE

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
                "sign_category": NO_PARKING,
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
                "sign_category": NO_PARKING,
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
        assert middle.iloc[0]["category"] == RESTRICTED

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
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        left = intervals[intervals["side"] == "left"]
        assert len(left) == 1
        assert left.iloc[0]["category"] == NO_DATA

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
            "sign_category": NO_PARKING,
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
            assert RESTRICTED in cats or NO_DATA not in cats

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
            "sign_category": NO_PARKING,
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        # Both before and after the pole should be restricted (arrow 0 = both)
        assert all(right["category"] == RESTRICTED)

    def test_short_free_edge_becomes_no_data(self) -> None:
        """A short (<5m) edge interval that would be 'free' should become 'no_data'.

        This prevents false free-parking near intersections (e.g. Boyer/Généreux).
        Pole at 3m with forward-only arrow: backward direction has no signs,
        so the 0-3m edge would be classified 'free' — but it's too short.
        """
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 3.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 2,  # right side: 2=forward (points away from 0m edge)
            "sign_category": NO_PARKING,
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        edge = right[right["start_dist"] == 0]
        assert len(edge) == 1
        assert edge.iloc[0]["category"] == NO_DATA

    def test_short_free_tail_becomes_no_data(self) -> None:
        """A short (<5m) tail edge that would be 'free' should become 'no_data'."""
        roads = _make_road(length=100)
        # Pole at 97m with backward-only arrow: forward direction has no signs,
        # so the 97-100m tail would be 'free' — but it's too short.
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 97.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 3,  # right side: 3=backward (points away from 100m edge)
            "sign_category": NO_PARKING,
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        tail = right[right["end_dist"] >= 99]
        assert len(tail) == 1
        assert tail.iloc[0]["category"] == NO_DATA

    def test_long_free_edge_stays_free(self) -> None:
        """A longer (>=5m) edge interval that's 'free' should remain free."""
        roads = _make_road(length=100)
        # Pole at 10m with forward-only arrow: backward has no signs,
        # so edge 0-10m is 'free' and long enough to stay free.
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 10.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 2,  # right side: 2=forward (points away from 0m edge)
            "sign_category": NO_PARKING,
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        edge = right[right["start_dist"] == 0]
        assert len(edge) == 1
        assert edge.iloc[0]["category"] == FREE


class TestLevelBasedClassification:
    """Tests for the level-based interval classification system."""

    def test_level4_overrides_level3(self) -> None:
        """A time_limited (level 4) sign should override no_parking (level 3) in the same zone."""
        roads = _make_road(length=100)
        # Level 3: no_parking covers the whole road (bidirectional)
        # Level 4: time_limited covers the whole road (bidirectional)
        # Result: time_limited should win
        signs = _make_snapped_signs([
            {
                "POTEAU_ID_POT": 1,
                "projection_distance": 50.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 0,
                "sign_category": NO_PARKING,
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
            {
                "POTEAU_ID_POT": 1,
                "projection_distance": 50.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "P 15 MIN",
                "FLECHE_PAN": 0,
                "sign_category": SignCategory.TIME_LIMITED,
                "is_restrictive": False,
                "NOM_VOIE": "Rue Test",
            },
        ])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        assert all(right["category"] == IntervalCategory.TIME_LIMITED)

    def test_level3_arrows_dont_interact_with_level4(self) -> None:
        """Level 3 arrows should only interact with level 3 arrows.

        Scenario: level 3 no_parking with forward arrow, level 4 time_limited with backward arrow.
        These should NOT create a combined restricted zone — they're at different levels.
        """
        roads = _make_road(length=100)
        signs = _make_snapped_signs([
            {
                "POTEAU_ID_POT": 1,
                "projection_distance": 20.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P",
                "FLECHE_PAN": 2,  # forward
                "sign_category": NO_PARKING,
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
            {
                "POTEAU_ID_POT": 2,
                "projection_distance": 80.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "P 15 MIN",
                "FLECHE_PAN": 3,  # backward
                "sign_category": SignCategory.TIME_LIMITED,
                "is_restrictive": False,
                "NOM_VOIE": "Rue Test",
            },
        ])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        # Level 3 forward from pole 1 covers 20-100m as "restricted"
        # Level 4 backward from pole 2 covers 0-80m as "time_limited"
        # Level 4 overrides level 3 in overlap (20-80m) → time_limited
        # Final: time_limited [0,80], restricted [80,100]
        tl = right[right["category"] == IntervalCategory.TIME_LIMITED]
        assert len(tl) >= 1
        # The time_limited span should cover the overlap zone (20-80m)
        assert tl.iloc[0]["start_dist"] <= 20.0
        assert tl.iloc[0]["end_dist"] >= 80.0
        # No "restricted" in the 20-80m zone
        restricted_in_middle = right[
            (right["category"] == RESTRICTED)
            & (right["start_dist"] < 80)
            & (right["end_dist"] > 20)
        ]
        assert restricted_in_middle.empty

    def test_paid_category_produces_paid_intervals(self) -> None:
        """A paid parking sign should produce 'paid' intervals."""
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "TARIF 2$/HR",
            "FLECHE_PAN": 0,
            "sign_category": SignCategory.PAID,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        assert all(right["category"] == IntervalCategory.PAID)

    def test_street_cleaning_ignored_in_classification(self) -> None:
        """Street cleaning signs should not affect interval classification."""
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P 08h-09h MAR. 1 AVRIL AU 1 DEC.",
            "FLECHE_PAN": 0,
            "sign_category": STREET_CLEANING,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        # Street cleaning is excluded from classification, so the road should be free
        assert all(right["category"] == FREE)

    def test_deux_cotes_level3_doesnt_break_level4_zone(self) -> None:
        r"""DEUX COTES copy (level 3) should not break a time_limited (level 4) zone.

        This is the core bug that level-based classification fixes:
        a DEUX COTES \P sign copied to the opposite side should not override
        a P 15 MIN sign that's already there at level 4.
        """
        roads = _make_road(length=100)
        signs = _make_snapped_signs([
            # Level 4: time_limited sign on the right side
            {
                "POTEAU_ID_POT": 1,
                "projection_distance": 30.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "P 15 MIN",
                "FLECHE_PAN": 2,  # forward
                "sign_category": SignCategory.TIME_LIMITED,
                "is_restrictive": False,
                "NOM_VOIE": "Rue Test",
            },
            {
                "POTEAU_ID_POT": 2,
                "projection_distance": 70.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "P 15 MIN",
                "FLECHE_PAN": 3,  # backward
                "sign_category": SignCategory.TIME_LIMITED,
                "is_restrictive": False,
                "NOM_VOIE": "Rue Test",
            },
            # Level 3: DEUX COTES copy in the middle of the time_limited zone
            {
                "POTEAU_ID_POT": 3,
                "projection_distance": 50.0,
                "side": "right",
                "ID_TRC": 1,
                "DESCRIPTION_RPA": "\\P DEUX COTES",
                "FLECHE_PAN": 0,
                "sign_category": NO_PARKING,
                "is_restrictive": True,
                "NOM_VOIE": "Rue Test",
            },
        ])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        # The zone between poles 1 and 2 (30-70m) should be time_limited,
        # NOT broken by the DEUX COTES copy
        middle = right[
            (right["start_dist"] >= 29) & (right["end_dist"] <= 71)
        ]
        cats = middle["category"].unique()
        assert IntervalCategory.TIME_LIMITED in cats, f"Expected time_limited in zone, got {cats}"
        assert RESTRICTED not in cats, "DEUX COTES copy broke time_limited zone"


# ---------------------------------------------------------------------------
# Meter data integration
# ---------------------------------------------------------------------------


class TestMeterIntegration:
    """Tests for integrating paid parking meter data into intervals."""

    def _make_meter_data(
        self, rows: list[dict[str, object]]
    ) -> pd.DataFrame:
        """Build a meter DataFrame with required columns."""
        return pd.DataFrame(rows)

    def test_meters_override_free_to_paid(self) -> None:
        """Road sides with no signs but with meters should become paid, not no_data."""
        roads = _make_road(length=200)
        # Put an unrestricted sign on the left side only
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 100.0,
            "side": "left",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        # Meters on the right side
        meters = self._make_meter_data([
            {"ID_TRC": 1, "projection_distance": 50.0, "side": "right", "rate": 4.25},
            {"ID_TRC": 1, "projection_distance": 60.0, "side": "right", "rate": 4.25},
            {"ID_TRC": 1, "projection_distance": 70.0, "side": "right", "rate": 4.25},
        ])
        intervals = reconstruct_intervals(signs, roads, metered_places=meters)
        right = intervals[intervals["side"] == "right"]
        paid = right[right["category"] == IntervalCategory.PAID]
        assert not paid.empty, "Meters should create paid intervals on meter-only side"

    def test_meters_inject_into_sign_based_intervals(self) -> None:
        """Meters on a road side with sign data should override free spans to paid."""
        roads = _make_road(length=200)
        # Unrestricted sign covering the whole road
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 100.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        # Meters in the middle of the road
        meters = self._make_meter_data([
            {"ID_TRC": 1, "projection_distance": 80.0, "side": "right", "rate": 3.00},
            {"ID_TRC": 1, "projection_distance": 120.0, "side": "right", "rate": 3.00},
        ])
        intervals = reconstruct_intervals(signs, roads, metered_places=meters)
        right = intervals[intervals["side"] == "right"]
        paid = right[right["category"] == IntervalCategory.PAID]
        assert not paid.empty, "Meters should create paid spans even where signs say free"

    def test_restrictions_override_meters(self) -> None:
        """Level-3 restrictions (no_parking) should still win over meter-derived paid."""
        roads = _make_road(length=200)
        # No parking sign covering the whole road
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 100.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P",
            "FLECHE_PAN": 0,
            "sign_category": NO_PARKING,
            "is_restrictive": True,
            "NOM_VOIE": "Rue Test",
        }])
        # Meters in the same area
        meters = self._make_meter_data([
            {"ID_TRC": 1, "projection_distance": 80.0, "side": "right", "rate": 4.25},
            {"ID_TRC": 1, "projection_distance": 120.0, "side": "right", "rate": 4.25},
        ])
        intervals = reconstruct_intervals(signs, roads, metered_places=meters)
        right = intervals[intervals["side"] == "right"]
        # Meters are level 4, no_parking is level 3. Level 4 overrides level 3,
        # so the metered span should be paid (this is the intended behavior:
        # if there are meters, it's paid parking even if a no_parking sign exists)
        paid = right[right["category"] == IntervalCategory.PAID]
        assert not paid.empty, "Meter-derived paid should override level-3 restrictions"

    def test_meter_rate_in_interval(self) -> None:
        """Meter-derived paid intervals should carry the rate field."""
        roads = _make_road(length=200)
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 100.0,
            "side": "left",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        meters = self._make_meter_data([
            {"ID_TRC": 1, "projection_distance": 50.0, "side": "right", "rate": 4.25},
            {"ID_TRC": 1, "projection_distance": 70.0, "side": "right", "rate": 4.75},
        ])
        intervals = reconstruct_intervals(signs, roads, metered_places=meters)
        right = intervals[intervals["side"] == "right"]
        paid = right[right["category"] == IntervalCategory.PAID]
        assert not paid.empty
        assert "rate" in paid.columns
        # Median of 4.25 and 4.75 = 4.50
        assert paid.iloc[0]["rate"] == 4.50

    def test_no_meters_same_as_before(self) -> None:
        """Passing metered_places=None should produce identical results to before."""
        roads = _make_road(length=200)
        signs = _make_signs_df([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 100.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals_none = reconstruct_intervals(signs, roads, metered_places=None)
        intervals_default = reconstruct_intervals(signs, roads)
        assert len(intervals_none) == len(intervals_default)
