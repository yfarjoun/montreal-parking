"""Reconstruct parking intervals from snapped signs."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.ops import substring

from montreal_parking.constants import CRS_MTM8, MIN_FREE_EDGE_M


def _get_signs_for_direction(
    pole_id: Any,
    direction: str,
    pole_signs: dict[Any, list[dict[str, Any]]],
    forward_arrow: int,
    backward_arrow: int,
) -> list[dict[str, Any]]:
    """Get signs on a pole pointing in the given direction ('forward' or 'backward')."""
    arrow = forward_arrow if direction == "forward" else backward_arrow
    return [s for s in pole_signs.get(pole_id, []) if s["arrow"] in (arrow, 0)]


def _classify_interval(active_signs: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Classify an interval given its active signs."""
    if not active_signs:
        return "free", []
    has_restrictive = any(s["is_restrictive"] for s in active_signs)
    if has_restrictive:
        return "restricted", [s["description"] for s in active_signs]
    if any(s["category"] == "time_limited" for s in active_signs):
        return "time_limited", [s["description"] for s in active_signs]
    return "free", [s["description"] for s in active_signs]


def _make_interval(
    start_dist: float,
    end_dist: float,
    category: str,
    descriptions: list[Any],
    road_geom: Any,
    id_trc: Any,
    side: str,
    street_name: str,
) -> dict[str, Any] | None:
    """Create an interval dict with offset geometry, or None if geometry fails."""
    if end_dist - start_dist < 1.0:
        return None
    try:
        sub_geom = substring(road_geom, start_dist, end_dist)
        if sub_geom.is_empty or sub_geom.length < 0.5:
            return None
        offset_dist = 3.0 if side == "left" else -3.0
        sub_geom = sub_geom.offset_curve(offset_dist)
        if sub_geom.is_empty:
            return None
    except Exception:
        return None
    return {
        "id_trc": id_trc,
        "side": side,
        "start_dist": start_dist,
        "end_dist": end_dist,
        "length_m": end_dist - start_dist,
        "category": category,
        "street_name": street_name,
        "descriptions": "; ".join(dict.fromkeys(str(d) for d in descriptions)),
        "geometry": sub_geom,
    }


def _build_side_intervals(
    pole_data: pd.DataFrame,
    pole_signs: dict[Any, list[dict[str, Any]]],
    forward_arrow: int,
    backward_arrow: int,
    road_geom: Any,
    id_trc: Any,
    side: str,
    street_name: str,
) -> list[dict[str, Any]]:
    """Build intervals for one (road, side) group from its poles."""
    intervals: list[dict[str, Any]] = []

    def _get(pole_id: Any, direction: str) -> list[dict[str, Any]]:
        return _get_signs_for_direction(pole_id, direction, pole_signs, forward_arrow, backward_arrow)

    # Edge interval before first pole
    first_pole = pole_data.iloc[0]
    first_id = first_pole["POTEAU_ID_POT"]
    first_dist = first_pole["projection_distance"]
    if first_dist > 1.0:
        cat, descs = _classify_interval(_get(first_id, "backward"))
        # Short edges near intersections can't be free parking
        if cat == "free" and first_dist < MIN_FREE_EDGE_M:
            cat = "no_data"
        iv = _make_interval(0, first_dist, cat, descs, road_geom, id_trc, side, street_name)
        if iv:
            intervals.append(iv)

    # Intervals between consecutive poles
    for i in range(len(pole_data) - 1):
        p1 = pole_data.iloc[i]
        p2 = pole_data.iloc[i + 1]
        active = _get(p1["POTEAU_ID_POT"], "forward") + _get(p2["POTEAU_ID_POT"], "backward")
        cat, descs = _classify_interval(active)
        iv = _make_interval(
            p1["projection_distance"], p2["projection_distance"],
            cat, descs, road_geom, id_trc, side, street_name,
        )
        if iv:
            intervals.append(iv)

    # Edge interval after last pole
    last_pole = pole_data.iloc[-1]
    last_id = last_pole["POTEAU_ID_POT"]
    last_dist = last_pole["projection_distance"]
    tail_length = road_geom.length - last_dist
    if tail_length > 1.0:
        cat, descs = _classify_interval(_get(last_id, "forward"))
        # Short edges near intersections can't be free parking
        if cat == "free" and tail_length < MIN_FREE_EDGE_M:
            cat = "no_data"
        iv = _make_interval(last_dist, road_geom.length, cat, descs, road_geom, id_trc, side, street_name)
        if iv:
            intervals.append(iv)

    return intervals


def reconstruct_intervals(
    snapped_signs: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """For each (ID_TRC, side) group, determine free/restricted intervals.

    Returns a GeoDataFrame with one row per interval, including geometry
    (sub-segment of the road), category, and metadata.
    """
    roads_mtm = roads_gdf.to_crs(CRS_MTM8)
    road_geoms = roads_mtm.set_index("ID_TRC")["geometry"]
    road_names = roads_mtm.set_index("ID_TRC")["NOM_VOIE"]

    intervals: list[dict[str, Any]] = []

    # Group signs by road segment and side
    grouped = snapped_signs.groupby(["ID_TRC", "side"])

    for (id_trc, side_key), group in grouped:
        if id_trc not in road_geoms.index:
            continue
        side = str(side_key)

        road_geom = road_geoms.loc[id_trc]
        street_name = road_names.get(id_trc, "Unknown")
        pole_data = (
            group.groupby("POTEAU_ID_POT")
            .agg(projection_distance=("projection_distance", "first"))
            .sort_values("projection_distance")
            .reset_index()
        )

        # Build lookup: pole_id -> list of sign info dicts
        pole_signs: dict[Any, list[dict[str, Any]]] = {}
        for _, sign_row in group.iterrows():
            pid = sign_row["POTEAU_ID_POT"]
            if pid not in pole_signs:
                pole_signs[pid] = []
            pole_signs[pid].append({
                "category": sign_row["sign_category"],
                "description": sign_row["DESCRIPTION_RPA"],
                "arrow": sign_row.get("FLECHE_PAN", 0),
                "is_restrictive": sign_row["is_restrictive"],
            })

        # Arrow codes: 2=left, 3=right (as seen facing the sign from the street).
        # Which arrow means "forward" (increasing chainage) depends on the
        # pole's side of the road:
        #   right side: left(2)=forward, right(3)=backward
        #   left side:  left(2)=backward, right(3)=forward
        if side == "right":
            forward_arrow, backward_arrow = 2, 3
        else:
            forward_arrow, backward_arrow = 3, 2

        intervals.extend(
            _build_side_intervals(
                pole_data, pole_signs, forward_arrow, backward_arrow,
                road_geom, id_trc, side, street_name,
            )
        )

    # --- Fill gaps ---
    covered_combos = set(grouped.groups.keys())
    covered_segments = snapped_signs["ID_TRC"].unique()

    # 1) Segments with poles on one side but not the other
    for id_trc in covered_segments:
        if id_trc not in road_geoms.index:
            continue
        road_geom = road_geoms.loc[id_trc]
        street_name = road_names.get(id_trc, "Unknown")
        if road_geom.length < 1.0:
            continue
        for fill_side in ("left", "right"):
            if (id_trc, fill_side) in covered_combos:
                continue
            try:
                offset_dist = 3.0 if fill_side == "left" else -3.0
                sub_geom = road_geom.offset_curve(offset_dist)
                if sub_geom.is_empty:
                    continue
            except Exception:
                continue
            intervals.append({
                "id_trc": id_trc,
                "side": fill_side,
                "start_dist": 0,
                "end_dist": road_geom.length,
                "length_m": road_geom.length,
                "category": "no_data",
                "street_name": street_name,
                "descriptions": "No sign data for this side",
                "geometry": sub_geom,
            })

    # 2) Short gap segments (<60m) on the same street with no poles at all.
    covered_set = set(covered_segments)
    covered_streets = set(
        roads_mtm.loc[roads_mtm["ID_TRC"].isin(covered_set), "NOM_VOIE"].dropna().unique()
    )
    gap_segments = roads_mtm[
        (roads_mtm["NOM_VOIE"].isin(covered_streets))
        & (~roads_mtm["ID_TRC"].isin(covered_set))
        & (roads_mtm.geometry.length < 60)
        & (roads_mtm.geometry.length > 1)
    ]
    for _, gap_row in gap_segments.iterrows():
        gap_id = gap_row["ID_TRC"]
        if gap_id not in road_geoms.index:
            continue
        road_geom = road_geoms.loc[gap_id]
        street_name = road_names.get(gap_id, gap_row.get("NOM_VOIE", "Unknown"))
        for fill_side in ("left", "right"):
            try:
                offset_dist = 3.0 if fill_side == "left" else -3.0
                sub_geom = road_geom.offset_curve(offset_dist)
                if sub_geom.is_empty:
                    continue
            except Exception:
                continue
            intervals.append({
                "id_trc": gap_id,
                "side": fill_side,
                "start_dist": 0,
                "end_dist": road_geom.length,
                "length_m": road_geom.length,
                "category": "no_data",
                "street_name": street_name,
                "descriptions": "Short gap segment — no sign data",
                "geometry": sub_geom,
            })

    print(f"  Reconstructed {len(intervals)} intervals")

    if not intervals:
        return gpd.GeoDataFrame(
            columns=[
                "id_trc", "side", "start_dist", "end_dist", "length_m",
                "category", "street_name", "descriptions", "geometry",
            ],
            crs=CRS_MTM8,
        )

    return gpd.GeoDataFrame(intervals, crs=CRS_MTM8)
