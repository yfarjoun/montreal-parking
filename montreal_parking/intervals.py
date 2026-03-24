"""Reconstruct parking intervals from snapped signs."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import substring

from montreal_parking.classify import sign_level
from montreal_parking.constants import (
    CRS_MTM8,
    METER_CLUSTER_GAP_M,
    METER_EDGE_BUFFER_M,
    MIN_FREE_EDGE_M,
    IntervalCategory,
    SignCategory,
)

# SignCategory members (PAID and TIME_LIMITED omitted — conflict with IntervalCategory)
NO_PARKING = SignCategory.NO_PARKING
PERMIT = SignCategory.PERMIT
UNRESTRICTED = SignCategory.UNRESTRICTED
STREET_CLEANING = SignCategory.STREET_CLEANING

# IntervalCategory members (PAID and TIME_LIMITED omitted — conflict with SignCategory)
FREE = IntervalCategory.FREE
RESTRICTED = IntervalCategory.RESTRICTED
NO_DATA = IntervalCategory.NO_DATA


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


def _classify_level_interval(
    active_signs: list[dict[str, Any]],
    level: int,
) -> str | None:
    """Classify an interval given its active signs at a specific level.

    Level 3: if any sign is restrictive -> "restricted", else None
    Level 4: "paid" > "time_limited" > "free" (unrestricted)
    Returns None if no signs contribute a classification.
    """
    if not active_signs:
        return None
    if level == 3:
        if any(s["category"] in (NO_PARKING, PERMIT) for s in active_signs):
            return RESTRICTED
        return None
    if level == 4:
        categories = {s["category"] for s in active_signs}
        if SignCategory.PAID in categories:
            return IntervalCategory.PAID
        if SignCategory.TIME_LIMITED in categories:
            return IntervalCategory.TIME_LIMITED
        if UNRESTRICTED in categories:
            return FREE
    return None


def _walk_level_spans(
    pole_data: pd.DataFrame,
    pole_signs: dict[Any, list[dict[str, Any]]],
    forward_arrow: int,
    backward_arrow: int,
    road_length: float,
    level: int,
) -> list[tuple[float, float, str]]:
    """Walk poles at a given level and produce classified spans.

    Returns list of (start, end, category) spans. Only poles that have at least
    one sign at this level are included in pole_data.
    """
    if pole_data.empty:
        return []

    spans: list[tuple[float, float, str]] = []

    def _get(pole_id: Any, direction: str) -> list[dict[str, Any]]:
        return _get_signs_for_direction(
            pole_id, direction, pole_signs, forward_arrow, backward_arrow,
        )

    # Edge interval before first pole
    first_pole = pole_data.iloc[0]
    first_id = first_pole["POTEAU_ID_POT"]
    first_dist = float(first_pole["projection_distance"])
    if first_dist > 1.0:
        cat = _classify_level_interval(_get(first_id, "backward"), level)
        if cat is not None:
            spans.append((0, first_dist, cat))

    # Intervals between consecutive poles
    for i in range(len(pole_data) - 1):
        p1 = pole_data.iloc[i]
        p2 = pole_data.iloc[i + 1]
        active = _get(p1["POTEAU_ID_POT"], "forward") + _get(p2["POTEAU_ID_POT"], "backward")
        cat = _classify_level_interval(active, level)
        if cat is not None:
            spans.append((
                float(p1["projection_distance"]),
                float(p2["projection_distance"]),
                cat,
            ))

    # Edge interval after last pole
    last_pole = pole_data.iloc[-1]
    last_id = last_pole["POTEAU_ID_POT"]
    last_dist = float(last_pole["projection_distance"])
    if road_length - last_dist > 1.0:
        cat = _classify_level_interval(_get(last_id, "forward"), level)
        if cat is not None:
            spans.append((last_dist, road_length, cat))

    return spans


def _merge_level_spans(
    level3_spans: list[tuple[float, float, str]],
    level4_spans: list[tuple[float, float, str]],
    road_length: float,
) -> list[tuple[float, float, str, list[str]]]:
    """Merge level 3 and level 4 spans. Level 4 overrides level 3; level 3 overrides default (free).

    Returns list of (start, end, category, descriptions) tuples covering [0, road_length].
    """
    # Collect all boundary points
    boundaries: set[float] = {0.0, road_length}
    for spans in (level3_spans, level4_spans):
        for start, end, _ in spans:
            boundaries.add(start)
            boundaries.add(end)
    sorted_bounds = sorted(boundaries)

    result: list[tuple[float, float, str, list[str]]] = []
    for i in range(len(sorted_bounds) - 1):
        seg_start = sorted_bounds[i]
        seg_end = sorted_bounds[i + 1]
        if seg_end - seg_start < 0.5:
            continue

        mid = (seg_start + seg_end) / 2

        # Find level 4 classification at midpoint (highest priority wins:
        # paid > time_limited > free, so overlapping meter spans beat sign-based free)
        _l4_priority: dict[str, int] = {
            IntervalCategory.PAID: 3, IntervalCategory.TIME_LIMITED: 2, FREE: 1,
        }
        l4_cat: str | None = None
        l4_rank = 0
        for start, end, cat in level4_spans:
            if start <= mid < end:
                rank = _l4_priority.get(cat, 0)
                if rank > l4_rank:
                    l4_cat = cat
                    l4_rank = rank

        # Find level 3 classification at midpoint
        l3_cat: str | None = None
        for start, end, cat in level3_spans:
            if start <= mid < end:
                l3_cat = cat
                break

        # Level 4 overrides level 3, level 3 overrides default (free)
        if l4_cat is not None:
            final_cat = l4_cat
        elif l3_cat is not None:
            final_cat = l3_cat
        else:
            final_cat = FREE

        result.append((seg_start, seg_end, final_cat, []))

    # Merge adjacent spans with the same category
    merged: list[tuple[float, float, str, list[str]]] = []
    for span in result:
        if merged and merged[-1][2] == span[2]:
            prev = merged[-1]
            merged[-1] = (prev[0], span[1], prev[2], prev[3])
        else:
            merged.append(span)

    return merged


def _make_interval(
    start_dist: float,
    end_dist: float,
    category: str,
    descriptions: list[Any],
    road_geom: Any,
    id_trc: Any,
    side: str,
    street_name: str,
    rate: float | None = None,
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
    interval: dict[str, Any] = {
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
    if rate is not None:
        interval["rate"] = rate
    return interval


def _build_meter_spans(
    meter_group: pd.DataFrame,
    road_length: float,
) -> tuple[list[tuple[float, float, str]], float | None]:
    """Cluster meter points into paid spans on one (road, side).

    Returns (spans, median_rate) where spans is a list of (start, end, "paid") tuples.
    Consecutive meters within METER_CLUSTER_GAP_M are merged into one span,
    with METER_EDGE_BUFFER_M added to each end.
    """
    if meter_group.empty:
        return [], None

    dists: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.sort(
        meter_group["projection_distance"].to_numpy()
    )
    rates = meter_group["rate"].dropna()
    median_rate = float(rates.median()) if not rates.empty else None

    # Cluster consecutive meters within gap threshold
    spans: list[tuple[float, float, str]] = []
    cluster_start = dists[0]
    cluster_end = dists[0]

    for d in dists[1:]:
        if d - cluster_end <= METER_CLUSTER_GAP_M:
            cluster_end = d
        else:
            # Emit current cluster
            s = max(0.0, cluster_start - METER_EDGE_BUFFER_M)
            e = min(road_length, cluster_end + METER_EDGE_BUFFER_M)
            spans.append((s, e, IntervalCategory.PAID))
            cluster_start = d
            cluster_end = d

    # Emit final cluster
    s = max(0.0, cluster_start - METER_EDGE_BUFFER_M)
    e = min(road_length, cluster_end + METER_EDGE_BUFFER_M)
    spans.append((s, e, IntervalCategory.PAID))

    return spans, median_rate


def _build_side_intervals(
    pole_data: pd.DataFrame,
    pole_signs: dict[Any, list[dict[str, Any]]],
    forward_arrow: int,
    backward_arrow: int,
    road_geom: Any,
    id_trc: Any,
    side: str,
    street_name: str,
    meter_group: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build intervals for one (road, side) group from its poles using level-based classification."""
    road_length = road_geom.length

    # Split pole_signs by level
    level_pole_signs: dict[int, dict[Any, list[dict[str, Any]]]] = {3: {}, 4: {}}
    for pid, signs in pole_signs.items():
        for sign in signs:
            lvl = sign_level(sign["category"])
            if lvl in level_pole_signs:
                if pid not in level_pole_signs[lvl]:
                    level_pole_signs[lvl][pid] = []
                level_pole_signs[lvl][pid].append(sign)

    # Build pole DataFrames for each level (poles that have at least one sign at that level)
    level_pole_data: dict[int, pd.DataFrame] = {}
    for lvl in (3, 4):
        level_pids = set(level_pole_signs[lvl].keys())
        if level_pids:
            mask = pole_data["POTEAU_ID_POT"].isin(level_pids)
            level_pole_data[lvl] = pole_data[mask].reset_index(drop=True)
        else:
            level_pole_data[lvl] = pole_data.iloc[0:0]  # empty with same columns

    # Walk each level independently
    level3_spans = _walk_level_spans(
        level_pole_data[3], level_pole_signs[3],
        forward_arrow, backward_arrow, road_length, level=3,
    )
    level4_spans = _walk_level_spans(
        level_pole_data[4], level_pole_signs[4],
        forward_arrow, backward_arrow, road_length, level=4,
    )

    # Inject meter-derived paid spans into level 4
    meter_rate: float | None = None
    if meter_group is not None and not meter_group.empty:
        meter_spans, meter_rate = _build_meter_spans(meter_group, road_length)
        level4_spans.extend(meter_spans)

    # Merge levels
    merged = _merge_level_spans(level3_spans, level4_spans, road_length)

    # Apply 5m edge rule and build geometry
    intervals: list[dict[str, Any]] = []
    for start, end, cat, descs in merged:
        # Short edges near intersections can't be free parking
        if cat == FREE and start == 0.0 and end < MIN_FREE_EDGE_M:
            cat = NO_DATA
        if cat == FREE and end == road_length and (road_length - start) < MIN_FREE_EDGE_M:
            cat = NO_DATA
        # Attach meter rate to paid intervals that came from meter data
        rate = meter_rate if cat == IntervalCategory.PAID else None
        iv = _make_interval(start, end, cat, descs, road_geom, id_trc, side, street_name, rate=rate)
        if iv:
            intervals.append(iv)

    return intervals


def reconstruct_intervals(
    snapped_signs: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    metered_places: pd.DataFrame | None = None,
) -> gpd.GeoDataFrame:
    """For each (ID_TRC, side) group, determine free/restricted intervals.

    Returns a GeoDataFrame with one row per interval, including geometry
    (sub-segment of the road), category, and metadata.

    If metered_places is provided (from snap_meters_to_roads), meter-derived
    paid spans are injected as level-4 overrides.
    """
    roads_mtm = roads_gdf.to_crs(CRS_MTM8)
    road_geoms = roads_mtm.set_index("ID_TRC")["geometry"]
    road_names = roads_mtm.set_index("ID_TRC")["NOM_VOIE"]

    intervals: list[dict[str, Any]] = []

    # Pre-group meter data by (ID_TRC, side)
    meter_groups: dict[tuple[Any, str], pd.DataFrame] = {}
    if metered_places is not None and not metered_places.empty:
        for (m_id_trc, m_side), m_group in metered_places.groupby(["ID_TRC", "side"]):
            meter_groups[(m_id_trc, str(m_side))] = m_group

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
        # Exclude street_cleaning from classification (kept in snapped data for popups)
        pole_signs: dict[Any, list[dict[str, Any]]] = {}
        for _, sign_row in group.iterrows():
            if sign_row["sign_category"] == STREET_CLEANING:
                continue
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
                meter_group=meter_groups.get((id_trc, side)),
            )
        )

    # --- Handle meter-only road sides (no sign data) ---
    covered_combos = set(grouped.groups.keys())
    for (m_id_trc, m_side), m_group in meter_groups.items():
        if (m_id_trc, m_side) in covered_combos:
            continue  # already handled above
        if m_id_trc not in road_geoms.index:
            continue
        road_geom = road_geoms.loc[m_id_trc]
        road_length = road_geom.length
        if road_length < 1.0:
            continue
        street_name = road_names.get(m_id_trc, "Unknown")
        meter_spans, median_rate = _build_meter_spans(m_group, road_length)
        for start, end, cat in meter_spans:
            iv = _make_interval(
                start, end, cat, [], road_geom, m_id_trc, m_side, street_name,
                rate=median_rate,
            )
            if iv:
                intervals.append(iv)
        covered_combos.add((m_id_trc, m_side))

    # --- Fill gaps ---
    covered_segments = snapped_signs["ID_TRC"].unique()
    # Include segments that only have meters
    if meter_groups:
        meter_segment_ids = {k[0] for k in meter_groups}
        covered_segments = np.union1d(covered_segments, list(meter_segment_ids))

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
                "category": NO_DATA,
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
                "category": NO_DATA,
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
