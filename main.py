"""Montreal Free Parking Finder.

Downloads Montreal open data on parking signage, identifies street segments
where parking is free (no permit, no payment, no prohibition), and produces
an interactive Folium HTML map.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import pandas as pd
import requests
from shapely.ops import substring

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

CRS_WGS84 = "EPSG:4326"
CRS_MTM8 = "EPSG:32188"  # NAD83 MTM zone 8, metric CRS for Montreal

MAX_SNAP_DISTANCE_M = 20.0

URLS = {
    "signage": (
        "https://donnees.montreal.ca/dataset/"
        "8ac6dd33-b0d3-4eab-a334-5a6283eb7940/resource/"
        "7f1d4ae9-1a12-46d7-953e-6b9c18c78680/download/"
        "signalisation_stationnement.csv"
    ),
    "rpa_codes": (
        "https://donnees.montreal.ca/dataset/"
        "8ac6dd33-b0d3-4eab-a334-5a6283eb7940/resource/"
        "1baac760-4311-4b4f-8996-db93d348cc24/download/"
        "signalisation-codification-rpa.csv"
    ),
    "geobase": (
        "https://donnees.montreal.ca/dataset/"
        "984f7a68-ab34-4092-9204-4bdfcca767c5/resource/"
        "9d3d60d8-4e7f-493e-8d6a-dcd040319d8d/download/"
        "geobase.json"
    ),
}

FILENAMES = {
    "signage": "signalisation_stationnement.csv",
    "rpa_codes": "signalisation-codification-rpa.csv",
    "geobase": "geobase.json",
}

# Map colors
COLOR_FREE = "#2ecc71"       # green
COLOR_TIME_LIMITED = "#f1c40f"  # yellow
COLOR_RESTRICTED = "#e74c3c"   # red
COLOR_NO_DATA = "#9b59b6"     # purple

# Borough filter for MVP
PLATEAU_FILTER = "Plateau"

# Google Maps-style tiles
GOOGLE_TILES = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"

# ---------------------------------------------------------------------------
# Step 1: Data Download
# ---------------------------------------------------------------------------


def download_data() -> None:
    """Download all datasets to data/ directory, skipping if already cached."""
    DATA_DIR.mkdir(exist_ok=True)
    for key, url in URLS.items():
        dest = DATA_DIR / FILENAMES[key]
        if dest.exists():
            print(f"  [cached] {dest}")
            continue
        print(f"  Downloading {key} -> {dest} ...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 2: Load Data
# ---------------------------------------------------------------------------


def load_signage(path: Path) -> pd.DataFrame:
    """Load the parking signage CSV."""
    df = pd.read_csv(path, low_memory=False)
    # Drop rows without coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df


def load_geobase(path: Path) -> gpd.GeoDataFrame:
    """Load the geobase GeoJSON with road centerlines."""
    gdf = gpd.read_file(path)
    # Keep only LineString geometries (skip any nulls or points)
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    return gdf


# ---------------------------------------------------------------------------
# Step 3: Sign Classification
# ---------------------------------------------------------------------------


# Street cleaning: \P <time> <weekday> <seasonal range>
# e.g., "\P 08h-09h MAR. 1 AVRIL AU 1 DEC."
# Also matches: "\P 12h30-14h30 MARDI 1er AVRIL AU 1er DEC."
_CLEANING_RE = re.compile(
    r"^\\[PA]\s+\d{1,2}[hH]"  # \P or \A + time
    r".*"
    r"(LUN|MAR|MER|JEU|VEN|SAM|DIM|LUNDI|MARDI|MERCREDI|JEUDI|VENDREDI|SAMEDI|DIMANCHE)"
    r".*"
    r"\d+\s*(E[Rr])?\s*(AVRIL|MARS|MAI|JUIN|JUIL|AOUT|SEPT|OCT|NOV|DEC)",
    re.IGNORECASE,
)


def classify_sign(description: str) -> str:
    """Classify a sign description into a category.

    Returns one of: 'no_parking', 'permit', 'paid', 'time_limited',
    'street_cleaning', 'unrestricted', 'panonceau', 'other'.
    """
    if not isinstance(description, str):
        return "other"

    desc = description.upper().strip()

    # No stopping / no parking (starts with \P or \A)
    if desc.startswith("\\P") or desc.startswith("\\A"):
        # Permit sign
        if "S3R" in desc or "AUTOCOL" in desc or "VIGNETTE" in desc:
            return "permit"
        # Street cleaning: short time window + specific weekday + seasonal range
        if _CLEANING_RE.match(description.strip()):
            return "street_cleaning"
        return "no_parking"

    # Permit / sticker required
    if "S3R" in desc or "AUTOCOL" in desc or "VIGNETTE" in desc:
        return "permit"

    # Paid parking
    if "TARIF" in desc or "PARCOFLEX" in desc or "PARCOMETRE" in desc or "PARCOMÈTRE" in desc or "PAYANT" in desc:
        return "paid"

    # Time-limited parking (e.g., "P 60 MIN", "P 120 MIN 9H-17H")
    if desc.startswith("P ") and ("MIN" in desc or "H" in desc):
        return "time_limited"

    # Explicit unrestricted parking
    if desc.startswith("P ") or desc == "P":
        return "unrestricted"

    # PANONCEAU = sub-sign/plaque that modifies the sign above it.
    # Not a standalone restriction — the parent sign is classified separately.
    if desc.startswith("PANONCEAU"):
        return "panonceau"

    return "other"


def is_restrictive(category: str) -> bool:
    """Whether a sign category prevents free parking."""
    return category in ("no_parking", "permit", "paid")


def classify_all_signs(df: pd.DataFrame) -> pd.DataFrame:
    """Add sign_category and is_restrictive columns."""
    df = df.copy()
    df["sign_category"] = df["DESCRIPTION_RPA"].apply(classify_sign)
    df["is_restrictive"] = df["sign_category"].apply(is_restrictive)
    return df


# ---------------------------------------------------------------------------
# Step 4: Snap Poles to Roads
# ---------------------------------------------------------------------------


def snap_poles_to_roads(
    signs_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Snap each unique pole to its nearest road segment.

    Returns a tuple of:
    - DataFrame of snapped signs with added columns:
      ID_TRC, NOM_VOIE, projection_distance, side, snap_distance.
    - DataFrame of unsnapped poles (too far from any road).
    """
    # Get unique poles
    poles = signs_df.drop_duplicates(subset="POTEAU_ID_POT")[
        ["POTEAU_ID_POT", "Latitude", "Longitude"]
    ].copy()

    # Create GeoDataFrame of poles
    poles_gdf = gpd.GeoDataFrame(
        poles,
        geometry=gpd.points_from_xy(poles["Longitude"], poles["Latitude"]),
        crs=CRS_WGS84,
    )

    # Reproject to metric CRS
    poles_gdf = poles_gdf.to_crs(CRS_MTM8)
    roads_mtm = roads_gdf.to_crs(CRS_MTM8)

    print(f"  Snapping {len(poles_gdf)} poles to {len(roads_mtm)} road segments...")

    # Spatial join: snap each pole to nearest road
    joined = gpd.sjoin_nearest(
        poles_gdf,
        roads_mtm[["ID_TRC", "NOM_VOIE", "geometry"]],
        how="left",
        max_distance=MAX_SNAP_DISTANCE_M,
        distance_col="snap_distance",
    )

    # Separate unsnapped poles
    unsnapped_mask = joined["ID_TRC"].isna()
    n_unsnapped = unsnapped_mask.sum()
    if n_unsnapped > 0:
        print(f"  Warning: {n_unsnapped} poles didn't snap to any road (>{MAX_SNAP_DISTANCE_M}m away)")
    unsnapped_pole_ids = joined.loc[unsnapped_mask, "POTEAU_ID_POT"].values
    unsnapped_signs = signs_df[signs_df["POTEAU_ID_POT"].isin(unsnapped_pole_ids)]
    joined = joined.dropna(subset=["ID_TRC"])
    joined["ID_TRC"] = joined["ID_TRC"].astype(int)

    # Compute projection distance along road and side
    road_geoms = roads_mtm.set_index("ID_TRC")["geometry"]

    proj_distances = []
    sides = []
    for _, row in joined.iterrows():
        road_geom = road_geoms.loc[row["ID_TRC"]]
        pole_point = row.geometry
        proj_dist = road_geom.project(pole_point)
        proj_distances.append(proj_dist)

        # Determine side: interpolate point on road at projection distance,
        # compute cross product with road tangent to determine left/right
        road_point = road_geom.interpolate(proj_dist)
        # Get tangent direction at this point
        p1 = road_geom.interpolate(max(0, proj_dist - 1))
        p2 = road_geom.interpolate(min(road_geom.length, proj_dist + 1))
        # tangent vector
        tx, ty = p2.x - p1.x, p2.y - p1.y
        # vector from road to pole
        dx, dy = pole_point.x - road_point.x, pole_point.y - road_point.y
        # cross product: positive = left, negative = right
        cross = tx * dy - ty * dx
        sides.append("left" if cross >= 0 else "right")

    joined["projection_distance"] = proj_distances
    joined["side"] = sides

    # Merge back to all signs (a pole can have multiple signs)
    result = signs_df.merge(
        joined[["POTEAU_ID_POT", "ID_TRC", "NOM_VOIE", "projection_distance", "side", "snap_distance"]],
        on="POTEAU_ID_POT",
        how="inner",
    )

    # Duplicate "DEUX COTES" signs to the opposite side of the road
    deux_cotes = result[
        result["DESCRIPTION_RPA"].str.contains("DEUX C", case=False, na=False)
    ].copy()
    if not deux_cotes.empty:
        deux_cotes["side"] = deux_cotes["side"].map({"left": "right", "right": "left"})
        result = pd.concat([result, deux_cotes], ignore_index=True)
        print(f"  Duplicated {len(deux_cotes)} 'DEUX COTES' signs to opposite side")

    return result, unsnapped_signs


# ---------------------------------------------------------------------------
# Step 5: Reconstruct Street Intervals
# ---------------------------------------------------------------------------


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
    if road_geom.length - last_dist > 1.0:
        cat, descs = _classify_interval(_get(last_id, "forward"))
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

    intervals = []

    # Group signs by road segment and side
    grouped = snapped_signs.groupby(["ID_TRC", "side"])

    for (id_trc, side_key), group in grouped:
        if id_trc not in road_geoms.index:
            continue
        side = str(side_key)

        road_geom = road_geoms.loc[id_trc]
        street_name = road_names.get(id_trc, "Unknown")
        # Build per-pole sign data with arrow directions
        # Arrow convention: 2 = toward increasing chainage, 3 = toward decreasing, 0 = both
        pole_data = (
            group.groupby("POTEAU_ID_POT")
            .agg(projection_distance=("projection_distance", "first"))
            .sort_values("projection_distance")
            .reset_index()
        )

        # Build lookup: pole_id -> list of (category, description, arrow, is_restrictive)
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

        # --- Build intervals for this (road, side) group ---
        intervals.extend(
            _build_side_intervals(
                pole_data, pole_signs, forward_arrow, backward_arrow,
                road_geom, id_trc, side, street_name,
            )
        )

    # --- Fill gaps ---
    covered_combos = set(grouped.groups.keys())  # (ID_TRC, side) pairs with poles
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
    #    These are typically intersection connectors between two signed blocks.
    #    Build a spatial index of covered segment endpoints to find adjacent gaps.
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

    gdf = gpd.GeoDataFrame(intervals, crs=CRS_MTM8)
    return gdf


# ---------------------------------------------------------------------------
# Step 6: Build Map
# ---------------------------------------------------------------------------


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


def build_map(
    intervals_gdf: gpd.GeoDataFrame,
    signs_df: pd.DataFrame,
    unsnapped_signs: pd.DataFrame | None = None,
) -> folium.Map:
    """Create an interactive Folium map showing parking zones and poles."""
    # Reproject to WGS84 for mapping
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

        # Build popup text as a column for GeoJsonPopup
        cat_label = category  # bind for lambda
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
    poles_unique = signs_df.drop_duplicates(subset="POTEAU_ID_POT")
    # Collect all signs per pole for the popup
    pole_sign_map: dict[Any, str] = (
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

    for _, row in poles_unique.iterrows():
        pid = row["POTEAU_ID_POT"]
        lat, lon = row["Latitude"], row["Longitude"]
        sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        popup_text = (
            f"<b>Pole {pid}</b><br>"
            f"{pole_sign_map.get(pid, '')}<br>"
            f'<a href="{sv_url}" target="_blank">Street View</a>'
        )
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=3,
            color="#333",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=400),
        ).add_to(poles_fg)
    poles_fg.add_to(m)

    # Add unsnapped poles layer (orange markers, off by default)
    if unsnapped_signs is not None and not unsnapped_signs.empty:
        unsnapped_fg = folium.FeatureGroup(name="Unmatched Poles", show=False)
        unsnapped_unique = unsnapped_signs.drop_duplicates(subset="POTEAU_ID_POT")
        unsnapped_sign_map: dict[Any, str] = (
            unsnapped_signs.groupby("POTEAU_ID_POT")
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
        for _, row in unsnapped_unique.iterrows():
            pid = row["POTEAU_ID_POT"]
            lat, lon = row["Latitude"], row["Longitude"]
            sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
            popup_text = (
                f"<b>Unmatched Pole {pid}</b><br>"
                f"{unsnapped_sign_map.get(pid, '')}<br>"
                f'<a href="{sv_url}" target="_blank">Street View</a>'
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#e67e22",
                fill=True,
                fill_color="#e67e22",
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=400),
            ).add_to(unsnapped_fg)
        unsnapped_fg.add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# Step 7: Statistics
# ---------------------------------------------------------------------------


def print_stats(intervals_gdf: gpd.GeoDataFrame) -> None:
    """Print summary statistics about the intervals."""
    if intervals_gdf.empty:
        print("\nNo intervals found.")
        return

    print("\n=== Summary Statistics ===")
    total = len(intervals_gdf)
    total_length_km = intervals_gdf["length_m"].sum() / 1000

    print(f"Total intervals: {total}")
    print(f"Total curb-length: {total_length_km:.1f} km")

    print("\nBy category:")
    for cat, group in intervals_gdf.groupby("category"):
        count = len(group)
        length_km = group["length_m"].sum() / 1000
        pct = count / total * 100
        print(f"  {cat:15s}: {count:5d} intervals ({pct:5.1f}%), {length_km:.1f} km")

    if "street_name" in intervals_gdf.columns:
        free = intervals_gdf[intervals_gdf["category"] == "free"]
        if not free.empty:
            print("\nTop 10 streets with most free parking (by length):")
            top = (
                free.groupby("street_name")["length_m"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            for street, length in top.items():
                print(f"  {street}: {length:.0f}m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Step 1: Downloading data...")
    download_data()

    print("\nStep 2: Loading signage data...")
    signs_df = load_signage(DATA_DIR / FILENAMES["signage"])
    print(f"  Loaded {len(signs_df)} signs")

    print("\nStep 3: Classifying signs...")
    signs_df = classify_all_signs(signs_df)
    print("  Category distribution:")
    for cat, count in signs_df["sign_category"].value_counts().items():
        print(f"    {cat}: {count}")

    # Filter to Le Plateau for MVP
    print("\n  Filtering to Le Plateau-Mont-Royal...")
    signs_df = signs_df[
        signs_df["NOM_ARROND"].str.contains(PLATEAU_FILTER, case=False, na=False)
    ]
    print(f"  {len(signs_df)} signs in Le Plateau")

    print("\nStep 4: Loading geobase and snapping poles to roads...")
    roads_gdf = load_geobase(DATA_DIR / FILENAMES["geobase"])
    print(f"  Loaded {len(roads_gdf)} road segments")
    snapped, unsnapped = snap_poles_to_roads(signs_df, roads_gdf)
    print(f"  Snapped {snapped['POTEAU_ID_POT'].nunique()} poles")
    if not unsnapped.empty:
        print(f"  Unmatched: {unsnapped['POTEAU_ID_POT'].nunique()} poles")

    print("\nStep 5: Reconstructing street intervals...")
    intervals = reconstruct_intervals(snapped, roads_gdf)

    print_stats(intervals)

    print("\nStep 6: Building map...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    m = build_map(intervals, signs_df, unsnapped)
    output_path = OUTPUT_DIR / "montreal_free_parking.html"
    m.save(str(output_path))
    print(f"  Map saved to {output_path}")


if __name__ == "__main__":
    main()
