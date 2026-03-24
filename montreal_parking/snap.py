"""Snap sign poles to road segments."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from montreal_parking.constants import CRS_MTM8, CRS_WGS84, MAX_SNAP_DISTANCE_M


def _compute_projection_and_side(
    joined: gpd.GeoDataFrame,
    road_geoms: pd.Series[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized projection distance and side computation.

    Returns (projection_distances, sides) arrays aligned with ``joined``.
    """
    # Build aligned arrays of road geometries and pole points
    road_geom_array = np.array([road_geoms.loc[id_trc] for id_trc in joined["ID_TRC"]])
    pole_point_array = joined.geometry.values

    # Projection distances along road (shapely.line_locate_point = vectorized .project())
    proj_distances = shapely.line_locate_point(road_geom_array, pole_point_array)

    # Tangent vectors via interpolation at proj ± 1m
    road_lengths = shapely.length(road_geom_array)
    p1 = shapely.line_interpolate_point(road_geom_array, np.maximum(0, proj_distances - 1))
    p2 = shapely.line_interpolate_point(road_geom_array, np.minimum(road_lengths, proj_distances + 1))

    # Road points at projection
    road_points = shapely.line_interpolate_point(road_geom_array, proj_distances)

    # Tangent vector (tx, ty) and pole offset vector (dx, dy)
    tx = shapely.get_x(p2) - shapely.get_x(p1)
    ty = shapely.get_y(p2) - shapely.get_y(p1)
    dx = shapely.get_x(pole_point_array) - shapely.get_x(road_points)
    dy = shapely.get_y(pole_point_array) - shapely.get_y(road_points)

    # Cross product: positive = left, negative = right
    cross = tx * dy - ty * dx
    sides = np.where(cross >= 0, "left", "right")

    return proj_distances, sides


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

    # Compute projection distance along road and side (vectorized)
    road_geoms = roads_mtm.set_index("ID_TRC")["geometry"]
    if joined.empty:
        joined["projection_distance"] = pd.Series(dtype=float)
        joined["side"] = pd.Series(dtype=str)
    else:
        proj_distances, sides = _compute_projection_and_side(joined, road_geoms)
        joined["projection_distance"] = proj_distances
        joined["side"] = sides

    # Merge back to all signs (a pole can have multiple signs)
    result = signs_df.merge(
        joined[["POTEAU_ID_POT", "ID_TRC", "NOM_VOIE", "projection_distance", "side", "snap_distance"]],
        on="POTEAU_ID_POT",
        how="inner",
    )

    # Duplicate "DEUX COTES" signs to the opposite side of the road
    result["is_deux_cotes_copy"] = False
    deux_cotes = result[
        result["DESCRIPTION_RPA"].str.contains("DEUX C", case=False, na=False)
    ].copy()
    if not deux_cotes.empty:
        deux_cotes["side"] = deux_cotes["side"].map({"left": "right", "right": "left"})
        # Flip directional arrows: left(2)/right(3) are relative to the viewer
        # facing the sign, so they swap when crossing to the opposite side.
        arrow_flip = {2: 3, 3: 2, 0: 0}
        deux_cotes["FLECHE_PAN"] = deux_cotes["FLECHE_PAN"].map(arrow_flip).fillna(0).astype(int)
        deux_cotes["is_deux_cotes_copy"] = True
        result = pd.concat([result, deux_cotes], ignore_index=True)
        print(f"  Duplicated {len(deux_cotes)} 'DEUX COTES' signs to opposite side")

    return result, unsnapped_signs


def snap_meters_to_roads(
    meters_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Snap paid parking meter spaces to their nearest road segment.

    Returns a DataFrame with columns: ID_TRC, projection_distance, side, rate.
    """
    meters_gdf = gpd.GeoDataFrame(
        meters_df,
        geometry=gpd.points_from_xy(
            meters_df["nPositionCentreLongitude"],
            meters_df["nPositionCentreLatitude"],
        ),
        crs=CRS_WGS84,
    )
    meters_gdf = meters_gdf.to_crs(CRS_MTM8)
    roads_mtm = roads_gdf.to_crs(CRS_MTM8)

    print(f"  Snapping {len(meters_gdf)} meter spaces to {len(roads_mtm)} road segments...")

    joined = gpd.sjoin_nearest(
        meters_gdf,
        roads_mtm[["ID_TRC", "geometry"]],
        how="left",
        max_distance=MAX_SNAP_DISTANCE_M,
        distance_col="snap_distance",
    )

    # Drop meters that didn't snap
    unsnapped = joined["ID_TRC"].isna().sum()
    if unsnapped > 0:
        print(f"  Warning: {unsnapped} meter spaces didn't snap to any road")
    joined = joined.dropna(subset=["ID_TRC"])
    joined["ID_TRC"] = joined["ID_TRC"].astype(int)

    # Compute projection distance and side
    road_geoms = roads_mtm.set_index("ID_TRC")["geometry"]
    if joined.empty:
        return pd.DataFrame(columns=["ID_TRC", "projection_distance", "side", "rate"])

    proj_distances, sides = _compute_projection_and_side(joined, road_geoms)
    joined["projection_distance"] = proj_distances
    joined["side"] = sides

    result: pd.DataFrame = joined[["ID_TRC", "projection_distance", "side", "rate"]].copy()
    print(f"  Snapped {len(result)} meter spaces")
    return result
