"""Snap sign poles to road segments."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from montreal_parking.constants import CRS_MTM8, CRS_WGS84, MAX_SNAP_DISTANCE_M


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
