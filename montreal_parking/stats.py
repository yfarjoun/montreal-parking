"""Summary statistics for parking intervals."""

from __future__ import annotations

import geopandas as gpd


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
