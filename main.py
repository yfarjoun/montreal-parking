"""Montreal Free Parking Finder — pipeline entry point."""

from __future__ import annotations

import argparse

from montreal_parking.classify import classify_all_signs
from montreal_parking.constants import DATA_DIR, FILENAMES, OUTPUT_DIR
from montreal_parking.data import download_data, load_geobase, load_signage
from montreal_parking.intervals import reconstruct_intervals
from montreal_parking.map import build_map
from montreal_parking.snap import snap_poles_to_roads
from montreal_parking.stats import generate_stats_html, print_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Montreal Free Parking Finder")
    parser.add_argument(
        "--borough",
        type=str,
        default=None,
        help="Filter to a specific borough (substring match on NOM_ARROND). "
        "Default: all Montreal.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Include debug layers (e.g. DEUX COTES copies) in the map.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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

    if args.borough:
        print(f"\n  Filtering to borough matching '{args.borough}'...")
        signs_df = signs_df[
            signs_df["NOM_ARROND"].str.contains(args.borough, case=False, na=False)
        ]
        print(f"  {len(signs_df)} signs after filter")

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
    build_map(intervals, snapped, unsnapped, roads_gdf=roads_gdf, borough=args.borough, debug=args.debug)
    generate_stats_html(intervals, snapped, OUTPUT_DIR / "stats.html")
    print(f"  Map saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
