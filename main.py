"""Montreal Free Parking Finder — pipeline entry point."""

from __future__ import annotations

from montreal_parking.classify import classify_all_signs
from montreal_parking.constants import DATA_DIR, FILENAMES, OUTPUT_DIR, PLATEAU_FILTER
from montreal_parking.data import download_data, load_geobase, load_signage
from montreal_parking.intervals import reconstruct_intervals
from montreal_parking.map import build_map
from montreal_parking.snap import snap_poles_to_roads
from montreal_parking.stats import print_stats


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
