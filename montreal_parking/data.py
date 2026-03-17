"""Data download and loading."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from montreal_parking.constants import DATA_DIR, FILENAMES, URLS


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


def load_signage(path: Path) -> pd.DataFrame:
    """Load the parking signage CSV."""
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df


def load_geobase(path: Path) -> gpd.GeoDataFrame:
    """Load the geobase GeoJSON with road centerlines."""
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    return gdf
