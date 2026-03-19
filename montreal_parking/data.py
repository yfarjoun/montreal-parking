"""Data download and loading."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from montreal_parking.constants import DATA_DIR, FILENAMES, URLS


def _last_modified_path(dest: Path) -> Path:
    """Path to the file storing the Last-Modified header for a cached download."""
    return dest.with_suffix(dest.suffix + ".last-modified")


def _is_stale(url: str, dest: Path) -> bool:
    """Check if a cached file is stale by comparing Last-Modified headers.

    Returns True if the file should be re-downloaded.
    """
    meta_path = _last_modified_path(dest)
    if not meta_path.exists():
        return True
    saved = meta_path.read_text().strip()
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        remote = resp.headers.get("Last-Modified", "")
        if not remote:
            return True
        return remote != saved
    except requests.RequestException:
        # If HEAD fails, don't block the build — use cached data
        return False


def _save_last_modified(url: str, dest: Path) -> None:
    """Fetch and save the Last-Modified header for a downloaded file."""
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        last_modified = resp.headers.get("Last-Modified", "")
        if last_modified:
            _last_modified_path(dest).write_text(last_modified)
    except requests.RequestException:
        pass


def _download_file(url: str, dest: Path) -> None:
    """Download a file from a URL."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def download_data() -> None:
    """Download all datasets to data/ directory, re-downloading if upstream changed."""
    DATA_DIR.mkdir(exist_ok=True)
    for key, url in URLS.items():
        dest = DATA_DIR / FILENAMES[key]
        if dest.exists():
            if not _is_stale(url, dest):
                print(f"  [up to date] {dest}")
                continue
            print(f"  [stale] Re-downloading {key} -> {dest} ...")
        else:
            print(f"  Downloading {key} -> {dest} ...")
        _download_file(url, dest)
        _save_last_modified(url, dest)
        print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")


def get_data_date() -> str:
    """Return the Last-Modified date of the signage data, or empty string if unknown."""
    meta = _last_modified_path(DATA_DIR / FILENAMES["signage"])
    if not meta.exists():
        return ""
    # Header format: "Thu, 19 Mar 2026 00:09:06 GMT" — extract the date portion
    raw = meta.read_text().strip()
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return raw


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
