"""Data download and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString

from montreal_parking.constants import CRS_MTM8, CRS_WGS84, DATA_DIR, FILENAMES, URLS


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


def _parse_overpass_crossings(data: dict[str, Any]) -> list[LineString]:
    """Extract crossing LineStrings from Overpass JSON response."""
    elements: list[dict[str, Any]] = data.get("elements", [])
    nodes: dict[Any, tuple[Any, Any]] = {
        e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"
    }
    ways = [e for e in elements if e["type"] == "way"]

    geoms: list[LineString] = []
    for w in ways:
        coords = [nodes[nid] for nid in w["nodes"] if nid in nodes]
        if len(coords) >= 2:
            geoms.append(LineString(coords))
    return geoms


_CACHE_MAX_AGE_DAYS = 30


def download_crossings(
    bbox: tuple[float, float, float, float],
    cache_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Download pedestrian crossings from OpenStreetMap via Overpass API.

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon) in WGS84.
        cache_path: If provided and fresh (< 30 days old), load from cache.

    Returns GeoDataFrame of crossing LineStrings in CRS_MTM8.
    Falls back to empty GeoDataFrame if the API is unavailable.
    """
    import json
    import math
    import time

    # Use cache if fresh enough
    if cache_path and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < _CACHE_MAX_AGE_DAYS:
            with open(cache_path) as f:
                data = json.load(f)
            print(f"  [cached, {age_days:.0f}d old] Loaded crossings from {cache_path}")
            geoms = _parse_overpass_crossings(data)
            if not geoms:
                return gpd.GeoDataFrame(geometry=[], crs=CRS_MTM8)
            gdf = gpd.GeoDataFrame(geometry=geoms, crs=CRS_WGS84).to_crs(CRS_MTM8)
            print(f"  {len(gdf)} crossings loaded")
            return gdf
        print(f"  [stale, {age_days:.0f}d old] Re-downloading crossings...")

    # Split large bboxes into tiles to avoid Overpass timeouts.
    lat_step = 0.05  # ~5.5 km per tile
    lon_step = 0.08  # ~6 km per tile

    min_lat, min_lon, max_lat, max_lon = bbox
    lat_tiles = max(1, math.ceil((max_lat - min_lat) / lat_step))
    lon_tiles = max(1, math.ceil((max_lon - min_lon) / lon_step))
    total_tiles = lat_tiles * lon_tiles
    print(f"  Querying Overpass API in {total_tiles} tile(s)...")

    all_elements: list[dict[str, object]] = []
    failed = 0
    for idx, (i, j) in enumerate(
        (i, j) for i in range(lat_tiles) for j in range(lon_tiles)
    ):
        if idx > 0:
            time.sleep(2)  # rate-limit: 2s between requests

        tile_bbox = (
            min_lat + i * lat_step,
            min_lon + j * lon_step,
            min(min_lat + (i + 1) * lat_step, max_lat),
            min(min_lon + (j + 1) * lon_step, max_lon),
        )
        query = (
            f'[out:json][timeout:180];'
            f'way["highway"="footway"]["footway"="crossing"]'
            f'({tile_bbox[0]},{tile_bbox[1]},{tile_bbox[2]},{tile_bbox[3]});'
            f'out body;>;out skel qt;'
        )
        try:
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=240,
            )
            if resp.status_code == 429:
                # Back off on rate limit
                print("  Rate limited, waiting 30s...")
                time.sleep(30)
                resp = requests.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": query},
                    timeout=240,
                )
            resp.raise_for_status()
            tile_data = resp.json()
            all_elements.extend(tile_data.get("elements", []))
        except requests.RequestException as e:
            failed += 1
            print(f"  Warning: Overpass tile {idx + 1}/{total_tiles} failed: {e}")

    if failed:
        print(f"  {failed}/{total_tiles} tiles failed")

    if not all_elements:
        print("  Warning: No crossings data available — skipping intersection trimming")
        return gpd.GeoDataFrame(geometry=[], crs=CRS_MTM8)

    data = {"elements": all_elements}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
    print("  Downloaded crossings from Overpass API")
    geoms = _parse_overpass_crossings(data)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_MTM8)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=CRS_WGS84).to_crs(CRS_MTM8)
    print(f"  {len(gdf)} crossings loaded")
    return gdf


def load_paid_places(path: Path) -> pd.DataFrame:
    """Load the paid parking places CSV from Agence de mobilité durable.

    Filters to on-street places (sLocalisation == 'S') with valid coordinates.
    Converts nTarifHoraire from cents to dollars.
    """
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    # Keep only on-street places
    df = df[df["sLocalisation"] == "S"].copy()
    df = df.dropna(subset=["nPositionCentreLongitude", "nPositionCentreLatitude"])
    # Convert rate from cents to dollars
    df["rate"] = df["nTarifHoraire"] / 100.0
    return df
