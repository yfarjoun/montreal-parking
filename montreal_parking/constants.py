"""Shared constants for the Montreal parking pipeline."""

from __future__ import annotations

import enum
from pathlib import Path


class SignCategory(enum.StrEnum):
    """Categories assigned to individual parking signs."""

    NO_PARKING = "no_parking"
    PERMIT = "permit"
    PAID = "paid"
    TIME_LIMITED = "time_limited"
    STREET_CLEANING = "street_cleaning"
    UNRESTRICTED = "unrestricted"
    PANONCEAU = "panonceau"
    OTHER = "other"


class IntervalCategory(enum.StrEnum):
    """Categories assigned to reconstructed parking intervals on the map."""

    FREE = "free"
    TIME_LIMITED = "time_limited"
    PAID = "paid"
    RESTRICTED = "restricted"
    NO_DATA = "no_data"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

CRS_WGS84 = "EPSG:4326"
CRS_MTM8 = "EPSG:32188"  # NAD83 MTM zone 8, metric CRS for Montreal

MAX_SNAP_DISTANCE_M = 20.0

# Edge intervals shorter than this are too close to an intersection to be free parking
MIN_FREE_EDGE_M = 5.0
# Buffer beyond crosswalk where parking is prohibited (QC Highway Safety Code)
CROSSWALK_NO_PARK_M = 5.0
# Max distance from road endpoint to search for crosswalks
CROSSWALK_SEARCH_RADIUS_M = 5.0

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
    "paid_places": "https://www.agencemobilitedurable.ca/images/data/Places.csv",
}

FILENAMES = {
    "signage": "signalisation_stationnement.csv",
    "rpa_codes": "signalisation-codification-rpa.csv",
    "geobase": "geobase.json",
    "paid_places": "Places.csv",
}

# Meter snapping: max gap between consecutive meters to merge into one span
METER_CLUSTER_GAP_M = 30.0
# Buffer added to each end of a meter cluster span (half a parking space)
METER_EDGE_BUFFER_M = 5.0

# Map colors
COLOR_FREE = "#2ecc71"  # green
COLOR_TIME_LIMITED = "#f1c40f"  # yellow
COLOR_RESTRICTED = "#e74c3c"  # red
COLOR_PAID = "#3498db"  # blue
COLOR_NO_DATA = "#9b59b6"  # purple

# Default map centre (Montreal city centre)
MAP_CENTER = [45.5017, -73.5673]
MAP_ZOOM_DEFAULT = 12
MAP_ZOOM_BOROUGH = 14

# Tile provider (CartoDB Positron — clean, free, no ToS issues)
TILES_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
TILES_ATTR = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/">CARTO</a>'
)

# Geometry simplification tolerance for GeoJSON export (≈5 m in WGS84 degrees)
SIMPLIFY_TOLERANCE = 0.00005
