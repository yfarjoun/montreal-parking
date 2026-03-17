# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Montreal Free Parking Finder — downloads Montreal open data (parking signage, RPA codes, road geobase), classifies signs, snaps sign poles to road segments, reconstructs free/restricted parking intervals per street side, and outputs a lightweight HTML map with external GeoJSON layers.

Supports all of Montreal (103k+ poles). Optionally filter to a single borough via `--borough`.

## Environment & Dependencies

Uses **pixi** (not uv/pip) for dependency management. Dependencies are declared in `pixi.toml`.

```bash
pixi install                    # install/sync runtime environment
pixi install -e dev             # install dev environment (includes test/lint tools)
pixi run build                  # run the full pipeline (all Montreal)
pixi run build-plateau          # run pipeline for Plateau only
pixi run python main.py --borough Rosemont  # any borough substring
```

## Dev Commands

```bash
pixi run -e dev test            # run pytest
pixi run -e dev lint            # run ruff check
pixi run -e dev format          # run ruff format
pixi run -e dev typecheck       # run mypy (strict mode)
```

Tool configuration (pytest, ruff, mypy) lives in `pyproject.toml`.

## Code Structure

`main.py` orchestrates the pipeline (with `argparse` for `--borough`). Logic lives in the `montreal_parking/` package:

| Module         | Responsibility                                                                                                                            |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `constants.py` | URLs, CRS strings, file paths, colors, thresholds, tile config                                                                            |
| `data.py`      | Download datasets and load signage CSV / geobase GeoJSON                                                                                  |
| `classify.py`  | Regex-based sign classification → `no_parking`, `permit`, `paid`, `time_limited`, `street_cleaning`, `unrestricted`, `panonceau`, `other` |
| `snap.py`      | Spatial join (sjoin_nearest) in metric CRS, vectorized projection distance + left/right side via cross product, DEUX COTES duplication    |
| `intervals.py` | Walk poles in chainage order per (road, side), arrow-based interval classification, gap filling                                           |
| `map.py`       | Export category GeoJSON files + lightweight HTML shell with Leaflet, GPS locate button                                                     |
| `stats.py`     | Summary statistics printing                                                                                                               |

## Key Design Decisions

- **Arrow convention**: arrow code 2 = left, 3 = right (as seen facing the sign). Forward/backward mapping flips based on which side of the road the pole is on.
- **"DEUX COTES" signs** are duplicated to the opposite side during snapping.
- **Offset curves** (±3m) visually separate left/right side intervals on the map.
- **Gap filling**: unsigned road sides get `no_data`; short (<60m) connector segments on streets that have signed blocks also get `no_data`.
- **Vectorized snapping** uses `shapely.line_locate_point` / `shapely.line_interpolate_point` + numpy cross product for city-wide performance.
- **External GeoJSON**: map output is a lightweight HTML shell (~50 KB) that fetches per-category GeoJSON via `fetch()`, keeping total payload manageable at city scale.
- **Tiles**: CartoDB Positron (free, no ToS issues).
- **GPS**: Leaflet LocateControl plugin for phone geolocation.

## Output Structure

```
output/
  index.html          # lightweight Leaflet shell
  data/
    free.geojson
    time_limited.geojson
    restricted.geojson
    no_data.geojson
    poles.geojson
    unmatched_poles.geojson
```

## CI/CD

- `.github/workflows/ci.yml` — lint + typecheck + test on push/PR
- `.github/workflows/deploy.yml` — weekly + manual: build full map and deploy to GitHub Pages

## Data Files (gitignored)

- `data/` — downloaded CSVs and GeoJSON (auto-created by `download_data()`)
- `output/` — generated HTML + GeoJSON files
