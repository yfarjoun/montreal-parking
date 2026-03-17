# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Montreal Free Parking Finder — downloads Montreal open data (parking signage, RPA codes, road geobase), classifies signs, snaps sign poles to road segments, reconstructs free/restricted parking intervals per street side, and outputs an interactive Folium HTML map.

Currently scoped to Le Plateau-Mont-Royal as an MVP.

## Environment & Dependencies

Uses **pixi** (not uv/pip) for dependency management. Dependencies are declared in `pixi.toml`.

```bash
pixi install                    # install/sync runtime environment
pixi install -e dev             # install dev environment (includes test/lint tools)
pixi run python main.py         # run the pipeline
```

## Dev Commands

```bash
pixi run -e dev test            # run pytest
pixi run -e dev lint            # run ruff check
pixi run -e dev format          # run ruff format
pixi run -e dev typecheck       # run mypy (strict mode)
```

Tool configuration (pytest, ruff, mypy) lives in `pyproject.toml`.

## How It Works (Pipeline)

`main.py` is a single-file pipeline with these steps:

1. **Download** — fetches three datasets from donnees.montreal.ca into `data/` (cached on disk)
2. **Load** — reads signage CSV and geobase GeoJSON
3. **Classify** — regex-based sign classification into: `no_parking`, `permit`, `paid`, `time_limited`, `street_cleaning`, `unrestricted`, `panonceau`, `other`
4. **Snap poles to roads** — spatial join (sjoin_nearest) in metric CRS (EPSG:32188), computes projection distance along road and left/right side via cross product
5. **Reconstruct intervals** — walks poles in chainage order per (road, side), uses arrow codes to determine which signs govern each interval, fills gaps for unsigned sides and short connector segments
6. **Build map** — Folium map with layers for free/time-limited/restricted/no-data intervals, plus pole markers with Street View links

## Key Design Decisions

- **Arrow convention**: arrow code 2 = left, 3 = right (as seen facing the sign). Forward/backward mapping flips based on which side of the road the pole is on.
- **"DEUX COTES" signs** are duplicated to the opposite side during snapping.
- **Offset curves** (±3m) visually separate left/right side intervals on the map.
- **Gap filling**: unsigned road sides get `no_data`; short (<60m) connector segments on streets that have signed blocks also get `no_data`.

## Data Files (gitignored)

- `data/` — downloaded CSVs and GeoJSON (auto-created by `download_data()`)
- `output/` — generated HTML maps