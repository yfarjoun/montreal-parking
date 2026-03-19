# Montreal Free Parking Finder

[![CI](https://github.com/yfarjoun/montreal-parking/actions/workflows/ci.yml/badge.svg)](https://github.com/yfarjoun/montreal-parking/actions/workflows/ci.yml)
[![Deploy](https://github.com/yfarjoun/montreal-parking/actions/workflows/deploy.yml/badge.svg)](https://github.com/yfarjoun/montreal-parking/actions/workflows/deploy.yml)
[![GitHub Pages](https://img.shields.io/badge/live_map-GitHub_Pages-blue)](https://yfarjoun.github.io/montreal-parking/)

Interactive map of free street parking in Montreal, built from the city's open data.

> **Disclaimer:** This is a hobby project and is not affiliated with or endorsed by the City of Montreal. Parking data may be incomplete or inaccurate. Always check posted signs before parking.

**[View the live map](https://yfarjoun.github.io/montreal-parking/)**

## How it works

1. **Downloads** parking signage, RPA codes, and road geobase from [Montreal Open Data](https://donnees.montreal.ca/)
2. **Classifies** each sign (no parking, permit, paid, time-limited, street cleaning, etc.)
3. **Snaps** sign poles to the nearest road segment and determines left/right side
4. **Reconstructs** free vs. restricted parking intervals per street side using arrow directions
5. **Outputs** a lightweight Leaflet map with per-category GeoJSON layers

## Features

- City-wide coverage (103k+ sign poles) or filter by borough
- GPS locate button for mobile use
- Address search bar (Nominatim geocoder)
- Drop pins and share locations via URL
- Toggleable layers: free, time-limited, restricted, no data, sign poles

## Quick start

Requires [pixi](https://pixi.sh/).

```bash
pixi run build                          # build map for all Montreal
pixi run build-plateau                  # build map for Le Plateau only
pixi run python main.py --borough Rosemont  # any borough substring
```

Serve locally:

```bash
python -m http.server 8000 --directory output
# open http://localhost:8000
```

## Development

```bash
pixi install -e dev       # install dev environment
pixi run -e dev test      # pytest
pixi run -e dev lint      # ruff check
pixi run -e dev typecheck # mypy (strict)
pixi run -e dev format    # ruff format
```

## Issues and bug reports

Found a bug or have a feature request? Please [open an issue](https://github.com/yfarjoun/montreal-parking/issues) on GitHub.

## License

MIT
