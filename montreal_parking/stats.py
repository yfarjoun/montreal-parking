"""Summary statistics for parking intervals."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from montreal_parking.constants import (
    COLOR_FREE,
    COLOR_NO_DATA,
    COLOR_PAID,
    COLOR_RESTRICTED,
    COLOR_TIME_LIMITED,
)

_CAT_COLORS: dict[str, str] = {
    "free": COLOR_FREE,
    "time_limited": COLOR_TIME_LIMITED,
    "paid": COLOR_PAID,
    "restricted": COLOR_RESTRICTED,
    "no_data": COLOR_NO_DATA,
}

_CAT_LABELS: dict[str, str] = {
    "free": "Free",
    "time_limited": "Time-Limited",
    "paid": "Paid",
    "restricted": "Restricted",
    "no_data": "No Data",
}

_CAT_ORDER = ["free", "time_limited", "paid", "restricted", "no_data"]


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


def _category_breakdown(intervals: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute per-category breakdown sorted in display order."""
    total_km = intervals["length_m"].sum() / 1000
    rows: list[dict[str, Any]] = []
    for cat in _CAT_ORDER:
        subset = intervals[intervals["category"] == cat]
        if subset.empty:
            continue
        km = subset["length_m"].sum() / 1000
        pct = 100 * km / total_km if total_km > 0 else 0
        rows.append({
            "category": cat,
            "label": _CAT_LABELS.get(cat, cat),
            "color": _CAT_COLORS.get(cat, "#999"),
            "km": km,
            "pct": pct,
        })
    return rows


def _breakdown_table_html(rows: list[dict[str, Any]], total_km: float) -> str:
    """Generate an HTML table for a category breakdown."""
    lines = [
        "<table>",
        "<tr><th>Category</th><th>Length</th><th>%</th><th></th></tr>",
    ]
    for r in rows:
        bar_width = max(1, int(r["pct"] * 2))
        lines.append(
            f"<tr>"
            f"<td><span class='swatch' style='background:{r['color']}'></span>"
            f"{html.escape(r['label'])}</td>"
            f"<td class='num'>{r['km']:.1f} km</td>"
            f"<td class='num'>{r['pct']:.1f}%</td>"
            f"<td><div class='bar' style='width:{bar_width}px;"
            f"background:{r['color']}'></div></td>"
            f"</tr>"
        )
    lines.append(
        f"<tr class='total'><td><b>Total</b></td>"
        f"<td class='num'><b>{total_km:.1f} km</b></td>"
        f"<td></td><td></td></tr>"
    )
    lines.append("</table>")
    return "\n".join(lines)


def generate_stats_html(
    intervals_gdf: gpd.GeoDataFrame,
    snapped_signs: pd.DataFrame,
    dest: Path,
) -> None:
    """Generate a stats HTML page with city-wide and per-borough breakdowns."""
    intervals = pd.DataFrame(intervals_gdf)

    # Map ID_TRC -> borough from snapped signs (most common borough per road)
    trc_borough = (
        snapped_signs.groupby("ID_TRC")["NOM_ARROND"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    )
    intervals["borough"] = intervals["id_trc"].map(trc_borough).fillna("Unknown")

    # City-wide stats
    total_km = intervals["length_m"].sum() / 1000
    city_rows = _category_breakdown(intervals)
    city_table = _breakdown_table_html(city_rows, total_km)

    # Per-borough stats
    borough_sections: list[str] = []
    borough_names = sorted(intervals["borough"].unique())
    for borough in borough_names:
        if borough == "Unknown":
            continue
        b_intervals = intervals[intervals["borough"] == borough]
        b_total_km = b_intervals["length_m"].sum() / 1000
        b_rows = _category_breakdown(b_intervals)
        if not b_rows:
            continue
        b_table = _breakdown_table_html(b_rows, b_total_km)
        borough_sections.append(
            f"<details><summary><b>{html.escape(borough)}</b>"
            f" — {b_total_km:.1f} km</summary>\n{b_table}\n</details>"
        )

    boroughs_html = "\n".join(borough_sections)

    page = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Montreal Parking Statistics</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
      color: #333;
      line-height: 1.5;
    }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #777; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
    th, td {{ text-align: left; padding: 4px 12px 4px 0; }}
    th {{ border-bottom: 2px solid #ddd; font-size: 13px; color: #777; }}
    td {{ border-bottom: 1px solid #eee; }}
    .total td {{ border-top: 2px solid #ddd; border-bottom: none; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .swatch {{
      display: inline-block; width: 14px; height: 14px;
      border-radius: 3px; margin-right: 6px; vertical-align: middle;
    }}
    .bar {{ height: 14px; border-radius: 2px; display: inline-block; }}
    details {{ margin: 4px 0; }}
    summary {{
      cursor: pointer; padding: 6px 0;
      border-bottom: 1px solid #eee;
    }}
    summary:hover {{ background: #f8f8f8; }}
    a {{ color: #3498db; }}
    .disclaimer {{ color: #999; font-size: 12px; margin-top: 32px; }}
  </style>
</head>
<body>
  <h1>Montreal Parking Statistics</h1>
  <p class="subtitle">
    <a href="index.html">&larr; Back to map</a>
  </p>

  <h2>City-wide</h2>
  {city_table}

  <h2>By Borough</h2>
  {boroughs_html}

  <p class="disclaimer">
    Hobby project &mdash; not official. Data from
    <a href="https://donnees.montreal.ca/">Montreal Open Data</a>.
    <a href="https://github.com/yfarjoun/montreal-parking">GitHub</a>
  </p>
  <script data-goatcounter="https://yfarjoun.goatcounter.com/count"
          async src="//gc.zgo.at/count.js"></script>
</body>
</html>"""

    with open(dest, "w") as f:
        f.write(page)
    print(f"    Stats page: {dest} ({len(page) / 1e3:.0f} KB)")
