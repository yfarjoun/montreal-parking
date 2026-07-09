# Street-Cleaning Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse Montreal street-cleaning sign text into structured schedules, show them in map popups, and add an opt-in overlay that dashes street sides due for cleaning within the next 24 hours.

**Architecture:** A new build-time parser (`cleaning.py`) turns French sign text into a structured `CleaningSchedule`. `intervals.py` walks cleaning signs with the existing arrow convention and attaches schedules to interval segments without altering their category. `map.py` renders schedules into popups, emits a self-contained `cleaning.geojson`, and adds a Leaflet overlay whose client-side JS computes the "next 24h" window in Montreal time and draws a black dashed line over qualifying segments.

**Tech Stack:** Python 3 (pandas, geopandas, shapely), Leaflet 1.9, pixi for env/tasks, pytest/ruff/mypy(strict).

## Global Constraints

- Dependency/tasks via **pixi** only. Tests: `pixi run -e dev test`. Lint: `pixi run -e dev lint`. Format: `pixi run -e dev format`. Types: `pixi run -e dev typecheck` (mypy **strict**).
- All new code must be **typed** (pass mypy strict), **tested**, and **lint-clean**.
- Weekday convention throughout the Python code: **0 = Monday … 6 = Sunday** (Python's `date.weekday()`).
- Schedule display text is **English**.
- Cleaning attaches as metadata; it must **never change an interval's `category`**.
- Cleaning **overlay** is a **black dashed line drawn on top** of the base color; base category color shows through the dash gaps.
- Overlay layer is **off by default**.
- "Next 24h" and seasonal checks are computed in the browser in **Montreal time** (`America/Toronto`).
- Unparseable schedules fall back to **raw French text** in popups and are **excluded** from the 24h overlay.

---

### Task 1: `cleaning.py` — parser and formatter

**Files:**
- Create: `montreal_parking/cleaning.py`
- Test: `tests/test_cleaning.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `class CleaningSchedule(TypedDict)` with keys `weekdays: list[int]`, `start_min: int`, `end_min: int`, `month_start: tuple[int, int]`, `month_end: tuple[int, int]`.
  - `parse_cleaning(description: str) -> CleaningSchedule | None`
  - `format_schedule(sched: CleaningSchedule) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cleaning.py`:

```python
"""Tests for parsing/formatting Montreal street-cleaning sign text."""

from __future__ import annotations

from montreal_parking.cleaning import CleaningSchedule, format_schedule, parse_cleaning


class TestParseCleaning:
    def test_single_weekday(self) -> None:
        sched = parse_cleaning("\\P 08h-09h MAR. 1 AVRIL AU 1 DEC.")
        assert sched is not None
        assert sched["weekdays"] == [1]
        assert sched["start_min"] == 8 * 60
        assert sched["end_min"] == 9 * 60
        assert sched["month_start"] == (4, 1)
        assert sched["month_end"] == (12, 1)

    def test_time_with_minutes(self) -> None:
        sched = parse_cleaning("\\P 12h30-14h30 MARDI 1er AVRIL AU 1er DEC.")
        assert sched is not None
        assert sched["start_min"] == 12 * 60 + 30
        assert sched["end_min"] == 14 * 60 + 30
        assert sched["weekdays"] == [1]

    def test_multiple_weekdays_et(self) -> None:
        sched = parse_cleaning("\\P 8h-9h MAR. ET JEU. 1 AVRIL AU 1 NOV.")
        assert sched is not None
        assert sched["weekdays"] == [1, 3]

    def test_weekday_range_au(self) -> None:
        sched = parse_cleaning("\\P 8h-9h LUN. AU VEN. 1 AVRIL AU 1 DEC.")
        assert sched is not None
        assert sched["weekdays"] == [0, 1, 2, 3, 4]

    def test_biweekly_is_unparseable(self) -> None:
        assert parse_cleaning("\\P 8h-9h 1ER ET 3E MER. 1 AVRIL AU 1 DEC.") is None

    def test_no_month_range_is_unparseable(self) -> None:
        assert parse_cleaning("\\P 8h-9h MARDI") is None

    def test_non_string_is_none(self) -> None:
        assert parse_cleaning(None) is None  # type: ignore[arg-type]


class TestFormatSchedule:
    def test_single_day(self) -> None:
        sched = CleaningSchedule(
            weekdays=[1], start_min=480, end_min=540,
            month_start=(4, 1), month_end=(12, 1),
        )
        assert format_schedule(sched) == "Cleaning: Tue 8:00–9:00 · Apr 1 – Dec 1"

    def test_two_days_ampersand(self) -> None:
        sched = CleaningSchedule(
            weekdays=[1, 3], start_min=480, end_min=540,
            month_start=(4, 1), month_end=(11, 1),
        )
        assert format_schedule(sched) == "Cleaning: Tue & Thu 8:00–9:00 · Apr 1 – Nov 1"

    def test_consecutive_range(self) -> None:
        sched = CleaningSchedule(
            weekdays=[0, 1, 2, 3, 4], start_min=750, end_min=870,
            month_start=(4, 1), month_end=(12, 1),
        )
        assert format_schedule(sched) == "Cleaning: Mon–Fri 12:30–14:30 · Apr 1 – Dec 1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_cleaning.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'montreal_parking.cleaning'`

- [ ] **Step 3: Write `montreal_parking/cleaning.py`**

```python
"""Parse Montreal street-cleaning sign text into structured schedules."""

from __future__ import annotations

import re
from typing import TypedDict


class CleaningSchedule(TypedDict):
    """Structured street-cleaning schedule (all times in Montreal local time)."""

    weekdays: list[int]           # 0=Mon … 6=Sun
    start_min: int                # minutes since midnight
    end_min: int
    month_start: tuple[int, int]  # (month, day)
    month_end: tuple[int, int]    # (month, day)


_MONTHS: dict[str, int] = {
    "JANVIER": 1, "JANV": 1,
    "FEVRIER": 2, "FEVR": 2, "FÉVRIER": 2,
    "MARS": 3,
    "AVRIL": 4,
    "MAI": 5,
    "JUIN": 6,
    "JUILLET": 7, "JUIL": 7,
    "AOUT": 8, "AOÛT": 8,
    "SEPTEMBRE": 9, "SEPT": 9,
    "OCTOBRE": 10, "OCT": 10,
    "NOVEMBRE": 11, "NOV": 11,
    "DECEMBRE": 12, "DEC": 12, "DÉCEMBRE": 12,
}

_WEEKDAYS: dict[str, int] = {
    "LUNDI": 0, "LUN": 0,
    "MARDI": 1, "MAR": 1,
    "MERCREDI": 2, "MER": 2,
    "JEUDI": 3, "JEU": 3,
    "VENDREDI": 4, "VEN": 4,
    "SAMEDI": 5, "SAM": 5,
    "DIMANCHE": 6, "DIM": 6,
}

# Longest-first so full names match before their abbreviations.
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_ALT = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_MONTH_RANGE_RE = re.compile(
    r"(\d{1,2})\s*(?:ER|RE|E)?\s*(" + _MONTH_ALT + r")"
    r"\s+AU\s+"
    r"(\d{1,2})\s*(?:ER|RE|E)?\s*(" + _MONTH_ALT + r")"
)
_TIME_RE = re.compile(
    r"(\d{1,2})\s*H\s*(\d{2})?\s*[-–À]\s*(\d{1,2})\s*H\s*(\d{2})?"
)
_WEEKDAY_TOKEN_RE = re.compile(r"\b(" + _WEEKDAY_ALT + r")\b")

_EN_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_EN_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_cleaning(description: str) -> CleaningSchedule | None:
    """Parse a street-cleaning sign description into a structured schedule.

    Returns None for descriptions we cannot confidently parse (e.g. biweekly /
    nth-weekday patterns, or missing time/weekday/month-range components).
    """
    if not isinstance(description, str):
        return None
    text = description.upper()

    # 1) Seasonal month range (required).
    mr = _MONTH_RANGE_RE.search(text)
    if not mr:
        return None
    month_start = (_MONTHS[mr.group(2)], int(mr.group(1)))
    month_end = (_MONTHS[mr.group(4)], int(mr.group(3)))
    remainder = text[: mr.start()] + " " + text[mr.end():]

    # 2) Time window (required).
    tm = _TIME_RE.search(remainder)
    if not tm:
        return None
    start_min = int(tm.group(1)) * 60 + int(tm.group(2) or 0)
    end_min = int(tm.group(3)) * 60 + int(tm.group(4) or 0)
    remainder = remainder[: tm.start()] + " " + remainder[tm.end():]

    # 3) Leftover digits imply nth-weekday / biweekly / other complexity.
    if re.search(r"\d", remainder):
        return None

    # 4) Weekdays (required).
    tokens = _WEEKDAY_TOKEN_RE.findall(remainder)
    if not tokens:
        return None
    indices = [_WEEKDAYS[tok] for tok in tokens]
    if "AU" in remainder and len(indices) == 2:
        lo, hi = indices[0], indices[1]
        weekdays = (
            list(range(lo, hi + 1))
            if lo <= hi
            else list(range(lo, 7)) + list(range(0, hi + 1))
        )
    else:
        weekdays = sorted(set(indices))

    return CleaningSchedule(
        weekdays=weekdays,
        start_min=start_min,
        end_min=end_min,
        month_start=month_start,
        month_end=month_end,
    )


def _fmt_time(minutes: int) -> str:
    hours, mins = divmod(minutes, 60)
    return f"{hours}:{mins:02d}"


def _fmt_days(weekdays: list[int]) -> str:
    days = sorted(set(weekdays))
    if not days:
        return ""
    if len(days) >= 3 and days == list(range(days[0], days[-1] + 1)):
        return f"{_EN_DAYS[days[0]]}–{_EN_DAYS[days[-1]]}"
    return " & ".join(_EN_DAYS[d] for d in days)


def format_schedule(sched: CleaningSchedule) -> str:
    """Render a schedule as English popup text (e.g. 'Cleaning: Tue 8:00–9:00 · Apr 1 – Dec 1')."""
    days = _fmt_days(sched["weekdays"])
    window = f"{_fmt_time(sched['start_min'])}–{_fmt_time(sched['end_min'])}"
    m1 = f"{_EN_MONTHS[sched['month_start'][0]]} {sched['month_start'][1]}"
    m2 = f"{_EN_MONTHS[sched['month_end'][0]]} {sched['month_end'][1]}"
    return f"Cleaning: {days} {window} · {m1} – {m2}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_cleaning.py`
Expected: PASS (all tests)

- [ ] **Step 5: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add montreal_parking/cleaning.py tests/test_cleaning.py
git commit -m "feat: parse and format Montreal street-cleaning schedules"
```

---

### Task 2: `intervals.py` — attach cleaning schedules to interval segments

**Files:**
- Modify: `montreal_parking/intervals.py`
- Test: `tests/test_intervals.py` (add a `TestCleaningAttachment` class)

**Interfaces:**
- Consumes: `parse_cleaning`, `format_schedule` from Task 1.
- Produces: every interval row now carries two extra fields:
  - `cleaning: list[CleaningSchedule]` — parsed schedules active on the segment (empty if none).
  - `cleaning_text: str` — `"; "`-joined display text (formatted where parsed, raw French otherwise; empty if none).
  - New internal helpers `_walk_cleaning_spans(...)` and `_cleaning_key(...)`; `_merge_level_spans` and `_build_side_intervals` and `_make_interval` gain cleaning parameters (signatures below).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_intervals.py`:

```python
class TestCleaningAttachment:
    """Cleaning schedules attach to intervals without changing category."""

    def test_cleaning_attaches_and_keeps_free(self) -> None:
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P 08h-09h MAR. 1 AVRIL AU 1 DEC.",
            "FLECHE_PAN": 0,
            "sign_category": STREET_CLEANING,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        assert all(right["category"] == FREE)
        # The whole side carries a parsed schedule.
        assert right["cleaning_text"].str.contains("Cleaning: Tue").any()
        with_sched = right[right["cleaning"].apply(len) > 0]
        assert not with_sched.empty

    def test_directional_cleaning_splits_side(self) -> None:
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "\\P 08h-09h MAR. 1 AVRIL AU 1 DEC.",
            "FLECHE_PAN": 2,  # right side: 2=forward → applies 50–100 only
            "sign_category": STREET_CLEANING,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        cleaned = right[right["cleaning"].apply(len) > 0]
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["start_dist"] >= 49.0
        # The near-intersection half has no cleaning.
        uncleaned = right[right["cleaning"].apply(len) == 0]
        assert not uncleaned.empty

    def test_unparseable_cleaning_falls_back_to_raw(self) -> None:
        roads = _make_road(length=100)
        raw = "\\P 08h-09h 1ER ET 3E MER. 1 AVRIL AU 1 DEC."
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": raw,
            "FLECHE_PAN": 0,
            "sign_category": STREET_CLEANING,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        right = intervals[intervals["side"] == "right"]
        # Raw text shows in cleaning_text, but no structured schedule.
        assert right["cleaning_text"].str.contains("1ER ET 3E").any()
        assert all(right["cleaning"].apply(len) == 0)

    def test_non_cleaning_intervals_have_empty_cleaning(self) -> None:
        roads = _make_road(length=100)
        signs = _make_snapped_signs([{
            "POTEAU_ID_POT": 1,
            "projection_distance": 50.0,
            "side": "right",
            "ID_TRC": 1,
            "DESCRIPTION_RPA": "P",
            "FLECHE_PAN": 0,
            "sign_category": UNRESTRICTED,
            "is_restrictive": False,
            "NOM_VOIE": "Rue Test",
        }])
        intervals = reconstruct_intervals(signs, roads)
        assert "cleaning" in intervals.columns
        assert "cleaning_text" in intervals.columns
        assert all(intervals["cleaning"].apply(len) == 0)
        assert all(intervals["cleaning_text"] == "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_intervals.py::TestCleaningAttachment`
Expected: FAIL (e.g. `KeyError: 'cleaning'` / `AttributeError`).

- [ ] **Step 3: Add imports and the cleaning walk helpers**

In `montreal_parking/intervals.py`, extend the imports near the top:

```python
from montreal_parking.classify import sign_level
from montreal_parking.cleaning import CleaningSchedule, format_schedule, parse_cleaning
```

Add these two helpers (place them just above `_merge_level_spans`):

```python
def _walk_cleaning_spans(
    pole_data: pd.DataFrame,
    pole_cleaning: dict[Any, list[dict[str, Any]]],
    forward_arrow: int,
    backward_arrow: int,
    road_length: float,
) -> list[tuple[float, float, list[dict[str, Any]]]]:
    """Walk poles and produce spans annotated with active cleaning items.

    Each span is (start, end, items); items is a list of cleaning dicts
    ({"schedule": CleaningSchedule | None, "text": str}) active over that span.
    Spans with no active cleaning are omitted. Mirrors _walk_level_spans and
    uses the same forward/backward arrow convention.
    """
    if pole_data.empty:
        return []

    spans: list[tuple[float, float, list[dict[str, Any]]]] = []

    def _get(pole_id: Any, direction: str) -> list[dict[str, Any]]:
        arrow = forward_arrow if direction == "forward" else backward_arrow
        return [s for s in pole_cleaning.get(pole_id, []) if s["arrow"] in (arrow, 0)]

    first = pole_data.iloc[0]
    first_dist = float(first["projection_distance"])
    if first_dist > 1.0:
        items = _get(first["POTEAU_ID_POT"], "backward")
        if items:
            spans.append((0.0, first_dist, items))

    for i in range(len(pole_data) - 1):
        p1 = pole_data.iloc[i]
        p2 = pole_data.iloc[i + 1]
        items = _get(p1["POTEAU_ID_POT"], "forward") + _get(p2["POTEAU_ID_POT"], "backward")
        if items:
            spans.append((
                float(p1["projection_distance"]),
                float(p2["projection_distance"]),
                items,
            ))

    last = pole_data.iloc[-1]
    last_dist = float(last["projection_distance"])
    if road_length - last_dist > 1.0:
        items = _get(last["POTEAU_ID_POT"], "forward")
        if items:
            spans.append((last_dist, road_length, items))

    return spans


def _cleaning_key(items: list[dict[str, Any]]) -> frozenset[str]:
    """Stable signature of a span's cleaning items, by display text."""
    return frozenset(item["text"] for item in items)
```

- [ ] **Step 4: Extend `_merge_level_spans` to carry cleaning**

Replace the existing `_merge_level_spans` function body with the version below (new third parameter `cleaning_spans`; result tuples gain a 5th element — the active cleaning items):

```python
def _merge_level_spans(
    level3_spans: list[tuple[float, float, str]],
    level4_spans: list[tuple[float, float, str]],
    cleaning_spans: list[tuple[float, float, list[dict[str, Any]]]],
    road_length: float,
) -> list[tuple[float, float, str, list[str], list[dict[str, Any]]]]:
    """Merge level-3/4 spans and overlay cleaning metadata.

    Level 4 overrides level 3; level 3 overrides default (free). Cleaning spans
    add boundaries and attach items but never change the category. Returns
    (start, end, category, descriptions, cleaning_items) covering [0, road_length].
    """
    boundaries: set[float] = {0.0, road_length}
    for spans in (level3_spans, level4_spans):
        for start, end, _ in spans:
            boundaries.add(start)
            boundaries.add(end)
    for start, end, _items in cleaning_spans:
        boundaries.add(start)
        boundaries.add(end)
    sorted_bounds = sorted(boundaries)

    result: list[tuple[float, float, str, list[str], list[dict[str, Any]]]] = []
    for i in range(len(sorted_bounds) - 1):
        seg_start = sorted_bounds[i]
        seg_end = sorted_bounds[i + 1]
        if seg_end - seg_start < 0.5:
            continue

        mid = (seg_start + seg_end) / 2

        _l4_priority: dict[str, int] = {
            IntervalCategory.PAID: 3, IntervalCategory.TIME_LIMITED: 2, FREE: 1,
        }
        l4_cat: str | None = None
        l4_rank = 0
        for start, end, cat in level4_spans:
            if start <= mid < end:
                rank = _l4_priority.get(cat, 0)
                if rank > l4_rank:
                    l4_cat = cat
                    l4_rank = rank

        l3_cat: str | None = None
        for start, end, cat in level3_spans:
            if start <= mid < end:
                l3_cat = cat
                break

        if l4_cat is not None:
            final_cat = l4_cat
        elif l3_cat is not None:
            final_cat = l3_cat
        else:
            final_cat = FREE

        cleaning_items: list[dict[str, Any]] = []
        for start, end, items in cleaning_spans:
            if start <= mid < end:
                cleaning_items.extend(items)

        result.append((seg_start, seg_end, final_cat, [], cleaning_items))

    # Merge adjacent spans with the same category AND same cleaning signature.
    merged: list[tuple[float, float, str, list[str], list[dict[str, Any]]]] = []
    for span in result:
        if (
            merged
            and merged[-1][2] == span[2]
            and _cleaning_key(merged[-1][4]) == _cleaning_key(span[4])
        ):
            prev = merged[-1]
            merged[-1] = (prev[0], span[1], prev[2], prev[3], prev[4])
        else:
            merged.append(span)

    return merged
```

- [ ] **Step 5: Extend `_make_interval` to store cleaning fields**

In `_make_interval`, change the signature to add cleaning parameters, and always set the two fields. Replace the signature line and add the field assignments:

Signature becomes:

```python
def _make_interval(
    start_dist: float,
    end_dist: float,
    category: str,
    descriptions: list[Any],
    road_geom: Any,
    id_trc: Any,
    side: str,
    street_name: str,
    rate: float | None = None,
    cleaning: list[CleaningSchedule] | None = None,
    cleaning_text: str = "",
) -> dict[str, Any] | None:
```

In the `interval` dict (right after `"descriptions": ...,` and before `"geometry": ...,`), add:

```python
        "cleaning": cleaning if cleaning is not None else [],
        "cleaning_text": cleaning_text,
```

(Leave the existing `if rate is not None: interval["rate"] = rate` block unchanged.)

- [ ] **Step 6: Wire cleaning through `_build_side_intervals`**

Change the `_build_side_intervals` signature to accept `pole_cleaning` (add after `meter_group`):

```python
def _build_side_intervals(
    pole_data: pd.DataFrame,
    pole_signs: dict[Any, list[dict[str, Any]]],
    forward_arrow: int,
    backward_arrow: int,
    road_geom: Any,
    id_trc: Any,
    side: str,
    street_name: str,
    meter_group: pd.DataFrame | None = None,
    pole_cleaning: dict[Any, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
```

Immediately before the `# Merge levels` line (`merged = _merge_level_spans(...)`), build cleaning spans:

```python
    # Build cleaning spans (poles that carry a cleaning sign).
    pole_cleaning = pole_cleaning or {}
    cleaning_pids = set(pole_cleaning.keys())
    if cleaning_pids:
        c_mask = pole_data["POTEAU_ID_POT"].isin(cleaning_pids)
        cleaning_pole_data = pole_data[c_mask].reset_index(drop=True)
    else:
        cleaning_pole_data = pole_data.iloc[0:0]
    cleaning_spans = _walk_cleaning_spans(
        cleaning_pole_data, pole_cleaning, forward_arrow, backward_arrow, road_length,
    )
```

Update the merge call to pass cleaning spans:

```python
    merged = _merge_level_spans(level3_spans, level4_spans, cleaning_spans, road_length)
```

Replace the interval-build loop (the `for start, end, cat, descs in merged:` block) with a version that unpacks the 5-tuple, dedups cleaning items, and forwards cleaning fields:

```python
    intervals: list[dict[str, Any]] = []
    for start, end, cat, descs, cleaning_items in merged:
        # Short edges near intersections can't be free parking
        if cat == FREE and start == 0.0 and end < MIN_FREE_EDGE_M:
            cat = NO_DATA
        if cat == FREE and end == road_length and (road_length - start) < MIN_FREE_EDGE_M:
            cat = NO_DATA
        # Attach meter rate to paid intervals that came from meter data
        rate = meter_rate if cat == IntervalCategory.PAID else None
        # Dedup cleaning items by display text, preserving order.
        unique_items: dict[str, dict[str, Any]] = {}
        for item in cleaning_items:
            unique_items.setdefault(item["text"], item)
        schedules = [
            it["schedule"] for it in unique_items.values() if it["schedule"] is not None
        ]
        cleaning_text = "; ".join(unique_items.keys())
        iv = _make_interval(
            start, end, cat, descs, road_geom, id_trc, side, street_name,
            rate=rate, cleaning=schedules, cleaning_text=cleaning_text,
        )
        if iv:
            intervals.append(iv)

    return intervals
```

- [ ] **Step 7: Build `pole_cleaning` in `reconstruct_intervals` and pass it down**

In `reconstruct_intervals`, inside the `for (id_trc, side_key), group in grouped:` loop, the current code builds `pole_signs` and skips `STREET_CLEANING`. Replace that block so it also collects cleaning:

```python
        # Build lookups: pole_id -> sign info (classification) and cleaning info.
        # Street cleaning is excluded from classification but captured for popups
        # and the 24h overlay.
        pole_signs: dict[Any, list[dict[str, Any]]] = {}
        pole_cleaning: dict[Any, list[dict[str, Any]]] = {}
        for _, sign_row in group.iterrows():
            pid = sign_row["POTEAU_ID_POT"]
            if sign_row["sign_category"] == STREET_CLEANING:
                desc = str(sign_row["DESCRIPTION_RPA"])
                sched = parse_cleaning(desc)
                text = format_schedule(sched) if sched is not None else desc
                pole_cleaning.setdefault(pid, []).append({
                    "schedule": sched,
                    "text": text,
                    "arrow": sign_row.get("FLECHE_PAN", 0),
                })
                continue
            pole_signs.setdefault(pid, []).append({
                "category": sign_row["sign_category"],
                "description": sign_row["DESCRIPTION_RPA"],
                "arrow": sign_row.get("FLECHE_PAN", 0),
                "is_restrictive": sign_row["is_restrictive"],
            })
```

Update the `_build_side_intervals` call to pass `pole_cleaning`:

```python
        intervals.extend(
            _build_side_intervals(
                pole_data, pole_signs, forward_arrow, backward_arrow,
                road_geom, id_trc, side, street_name,
                meter_group=meter_groups.get((id_trc, side)),
                pole_cleaning=pole_cleaning,
            )
        )
```

- [ ] **Step 8: Add cleaning keys to gap-fill / meter-only rows and the empty schema**

Meter-only intervals already flow through `_make_interval` (they get empty cleaning by default) — no change needed there.

For the two gap-fill literal-dict appends (the "No sign data for this side" and "Short gap segment — no sign data" dicts), add these two keys to each dict (right after `"descriptions": ...,`):

```python
                "cleaning": [],
                "cleaning_text": "",
```

In the empty-result `gpd.GeoDataFrame(columns=[...])` near the end of `reconstruct_intervals`, add `"cleaning", "cleaning_text"` to the column list (before `"geometry"`):

```python
        return gpd.GeoDataFrame(
            columns=[
                "id_trc", "side", "start_dist", "end_dist", "length_m",
                "category", "street_name", "descriptions",
                "cleaning", "cleaning_text", "geometry",
            ],
            crs=CRS_MTM8,
        )
```

- [ ] **Step 9: Run the new + existing interval tests**

Run: `pixi run -e dev test tests/test_intervals.py`
Expected: PASS (new `TestCleaningAttachment` plus all previously passing tests, including `test_street_cleaning_ignored_in_classification`).

- [ ] **Step 10: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add montreal_parking/intervals.py tests/test_intervals.py
git commit -m "feat: attach street-cleaning schedules to interval segments"
```

---

### Task 3: `map.py` — popups and `cleaning.geojson` export

**Files:**
- Modify: `montreal_parking/map.py`
- Test: `tests/test_map.py` (new)

**Interfaces:**
- Consumes: interval `cleaning` / `cleaning_text` fields (Task 2); `parse_cleaning`, `format_schedule`, `SignCategory` (Task 1 / constants).
- Produces: `_export_cleaning_geojson(intervals_wgs: gpd.GeoDataFrame, dest: Path) -> bool`; `output/data/cleaning.geojson` with per-feature properties `popup_html`, `cleaning` (list of schedules), `cleaning_text`. Line and pole popups now include cleaning text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_map.py`:

```python
"""Tests for map GeoJSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from montreal_parking.constants import CRS_WGS84, IntervalCategory
from montreal_parking.map import _export_cleaning_geojson


def _intervals(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, crs=CRS_WGS84)


def test_cleaning_geojson_includes_only_parsed(tmp_path: Path) -> None:
    gdf = _intervals([
        {
            "street_name": "Rue Test",
            "category": IntervalCategory.FREE,
            "cleaning": [{
                "weekdays": [1], "start_min": 480, "end_min": 540,
                "month_start": (4, 1), "month_end": (12, 1),
            }],
            "cleaning_text": "Cleaning: Tue 8:00–9:00 · Apr 1 – Dec 1",
            "geometry": LineString([(-73.57, 45.50), (-73.569, 45.50)]),
        },
        {
            "street_name": "Rue Sans",
            "category": IntervalCategory.FREE,
            "cleaning": [],
            "cleaning_text": "",
            "geometry": LineString([(-73.57, 45.51), (-73.569, 45.51)]),
        },
    ])
    dest = tmp_path / "cleaning.geojson"
    assert _export_cleaning_geojson(gdf, dest) is True

    data = json.loads(dest.read_text())
    assert len(data["features"]) == 1
    props = data["features"][0]["properties"]
    assert "Cleaning: Tue" in props["popup_html"]
    assert props["cleaning"][0]["weekdays"] == [1]
    assert props["cleaning"][0]["month_start"] == [4, 1]  # tuple → JSON array


def test_cleaning_geojson_empty_returns_false(tmp_path: Path) -> None:
    gdf = _intervals([{
        "street_name": "Rue Sans",
        "category": IntervalCategory.FREE,
        "cleaning": [],
        "cleaning_text": "",
        "geometry": LineString([(-73.57, 45.51), (-73.569, 45.51)]),
    }])
    dest = tmp_path / "cleaning.geojson"
    assert _export_cleaning_geojson(gdf, dest) is False
    assert not dest.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_map.py`
Expected: FAIL with `ImportError: cannot import name '_export_cleaning_geojson'`.

- [ ] **Step 3: Add imports to `map.py`**

Extend the imports at the top of `montreal_parking/map.py`:

```python
from montreal_parking.cleaning import format_schedule, parse_cleaning
from montreal_parking.constants import (
    COLOR_FREE,
    COLOR_NO_DATA,
    COLOR_PAID,
    COLOR_RESTRICTED,
    COLOR_TIME_LIMITED,
    CRS_WGS84,
    MAP_CENTER,
    MAP_ZOOM_BOROUGH,
    MAP_ZOOM_DEFAULT,
    OUTPUT_DIR,
    SIMPLIFY_TOLERANCE,
    TILES_ATTR,
    TILES_URL,
    IntervalCategory,
    SignCategory,
)
```

- [ ] **Step 4: Add cleaning line to interval popups**

In `_export_category_geojson._popup`, add a cleaning line before the descriptions line. The block becomes:

```python
    def _popup(row: Any) -> str:
        lines = [
            f"<b>{row['street_name']}</b><br>",
            f"Category: {category}<br>",
        ]
        if "rate" in row.index and pd.notna(row.get("rate")):
            lines.append(f"Rate: ${row['rate']:.2f}/hr<br>")
        lines.append(f"Length: {row['length_m']:.0f}m<br>")
        if "cleaning_text" in row.index and row.get("cleaning_text"):
            lines.append(
                f"\U0001f9f9 {html.escape(str(row['cleaning_text']))}<br>"
            )
        lines.append(f"<small>{html.escape(str(row['descriptions'])[:200])}</small>")
        return "".join(lines)
```

- [ ] **Step 5: Format cleaning schedules in pole popups**

In `_build_pole_geojson`, replace the per-sign line construction inside the `groupby(...).apply(...)` lambda so cleaning signs show their formatted schedule when parseable. Replace the lambda block:

```python
    def _sign_line(row: Any) -> str:
        desc = str(row["DESCRIPTION_RPA"])
        arrow = arrow_display.get(row["FLECHE_PAN"], "")
        cat = row["sign_category"]
        if cat == SignCategory.STREET_CLEANING:
            sched = parse_cleaning(desc)
            if sched is not None:
                return f"{arrow} {cat}: {html.escape(format_schedule(sched))}"
        return f"{arrow} {cat}: {html.escape(desc)}"

    sign_html: dict[Any, str] = (
        signs_df.groupby(group_cols)
        .apply(  # type: ignore[call-overload]
            lambda g: "<br>".join(_sign_line(row) for _, row in g.iterrows()),
            include_groups=False,
        )
        .to_dict()
    )
```

- [ ] **Step 6: Add `_export_cleaning_geojson`**

Add this function to `map.py` (e.g. after `_export_category_geojson`):

```python
def _export_cleaning_geojson(
    intervals_wgs: gpd.GeoDataFrame,
    dest: Path,
) -> bool:
    """Export intervals that carry a parsed cleaning schedule to GeoJSON.

    One feature per interval with a non-empty structured schedule. Properties:
    popup_html, cleaning (list of CleaningSchedule dicts), cleaning_text.
    Returns True if non-empty.
    """
    if "cleaning" not in intervals_wgs.columns:
        return False
    mask = intervals_wgs["cleaning"].apply(
        lambda c: isinstance(c, list) and len(c) > 0
    )
    subset = intervals_wgs[mask].copy()
    if subset.empty:
        return False

    subset["geometry"] = subset["geometry"].simplify(SIMPLIFY_TOLERANCE)
    features = []
    for _, row in subset.iterrows():
        text = str(row.get("cleaning_text", ""))
        popup = (
            f"<b>{html.escape(str(row['street_name']))}</b><br>"
            f"\U0001f9f9 {html.escape(text)}"
        )
        features.append({
            "type": "Feature",
            "geometry": shapely.geometry.mapping(row["geometry"]),
            "properties": {
                "popup_html": popup,
                "cleaning": row["cleaning"],
                "cleaning_text": text,
            },
        })

    with open(dest, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"    cleaning: {len(features)} features -> {dest}")
    return True
```

- [ ] **Step 7: Run the map test**

Run: `pixi run -e dev test tests/test_map.py`
Expected: PASS.

- [ ] **Step 8: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add montreal_parking/map.py tests/test_map.py
git commit -m "feat: cleaning text in popups and cleaning.geojson export"
```

---

### Task 4: `map.py` — "Cleaning ≤24h" overlay layer and client JS

**Files:**
- Modify: `montreal_parking/map.py` (`_build_html_shell` and `build_map`)

**Interfaces:**
- Consumes: `cleaning.geojson` (Task 3); the existing `layers` list plumbing in `build_map` / `_build_html_shell`.
- Produces: a new overlay layer with `is_cleaning: True`; client JS `isCleaningSoon(cleaning)` filtering features to the next-24h Montreal-time window and rendering them as a black dashed line; a periodic re-render so the overlay stays current.

- [ ] **Step 1: Register the cleaning overlay in `build_map`**

In `build_map`, after the category-layer export loop (right after the `for display_name, category, color, default_on in _LAYER_CONFIG:` block) and before the poles export, add:

```python
    # Cleaning overlay (off by default; JS filters to the next 24h)
    cleaning_dest = data_dir / "cleaning.geojson"
    if _export_cleaning_geojson(intervals_wgs, cleaning_dest):
        layers.append({
            "var": "layer_cleaning",
            "file": "cleaning.geojson",
            "name": "Cleaning ≤24h",
            "color": "#000",
            "default_on": False,
            "is_cleaning": True,
        })
```

- [ ] **Step 2: Give the cleaning layer dashed style + 24h filter in `_build_html_shell`**

In `_build_html_shell`, inside the `for layer in layers:` loop, read the flag and add a cleaning branch to the `opts` selection. Replace the `is_point = layer.get("is_point", False)` line and the `opts = (...)` assignment with:

```python
        is_point = layer.get("is_point", False)
        is_cleaning = layer.get("is_cleaning", False)

        if is_cleaning:
            opts = (
                "filter: function(f) { return f.properties "
                "&& isCleaningSoon(f.properties.cleaning); }, "
                "style: {color:'#000',weight:3,opacity:0.9,"
                "dashArray:'6,6',lineCap:'butt'}"
            )
        elif is_point:
            opts = (
                f"pointToLayer: function(f, ll) {{"
                f" return L.circleMarker(ll, {{radius:3,color:'{color}',"
                f"fillColor:'{color}',fillOpacity:0.7}}); }}"
            )
        else:
            opts = f"style: {{color:'{color}',weight:5,opacity:0.8,lineCap:'butt'}}"
```

- [ ] **Step 3: Exclude the cleaning layer from driving-mode category colors**

In `_build_html_shell`, update the `_cc_entries` comprehension so the cleaning overlay does not feed the driving-mode side bars:

```python
    # Category colors for driving mode (line layers only; exclude cleaning overlay)
    _cc_entries = ", ".join(
        f'"{layer["var"]}": "{layer["color"]}"'
        for layer in layers
        if not layer.get("is_point") and not layer.get("is_cleaning")
    )
    category_colors_js = f"var categoryColors = {{{_cc_entries}}};"
```

- [ ] **Step 4: Compute the periodic-refresh snippet**

In `_build_html_shell`, just before the `return f"""<!DOCTYPE html> ...` line, compute a refresh snippet for the cleaning layer (re-render every 10 minutes so a long-open page crosses window boundaries correctly):

```python
    cleaning_layer = next((lyr for lyr in layers if lyr.get("is_cleaning")), None)
    cleaning_refresh_js = (
        f"setInterval(function() {{ renderLayer('{cleaning_layer['var']}'); }}, 600000);"
        if cleaning_layer
        else ""
    )
```

- [ ] **Step 5: Add the `isCleaningSoon` JS helpers and inject the refresh**

In the returned HTML template string, add the cleaning-time helper block immediately **before** the `{layers_init}` placeholder (so it is defined before any layer renders):

```javascript
    // --- Street-cleaning "next 24h" (Montreal time) ---
    function montrealNow() {{
      var fmt = new Intl.DateTimeFormat('en-CA', {{
        timeZone: 'America/Toronto', hour12: false,
        year: 'numeric', month: 'numeric', day: 'numeric',
        hour: 'numeric', minute: 'numeric', weekday: 'short'
      }});
      var parts = {{}};
      fmt.formatToParts(new Date()).forEach(function(p) {{ parts[p.type] = p.value; }});
      var wdMap = {{Sun:6, Mon:0, Tue:1, Wed:2, Thu:3, Fri:4, Sat:5}};
      return {{
        year: +parts.year, month: +parts.month, day: +parts.day,
        hour: (+parts.hour) % 24, minute: +parts.minute,
        weekday: wdMap[parts.weekday]
      }};
    }}

    function addDays(now, d) {{
      var base = new Date(Date.UTC(now.year, now.month - 1, now.day));
      base.setUTCDate(base.getUTCDate() + d);
      return {{
        month: base.getUTCMonth() + 1,
        day: base.getUTCDate(),
        weekday: (base.getUTCDay() + 6) % 7  // JS Sun=0 → Mon=0
      }};
    }}

    function inMonthRange(month, day, ms, me) {{
      function key(m, d) {{ return m * 100 + d; }}
      var k = key(month, day), a = key(ms[0], ms[1]), b = key(me[0], me[1]);
      if (a <= b) return k >= a && k <= b;
      return k >= a || k <= b;  // wraps year end
    }}

    function scheduleSoon(s, now) {{
      var nowMin = now.hour * 60 + now.minute;
      var horizonEnd = nowMin + 1440;  // +24h, in minutes from today's midnight
      for (var d = 0; d <= 1; d++) {{
        var dt = addDays(now, d);
        if (s.weekdays.indexOf(dt.weekday) === -1) continue;
        if (!inMonthRange(dt.month, dt.day, s.month_start, s.month_end)) continue;
        var winStart = d * 1440 + s.start_min;
        var winEnd = d * 1440 + s.end_min;
        if (winEnd > nowMin && winStart < horizonEnd) return true;
      }}
      return false;
    }}

    function isCleaningSoon(cleaning) {{
      if (!cleaning || !cleaning.length) return false;
      var now = montrealNow();
      for (var i = 0; i < cleaning.length; i++) {{
        if (scheduleSoon(cleaning[i], now)) return true;
      }}
      return false;
    }}
```

Then, immediately **after** the `{default_adds}` placeholder in the template, inject the refresh timer:

```
    {cleaning_refresh_js}
```

- [ ] **Step 6: Build a borough and verify the overlay end-to-end**

Run: `pixi run python main.py --borough Plateau`
Then open `output/index.html` in a browser (or `python -m http.server` from `output/`).
Expected/verify:
- `output/data/cleaning.geojson` exists and is non-empty.
- A line popup on a street with cleaning shows a `🧹 Cleaning: …` line.
- The layer control lists **"Cleaning ≤24h"**, off by default; toggling it on draws black dashed lines over some colored segments (which segments depend on the current Montreal date/time — if none qualify today, temporarily confirm the mechanism by checking the browser console with a hand-called `isCleaningSoon` on a known feature's `cleaning` array).
- The base category color is still visible through the dash gaps.

- [ ] **Step 7: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add montreal_parking/map.py
git commit -m "feat: 'Cleaning <=24h' dashed overlay with Montreal-time window"
```

---

### Task 5: Documentation sweep

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the module table and output structure**

In `CLAUDE.md`:

- Add a row to the Code Structure table:

```
| `cleaning.py`  | Parse French street-cleaning sign text into structured schedules (weekday/time/season) + English formatting |
```

- Update the `map.py` row to mention the cleaning overlay, e.g. append: `, "Cleaning ≤24h" dashed overlay (client-side Montreal-time window)`.

- Update the `intervals.py` row to note it attaches cleaning schedules to segments (append): `; attaches street-cleaning schedules to segments as metadata`.

- In the Output Structure block, add `cleaning.geojson` under `data/`:

```
    cleaning.geojson
```

- Under Key Design Decisions, add a bullet:

```
- **Street cleaning**: cleaning signs are parsed into structured schedules at build time and attached to interval segments (without changing category). The "Cleaning ≤24h" overlay computes the next-24h window client-side in Montreal time and draws a black dashed line over affected segments.
```

- [ ] **Step 2: Verify the whole suite + checks pass**

Run: `pixi run -e dev test && pixi run -e dev lint && pixi run -e dev typecheck`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document street-cleaning schedule parsing and overlay"
```

---

## Self-Review Notes

- **Spec coverage:** parser (Task 1) ✓; per-span arrow-walk attach without category change (Task 2) ✓; both line + pole popups (Task 3) ✓; dedicated `cleaning.geojson` (Task 3) ✓; off-by-default dashed overlay in Montreal time (Task 4) ✓; English formatting (Task 1) ✓; raw-text fallback excluded from 24h (Tasks 1–4) ✓; docs (Task 5) ✓.
- **Type consistency:** `CleaningSchedule` keys (`weekdays`, `start_min`, `end_min`, `month_start`, `month_end`) are used identically in Python (Tasks 1–3) and JS (`s.weekdays`, `s.start_min`, `s.end_min`, `s.month_start`, `s.month_end`, Task 4). `_export_cleaning_geojson`, `_walk_cleaning_spans`, `_cleaning_key`, `_merge_level_spans` signatures match across their call sites.
- **Excluded (YAGNI, per spec):** driving-mode integration; biweekly/nth-weekday 24h math; non-English formatting.
- **Known behavior:** a currently-in-progress cleaning window counts as "within 24h" (intentional — see spec note).
```
