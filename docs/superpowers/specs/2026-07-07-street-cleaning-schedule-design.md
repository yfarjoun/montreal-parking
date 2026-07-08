# Street-Cleaning Schedule: Popups + "Cleaning ≤24h" Overlay

**Date:** 2026-07-07
**Status:** Approved design, ready for implementation planning

## Goal

1. Show a human-readable street-cleaning schedule in map popups (both street-side
   interval/line popups and sign-pole popups).
2. Add an opt-in map overlay that highlights street sides where cleaning will
   occur within the next 24 hours, computed live in the browser at page load.

## Background / current state

- Street-cleaning signs are already classified as `SignCategory.STREET_CLEANING`
  by the `_CLEANING_RE` regex in `classify.py`.
- They are already snapped to road segments/sides (`snap.py`), so their geometry
  and arrow (`FLECHE_PAN`) are available in the snapped signs table.
- They are **deliberately excluded** from interval reconstruction today
  (`intervals.py`, in `reconstruct_intervals`: signs with category
  `STREET_CLEANING` are skipped when building `pole_signs`).
- The schedule detail (weekday, time window, seasonal month range) lives
  **unparsed** inside `DESCRIPTION_RPA`, e.g. `\P 08h-09h MAR. 1 AVRIL AU 1 DEC.`.
- The site is a **static build** (weekly + manual deploy). Therefore "next 24
  hours" must be computed **client-side at page load** from a structured schedule
  baked into the data — it cannot be a build-time flag (would go stale within a
  day).

## Key decisions (from brainstorming)

- **Timing model:** parse the French text at build time into structured fields;
  compute the "next 24h" window in the browser using the user's clock,
  interpreted in Montreal time.
- **Popup scope:** schedule shown in **both** line/interval popups and pole popups.
- **Visual mode:** a **toggleable overlay layer** in the existing layer control,
  **off by default** (opt-in). No driving-mode integration in this iteration.
- **Granularity:** attach schedules **per span** via a cleaning-specific arrow
  walk (mirroring the existing level-3/4 walk), so a cleaning block is its own
  interval segment and boundaries line up with existing intervals.
- **Parse coverage:** parse common cases; fall back to raw text for the rest and
  **exclude unparsed schedules from the 24h coloring** (conservative).
- **Overlay data source:** a **dedicated `cleaning.geojson`** file (approved),
  rather than re-scanning the category layers client-side.
- **Formatting:** **English** schedule text, matching existing UI labels.

## Components

### 1. `montreal_parking/cleaning.py` (new module) — the parser

Responsibility: turn a single `DESCRIPTION_RPA` string into a structured cleaning
schedule, and format a structured schedule for display.

Data shape:

```
CleaningSchedule = {
    "weekdays": list[int],   # 0=Mon … 6=Sun (Python weekday convention)
    "start_min": int,        # minutes since midnight, local Montreal time
    "end_min": int,
    "month_start": tuple[int, int],  # (month, day), e.g. (4, 1) for Apr 1
    "month_end": tuple[int, int],    # e.g. (12, 1) for Dec 1
}
```

API:

- `parse_cleaning(description: str) -> CleaningSchedule | None`
- `format_schedule(sched: CleaningSchedule) -> str`
  - Example output: `Cleaning: Tue 8:00–9:00 · Apr 1 – Dec 1`
  - Multiple weekdays: `Mon–Fri`, `Tue & Thu`.

Parsing rules (common cases — return a `CleaningSchedule`):

- Single weekday: `MAR.` / `MARDI`.
- Multiple weekdays joined with `ET`: `MAR. ET JEU.`.
- Weekday ranges with `AU`: `LUN. AU VEN.`.
- Time window: `08h-09h`, `12h30-14h30` (both `h` and `H`, optional minutes).
- Seasonal month range: `1 AVRIL AU 1 DEC.`, `1er AVRIL AU 1er DEC.`.
- Accept both abbreviated (`LUN MAR MER JEU VEN SAM DIM`) and full French
  weekday and month names, case-insensitive.

Fallback rules (return `None`):

- Biweekly / nth-weekday: `1ER ET 3E MAR.`, `2E ET 4E MER.`.
- Any string that does not yield a weekday set + time window + month range with
  confidence.

Observability: `parse_cleaning` is pure. A build-time helper (called from
`intervals.py` or a small wrapper) logs the count of parsed vs. failed
street-cleaning descriptions so parse coverage is visible in build output.

Testing: `tests/test_cleaning.py` with a table of real sign strings covering each
supported pattern and representative fallback cases. Assert both the parsed
structure and the formatted output.

### 2. `intervals.py` — attach cleaning schedules to spans

Replace the current "drop street_cleaning" behavior with a parallel cleaning walk
that does not affect interval categories.

- Add `_walk_cleaning_spans(...)`, mirroring `_walk_level_spans`, using the same
  forward/backward arrow convention (`FLECHE_PAN` 2=left, 3=right, direction
  flips by side). It collects the `CleaningSchedule`(s) active on each span rather
  than a single category. Raw (unparsed) cleaning descriptions are carried too,
  for popup text.
- Extend `_merge_level_spans` to take cleaning spans as a **third input**:
  - Cleaning span boundaries are added to the boundary set, so a cleaning block
    becomes its own segment.
  - Cleaning **does not determine the category** — category is still decided by
    level-3/level-4 logic (a free block with cleaning stays `free`).
  - At each micro-segment midpoint, collect the overlapping cleaning schedules
    and raw texts.
  - Adjacent segments merge only when category **and** the attached
    cleaning-schedule-set match (so cleaning boundaries are preserved).
- New per-interval fields produced by `_make_interval` / the merge:
  - `cleaning`: list of structured `CleaningSchedule` dicts (JSON-serializable;
    empty list if none).
  - `cleaning_text`: formatted, human-readable string joining parsed schedules;
    falls back to the raw French text for descriptions that did not parse.
    Empty string if no cleaning applies.

Note: `snap.py` and `classify.py` are unchanged — cleaning signs are already
classified and snapped; they were only being filtered out at the interval stage.

### 3. `map.py` — popups, overlay GeoJSON, and client JS

Popups:

- Interval/line popups (`_export_category_geojson._popup`): append `cleaning_text`
  when non-empty.
- Pole popups (`_build_pole_geojson`): continue to show street-cleaning sign lines;
  where a parsed schedule exists, show the formatted text; otherwise the raw text.

Overlay data:

- Emit `output/data/cleaning.geojson`: one feature per interval that has **≥1
  parsed** `CleaningSchedule`. Geometry = the interval's offset line (same
  geometry as the category layers). Properties:
  - `cleaning`: structured schedule list (drives the 24h computation),
  - `cleaning_text`: formatted string (for the overlay feature's popup),
  - `popup_html`: same convention as other layers.

Overlay layer + JS:

- Add a new overlay entry **"Cleaning ≤24h"** to the existing
  `L.control.layers` overlays, **off by default**.
- When enabled, JS loads `cleaning.geojson`, filters to features whose schedule
  intersects `[now, now + 24h]`, and renders them as a **black dashed line drawn
  on top of** the existing colored category line. The dash pattern lets the
  underlying category color show through the gaps — it marks the segment as
  "cleaning soon" without replacing its color. (Leaflet path style:
  `color: '#000'`, `dashArray`, and a weight equal to or slightly less than the
  base line so the category color remains visible around it.)
- Recompute on `overlayadd` and on a periodic timer (so a page left open crosses
  into a new cleaning window correctly).
- "Now" and the seasonal check are computed in **Montreal time**
  (`Intl.DateTimeFormat` with `timeZone: 'America/Toronto'`), independent of the
  user's own timezone.

24h window semantics (client-side):

- A feature is "within 24h" if, for either calendar day spanned by
  `[now, now+24h]` (in Montreal time):
  - the day's weekday is in `weekdays`, and
  - that day (month, day) falls within `[month_start, month_end]`, and
  - the day's `[start_min, end_min]` window intersects `[now, now+24h]`.

## Data flow

```
classify.py   (unchanged: STREET_CLEANING already tagged)
   ↓
snap.py       (unchanged: cleaning signs already snapped, arrow preserved)
   ↓
intervals.py  (new: _walk_cleaning_spans + attach cleaning/cleaning_text to intervals)
   ↓
map.py        (popups show cleaning_text; emit cleaning.geojson; add "Cleaning ≤24h" overlay)
```

## Scope boundaries (YAGNI)

- No driving-mode side-bar integration for cleaning in this iteration.
- No biweekly / nth-weekday 24h math — raw-text fallback only, excluded from the
  overlay.
- English formatting only.
- No new external data sources — all derived from existing signage data.

## Testing

- `tests/test_cleaning.py`: parser + formatter table tests (supported patterns +
  fallbacks).
- Extend interval tests to assert cleaning spans attach to the correct
  side/segment given a small synthetic pole layout with a cleaning sign and an
  arrow, and that category is unaffected.
- Manual/local verification: build a borough (`--borough Plateau`), open the map,
  confirm popups show schedules and the "Cleaning ≤24h" overlay toggles and
  highlights plausibly (with a note that the highlight depends on current
  Montreal date/time).

## Files touched

- `montreal_parking/cleaning.py` (new)
- `montreal_parking/intervals.py`
- `montreal_parking/map.py`
- `tests/test_cleaning.py` (new), plus interval-test additions
- `CLAUDE.md` module table: add a row for `cleaning.py` and note the cleaning
  overlay in map.py's responsibility.