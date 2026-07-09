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
