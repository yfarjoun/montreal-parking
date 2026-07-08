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
