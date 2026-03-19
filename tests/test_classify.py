"""Tests for sign classification logic.

Test cases derived from real Montreal parking sign descriptions and
specific debugging scenarios (Poitevin, Saint-Andre, Genereux).
"""

from __future__ import annotations

import pytest

from montreal_parking.classify import classify_sign, is_restrictive, sign_level
from montreal_parking.constants import SignCategory

# ---------------------------------------------------------------------------
# classify_sign
# ---------------------------------------------------------------------------


class TestClassifySign:
    """classify_sign should map sign descriptions to the correct category."""

    # --- No parking / no stopping ---

    def test_backslash_p_is_no_parking(self) -> None:
        assert classify_sign("\\P") == SignCategory.NO_PARKING

    def test_backslash_a_is_no_parking(self) -> None:
        r"""\A (arrêt interdit) is stricter than \P but same category."""
        assert classify_sign("\\A") == SignCategory.NO_PARKING

    def test_backslash_p_with_times_is_no_parking(self) -> None:
        assert classify_sign("\\P 09h-17h") == SignCategory.NO_PARKING

    def test_backslash_p_deux_cotes_is_no_parking(self) -> None:
        assert classify_sign("\\P DEUX COTES") == SignCategory.NO_PARKING

    # --- Permit / sticker required ---

    def test_backslash_p_with_s3r_is_permit(self) -> None:
        assert classify_sign("\\P 9h-23h EXCEPTE S3R") == SignCategory.PERMIT

    def test_s3r_without_backslash_p_is_permit(self) -> None:
        assert classify_sign("S3R SECTEUR 1") == SignCategory.PERMIT

    def test_autocol_is_permit(self) -> None:
        assert classify_sign("AUTOCOL SECTEUR 3") == SignCategory.PERMIT

    def test_vignette_is_permit(self) -> None:
        assert classify_sign("VIGNETTE DE STATIONNEMENT") == SignCategory.PERMIT

    def test_backslash_p_with_vignette_is_permit(self) -> None:
        assert classify_sign("\\P EXCEPTE VIGNETTE 7") == SignCategory.PERMIT

    # --- Paid parking ---

    def test_tarif_is_paid(self) -> None:
        assert classify_sign("TARIF 2$/HR") == SignCategory.PAID

    def test_parcoflex_is_paid(self) -> None:
        assert classify_sign("PARCOFLEX") == SignCategory.PAID

    def test_parcometre_is_paid(self) -> None:
        assert classify_sign("PARCOMETRE") == SignCategory.PAID

    def test_parcometre_accented_is_paid(self) -> None:
        assert classify_sign("PARCOMÈTRE") == SignCategory.PAID

    def test_stationnement_payant_is_paid(self) -> None:
        assert classify_sign("STATIONNEMENT PAYANT") == SignCategory.PAID

    # --- Time-limited ---

    def test_p_60_min_is_time_limited(self) -> None:
        assert classify_sign("P 60 MIN") == SignCategory.TIME_LIMITED

    def test_p_120_min_with_hours_is_time_limited(self) -> None:
        assert classify_sign("P 120 MIN 9H-17H") == SignCategory.TIME_LIMITED

    def test_p_60_min_weekday_is_time_limited(self) -> None:
        assert classify_sign("P 60 min 08h-19h LUN. AU VEN.") == SignCategory.TIME_LIMITED

    # --- Street cleaning ---

    def test_street_cleaning_seasonal(self) -> None:
        assert classify_sign("\\P 08h-09h MAR. 1 AVRIL AU 1 DEC.") == SignCategory.STREET_CLEANING

    def test_street_cleaning_with_er(self) -> None:
        assert classify_sign("\\P 12h30-14h30 MARDI 1er AVRIL AU 1er DEC.") == SignCategory.STREET_CLEANING

    def test_street_cleaning_a_prefix(self) -> None:
        """Street cleaning with \\A prefix."""
        assert classify_sign("\\A 08h-09h MERCREDI 1 AVRIL AU 30 NOV.") == SignCategory.STREET_CLEANING

    # --- Unrestricted ---

    def test_bare_p_is_unrestricted(self) -> None:
        assert classify_sign("P") == SignCategory.UNRESTRICTED

    def test_p_with_space_is_unrestricted(self) -> None:
        """'P ' followed by something without MIN or H → unrestricted."""
        assert classify_sign("P LIVRAISON") == SignCategory.UNRESTRICTED

    # --- Panonceau (sub-sign, non-restrictive) ---

    def test_panonceau_is_panonceau(self) -> None:
        assert classify_sign("PANONCEAU EXCEPTE PERIODE INTERDITE") == SignCategory.PANONCEAU

    def test_panonceau_remorquage(self) -> None:
        assert classify_sign("PANONCEAU ZONE DE REMORQUAGE") == SignCategory.PANONCEAU

    def test_panonceau_reserve(self) -> None:
        assert classify_sign("PANONCEAU RESERVE DETENTEUR DE PERMIS") == SignCategory.PANONCEAU

    # --- Other / edge cases ---

    def test_non_string_returns_other(self) -> None:
        assert classify_sign(None) == SignCategory.OTHER  # type: ignore[arg-type]
        assert classify_sign(42) == SignCategory.OTHER  # type: ignore[arg-type]

    def test_empty_string_returns_other(self) -> None:
        assert classify_sign("") == SignCategory.OTHER

    def test_unknown_description_returns_other(self) -> None:
        assert classify_sign("ZONE SCOLAIRE") == SignCategory.OTHER


# ---------------------------------------------------------------------------
# is_restrictive
# ---------------------------------------------------------------------------


class TestIsRestrictive:
    """is_restrictive should be True only for categories that block free parking."""

    @pytest.mark.parametrize("category", [SignCategory.NO_PARKING, SignCategory.PERMIT])
    def test_restrictive_categories(self, category: SignCategory) -> None:
        assert is_restrictive(category) is True

    @pytest.mark.parametrize(
        "category",
        [SignCategory.PAID, SignCategory.TIME_LIMITED, SignCategory.STREET_CLEANING,
         SignCategory.UNRESTRICTED, SignCategory.PANONCEAU, SignCategory.OTHER],
    )
    def test_non_restrictive_categories(self, category: SignCategory) -> None:
        assert is_restrictive(category) is False


# ---------------------------------------------------------------------------
# sign_level
# ---------------------------------------------------------------------------


class TestSignLevel:
    """sign_level should return the correct priority level for each category."""

    @pytest.mark.parametrize("category", [SignCategory.NO_PARKING, SignCategory.PERMIT])
    def test_level_3_categories(self, category: SignCategory) -> None:
        assert sign_level(category) == 3

    @pytest.mark.parametrize("category", [SignCategory.TIME_LIMITED, SignCategory.UNRESTRICTED, SignCategory.PAID])
    def test_level_4_categories(self, category: SignCategory) -> None:
        assert sign_level(category) == 4

    @pytest.mark.parametrize("category", [SignCategory.STREET_CLEANING, SignCategory.PANONCEAU, SignCategory.OTHER])
    def test_none_categories(self, category: SignCategory) -> None:
        assert sign_level(category) is None
