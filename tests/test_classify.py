"""Tests for sign classification logic.

Test cases derived from real Montreal parking sign descriptions and
specific debugging scenarios (Poitevin, Saint-Andre, Genereux).
"""

from __future__ import annotations

import pytest

from montreal_parking.classify import classify_sign, is_restrictive

# ---------------------------------------------------------------------------
# classify_sign
# ---------------------------------------------------------------------------


class TestClassifySign:
    """classify_sign should map sign descriptions to the correct category."""

    # --- No parking / no stopping ---

    def test_backslash_p_is_no_parking(self) -> None:
        assert classify_sign("\\P") == "no_parking"

    def test_backslash_a_is_no_parking(self) -> None:
        r"""\A (arrêt interdit) is stricter than \P but same category."""
        assert classify_sign("\\A") == "no_parking"

    def test_backslash_p_with_times_is_no_parking(self) -> None:
        assert classify_sign("\\P 09h-17h") == "no_parking"

    def test_backslash_p_deux_cotes_is_no_parking(self) -> None:
        assert classify_sign("\\P DEUX COTES") == "no_parking"

    # --- Permit / sticker required ---

    def test_backslash_p_with_s3r_is_permit(self) -> None:
        assert classify_sign("\\P 9h-23h EXCEPTE S3R") == "permit"

    def test_s3r_without_backslash_p_is_permit(self) -> None:
        assert classify_sign("S3R SECTEUR 1") == "permit"

    def test_autocol_is_permit(self) -> None:
        assert classify_sign("AUTOCOL SECTEUR 3") == "permit"

    def test_vignette_is_permit(self) -> None:
        assert classify_sign("VIGNETTE DE STATIONNEMENT") == "permit"

    def test_backslash_p_with_vignette_is_permit(self) -> None:
        assert classify_sign("\\P EXCEPTE VIGNETTE 7") == "permit"

    # --- Paid parking ---

    def test_tarif_is_paid(self) -> None:
        assert classify_sign("TARIF 2$/HR") == "paid"

    def test_parcoflex_is_paid(self) -> None:
        assert classify_sign("PARCOFLEX") == "paid"

    def test_parcometre_is_paid(self) -> None:
        assert classify_sign("PARCOMETRE") == "paid"

    def test_parcometre_accented_is_paid(self) -> None:
        assert classify_sign("PARCOMÈTRE") == "paid"

    def test_stationnement_payant_is_paid(self) -> None:
        assert classify_sign("STATIONNEMENT PAYANT") == "paid"

    # --- Time-limited ---

    def test_p_60_min_is_time_limited(self) -> None:
        assert classify_sign("P 60 MIN") == "time_limited"

    def test_p_120_min_with_hours_is_time_limited(self) -> None:
        assert classify_sign("P 120 MIN 9H-17H") == "time_limited"

    def test_p_60_min_weekday_is_time_limited(self) -> None:
        assert classify_sign("P 60 min 08h-19h LUN. AU VEN.") == "time_limited"

    # --- Street cleaning ---

    def test_street_cleaning_seasonal(self) -> None:
        assert classify_sign("\\P 08h-09h MAR. 1 AVRIL AU 1 DEC.") == "street_cleaning"

    def test_street_cleaning_with_er(self) -> None:
        assert classify_sign("\\P 12h30-14h30 MARDI 1er AVRIL AU 1er DEC.") == "street_cleaning"

    def test_street_cleaning_a_prefix(self) -> None:
        """Street cleaning with \\A prefix."""
        assert classify_sign("\\A 08h-09h MERCREDI 1 AVRIL AU 30 NOV.") == "street_cleaning"

    # --- Unrestricted ---

    def test_bare_p_is_unrestricted(self) -> None:
        assert classify_sign("P") == "unrestricted"

    def test_p_with_space_is_unrestricted(self) -> None:
        """'P ' followed by something without MIN or H → unrestricted."""
        assert classify_sign("P LIVRAISON") == "unrestricted"

    # --- Panonceau (sub-sign, non-restrictive) ---

    def test_panonceau_is_panonceau(self) -> None:
        assert classify_sign("PANONCEAU EXCEPTE PERIODE INTERDITE") == "panonceau"

    def test_panonceau_remorquage(self) -> None:
        assert classify_sign("PANONCEAU ZONE DE REMORQUAGE") == "panonceau"

    def test_panonceau_reserve(self) -> None:
        assert classify_sign("PANONCEAU RESERVE DETENTEUR DE PERMIS") == "panonceau"

    # --- Other / edge cases ---

    def test_non_string_returns_other(self) -> None:
        assert classify_sign(None) == "other"  # type: ignore[arg-type]
        assert classify_sign(42) == "other"  # type: ignore[arg-type]

    def test_empty_string_returns_other(self) -> None:
        assert classify_sign("") == "other"

    def test_unknown_description_returns_other(self) -> None:
        assert classify_sign("ZONE SCOLAIRE") == "other"


# ---------------------------------------------------------------------------
# is_restrictive
# ---------------------------------------------------------------------------


class TestIsRestrictive:
    """is_restrictive should be True only for categories that block free parking."""

    @pytest.mark.parametrize("category", ["no_parking", "permit", "paid"])
    def test_restrictive_categories(self, category: str) -> None:
        assert is_restrictive(category) is True

    @pytest.mark.parametrize(
        "category",
        ["time_limited", "street_cleaning", "unrestricted", "panonceau", "other"],
    )
    def test_non_restrictive_categories(self, category: str) -> None:
        assert is_restrictive(category) is False
