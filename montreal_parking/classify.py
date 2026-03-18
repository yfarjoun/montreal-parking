"""Sign classification logic."""

from __future__ import annotations

import re

import pandas as pd

# Street cleaning: \P <time> <weekday> <seasonal range>
# e.g., "\P 08h-09h MAR. 1 AVRIL AU 1 DEC."
# Also matches: "\P 12h30-14h30 MARDI 1er AVRIL AU 1er DEC."
_CLEANING_RE = re.compile(
    r"^\\[PA]\s+\d{1,2}[hH]"  # \P or \A + time
    r".*"
    r"(LUN|MAR|MER|JEU|VEN|SAM|DIM|LUNDI|MARDI|MERCREDI|JEUDI|VENDREDI|SAMEDI|DIMANCHE)"
    r".*"
    r"\d+\s*(E[Rr])?\s*(AVRIL|MARS|MAI|JUIN|JUIL|AOUT|SEPT|OCT|NOV|DEC)",
    re.IGNORECASE,
)


def classify_sign(description: str) -> str:
    """Classify a sign description into a category.

    Returns one of: 'no_parking', 'permit', 'paid', 'time_limited',
    'street_cleaning', 'unrestricted', 'panonceau', 'other'.
    """
    if not isinstance(description, str):
        return "other"

    desc = description.upper().strip()

    # No stopping / no parking (starts with \P or \A)
    if desc.startswith("\\P") or desc.startswith("\\A"):
        # Permit sign
        if "S3R" in desc or "AUTOCOL" in desc or "VIGNETTE" in desc:
            return "permit"
        # Street cleaning: short time window + specific weekday + seasonal range
        if _CLEANING_RE.match(description.strip()):
            return "street_cleaning"
        return "no_parking"

    # Permit / sticker required
    if "S3R" in desc or "AUTOCOL" in desc or "VIGNETTE" in desc:
        return "permit"

    # Paid parking
    if "TARIF" in desc or "PARCOFLEX" in desc or "PARCOMETRE" in desc or "PARCOMÈTRE" in desc or "PAYANT" in desc:
        return "paid"

    # Time-limited parking (e.g., "P 60 MIN", "P 120 MIN 9H-17H")
    if desc.startswith("P ") and ("MIN" in desc or "H" in desc):
        return "time_limited"

    # Explicit unrestricted parking
    if desc.startswith("P ") or desc == "P":
        return "unrestricted"

    # PANONCEAU = sub-sign/plaque that modifies the sign above it.
    # Not a standalone restriction — the parent sign is classified separately.
    if desc.startswith("PANONCEAU"):
        return "panonceau"

    return "other"


def is_restrictive(category: str) -> bool:
    """Whether a sign category prevents free parking."""
    return category in ("no_parking", "permit")


def sign_level(category: str) -> int | None:
    """Priority level for interval classification. Higher overrides lower.

    Level 3: parking disallowed (no_parking, permit)
    Level 4: parking allowed with conditions (time_limited, unrestricted, paid)
    None: not used in interval classification (street_cleaning, panonceau, other)
    """
    if category in ("no_parking", "permit"):
        return 3
    if category in ("time_limited", "unrestricted", "paid"):
        return 4
    return None


def classify_all_signs(df: pd.DataFrame) -> pd.DataFrame:
    """Add sign_category and is_restrictive columns."""
    df = df.copy()
    df["sign_category"] = df["DESCRIPTION_RPA"].apply(classify_sign)
    df["is_restrictive"] = df["sign_category"].apply(is_restrictive)
    return df
