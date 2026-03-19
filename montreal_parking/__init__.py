"""Montreal Free Parking Finder."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_PIXI_TOML = Path(__file__).resolve().parent.parent / "pixi.toml"


def _read_version() -> str:
    try:
        data = tomllib.loads(_PIXI_TOML.read_text())
        return str(data["workspace"]["version"])
    except Exception:
        return "unknown"


__version__: str = _read_version()
