"""Turbine specifications database.

Loads turbine power-curve data from ``data/turbine_specs.json`` and exposes
helpers for lookup, listing, and power estimation (scalar and vectorised).
"""

from __future__ import annotations

import json
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path to the JSON database (relative to repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
TURBINE_SPECS_PATH: Path = _REPO_ROOT / "data" / "turbine_specs.json"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerCurvePoint:
    """Single point on a turbine power curve."""

    wind_speed: float  # m/s
    power_kw: float


@dataclass(frozen=True)
class TurbineSpec:
    """Complete specification for a single wind-turbine model."""

    manufacturer: str
    model: str
    rated_power_kw: float
    rotor_diameter_m: float
    hub_height_m: float
    cut_in_speed_ms: float
    rated_speed_ms: float
    cut_out_speed_ms: float
    swept_area_m2: float
    power_curve: list[PowerCurvePoint] = field(default_factory=list)

    # Convenience -----------------------------------------------------------

    @property
    def full_name(self) -> str:
        """Return ``'Manufacturer Model'`` string."""
        return f"{self.manufacturer} {self.model}"

    @property
    def power_curve_wind_speeds(self) -> np.ndarray:
        """Wind speeds from the power curve as a 1-D array."""
        return np.array([p.wind_speed for p in self.power_curve])

    @property
    def power_curve_powers(self) -> np.ndarray:
        """Power values from the power curve as a 1-D array (kW)."""
        return np.array([p.power_kw for p in self.power_curve])


# ---------------------------------------------------------------------------
# Loading & caching
# ---------------------------------------------------------------------------

def _parse_turbine(entry: dict) -> TurbineSpec:
    """Parse a single turbine dict from the JSON file."""
    curve = [
        PowerCurvePoint(wind_speed=pt["wind_speed"], power_kw=pt["power_kw"])
        for pt in entry["power_curve"]
    ]
    return TurbineSpec(
        manufacturer=entry["manufacturer"],
        model=entry["model"],
        rated_power_kw=entry["rated_power_kw"],
        rotor_diameter_m=entry["rotor_diameter_m"],
        hub_height_m=entry["hub_height_m"],
        cut_in_speed_ms=entry["cut_in_speed_ms"],
        rated_speed_ms=entry["rated_speed_ms"],
        cut_out_speed_ms=entry["cut_out_speed_ms"],
        swept_area_m2=entry["swept_area_m2"],
        power_curve=curve,
    )


@functools.lru_cache(maxsize=1)
def load_specs(path: Optional[str] = None) -> dict[str, TurbineSpec]:
    """Load turbine specifications from JSON and return a dict keyed by full
    model name (``'Manufacturer Model'``).

    Results are cached after the first call.
    """
    specs_path = Path(path) if path else TURBINE_SPECS_PATH
    with open(specs_path, "r") as fh:
        raw = json.load(fh)

    specs: dict[str, TurbineSpec] = {}
    for entry in raw["turbines"]:
        ts = _parse_turbine(entry)
        specs[ts.full_name] = ts
    return specs


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return " ".join(text.lower().split())


def list_turbines() -> list[str]:
    """Return a sorted list of all turbine full-names in the database."""
    return sorted(load_specs().keys())


def get_turbine(name: str) -> TurbineSpec:
    """Look up a turbine by name with case-insensitive fuzzy matching.

    Matching strategy (in order):
    1. Exact match (case-insensitive).
    2. The query is a substring of a full name.
    3. All tokens in the query appear somewhere in a full name.

    Raises ``KeyError`` if no match is found.
    """
    specs = load_specs()
    query = _normalise(name)

    # 1. Exact (case-insensitive)
    for key, spec in specs.items():
        if _normalise(key) == query:
            return spec

    # 2. Substring
    matches = [
        (key, spec) for key, spec in specs.items() if query in _normalise(key)
    ]
    if len(matches) == 1:
        return matches[0][1]

    # 3. Token overlap
    query_tokens = set(query.split())
    token_matches = []
    for key, spec in specs.items():
        key_norm = _normalise(key)
        if query_tokens.issubset(set(key_norm.split())):
            token_matches.append((key, spec))
    if len(token_matches) == 1:
        return token_matches[0][1]

    # Build informative error
    candidates = matches or token_matches
    if candidates:
        names = [c[0] for c in candidates]
        raise KeyError(
            f"Ambiguous turbine query '{name}'. Matches: {names}"
        )
    raise KeyError(
        f"No turbine found matching '{name}'. "
        f"Available: {list_turbines()}"
    )


# ---------------------------------------------------------------------------
# Power estimation (interpolation from power curve)
# ---------------------------------------------------------------------------

def estimate_power(name: str, wind_speed: float) -> float:
    """Estimate power output (kW) for a single wind speed by linearly
    interpolating the turbine's power curve.

    Wind speeds outside the curve range return 0.
    """
    spec = get_turbine(name)
    ws = spec.power_curve_wind_speeds
    pw = spec.power_curve_powers
    return float(np.interp(wind_speed, ws, pw, left=0.0, right=0.0))


def estimate_power_array(
    name: str, wind_speeds: np.ndarray
) -> np.ndarray:
    """Vectorised power estimation (kW) for an array of wind speeds.

    Uses ``numpy.interp`` under the hood; wind speeds outside the power-curve
    range are mapped to 0.
    """
    spec = get_turbine(name)
    ws = spec.power_curve_wind_speeds
    pw = spec.power_curve_powers
    return np.interp(wind_speeds, ws, pw, left=0.0, right=0.0)
