"""Synthetic power generation helpers.

Provides functions to generate synthetic power output from wind speed data
using turbine power curves, and to apply physical constraints (cut-in,
cut-out, rated power clipping) to power predictions.
"""

from __future__ import annotations

import numpy as np

from src.data.turbine_db import get_turbine, estimate_power_array


def generate_synthetic_power(
    wind_speeds: np.ndarray,
    turbine_name: str,
) -> np.ndarray:
    """Generate synthetic power output (MW) from wind speeds using a turbine power curve.

    Parameters
    ----------
    wind_speeds : np.ndarray
        Array of wind speeds in m/s.
    turbine_name : str
        Turbine model name (passed to ``turbine_db.get_turbine``).

    Returns
    -------
    np.ndarray
        Power output in **MW** (same length as *wind_speeds*).
    """
    wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
    power_kw = estimate_power_array(turbine_name, wind_speeds)
    power_mw = power_kw / 1000.0
    return power_mw


def apply_physical_constraints(
    power_kw: np.ndarray,
    wind_speeds: np.ndarray,
    turbine_name: str,
) -> np.ndarray:
    """Apply physical turbine constraints to a power prediction array.

    Constraints applied (in order):
    1. Clip negative values to zero.
    2. Clip values above the turbine's rated power to rated power.
    3. Zero out power where wind speed is below the cut-in speed.
    4. Zero out power where wind speed is above the cut-out speed.

    Parameters
    ----------
    power_kw : np.ndarray
        Predicted power values in kW (may come from WindFM or other source).
    wind_speeds : np.ndarray
        Corresponding wind speeds in m/s (same length as *power_kw*).
    turbine_name : str
        Turbine model name for spec lookup.

    Returns
    -------
    np.ndarray
        Physically constrained power values in kW.
    """
    spec = get_turbine(turbine_name)
    power_kw = np.asarray(power_kw, dtype=np.float64).copy()
    wind_speeds = np.asarray(wind_speeds, dtype=np.float64)

    # 1. No negative power
    np.clip(power_kw, 0.0, None, out=power_kw)

    # 2. Cap at rated power
    np.clip(power_kw, None, spec.rated_power_kw, out=power_kw)

    # 3. Below cut-in speed -> zero power
    below_cut_in = wind_speeds < spec.cut_in_speed_ms
    power_kw[below_cut_in] = 0.0

    # 4. Above cut-out speed -> zero power
    above_cut_out = wind_speeds > spec.cut_out_speed_ms
    power_kw[above_cut_out] = 0.0

    return power_kw
