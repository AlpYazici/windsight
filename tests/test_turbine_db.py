"""Tests for the turbine specifications database (src/data/turbine_db)."""

import math

import numpy as np
import pytest

from src.data.turbine_db import (
    TURBINE_SPECS_PATH,
    TurbineSpec,
    estimate_power,
    estimate_power_array,
    get_turbine,
    list_turbines,
    load_specs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def specs() -> dict[str, TurbineSpec]:
    """Load all turbine specs once for the module."""
    return load_specs()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadSpecs:
    def test_specs_file_exists(self):
        assert TURBINE_SPECS_PATH.exists(), (
            f"Specs file not found at {TURBINE_SPECS_PATH}"
        )

    def test_loads_all_turbines(self, specs):
        assert len(specs) >= 13

    def test_values_are_turbine_spec(self, specs):
        for ts in specs.values():
            assert isinstance(ts, TurbineSpec)

    def test_power_curve_not_empty(self, specs):
        for ts in specs.values():
            assert len(ts.power_curve) > 0, f"{ts.full_name} has empty curve"

    def test_swept_area_consistent(self, specs):
        for ts in specs.values():
            expected = math.pi * (ts.rotor_diameter_m / 2) ** 2
            assert abs(ts.swept_area_m2 - expected) < 0.1, (
                f"{ts.full_name}: swept_area mismatch"
            )


# ---------------------------------------------------------------------------
# Lookup & listing
# ---------------------------------------------------------------------------


class TestLookup:
    def test_list_turbines_returns_strings(self):
        names = list_turbines()
        assert len(names) >= 13
        assert all(isinstance(n, str) for n in names)

    def test_list_turbines_sorted(self):
        names = list_turbines()
        assert names == sorted(names)

    def test_get_turbine_exact(self):
        ts = get_turbine("Vestas V90-2.0")
        assert ts.rated_power_kw == 2000
        assert ts.rotor_diameter_m == 90

    def test_get_turbine_case_insensitive(self):
        ts = get_turbine("vestas v90-2.0")
        assert ts.model == "V90-2.0"

    def test_get_turbine_substring(self):
        ts = get_turbine("E-126 EP4")
        assert ts.manufacturer == "Enercon"

    def test_get_turbine_token_match(self):
        ts = get_turbine("GE 1.5sle")
        assert ts.rated_power_kw == 1500

    def test_get_turbine_not_found(self):
        with pytest.raises(KeyError, match="No turbine found"):
            get_turbine("FakeTurbine XYZ-9000")


# ---------------------------------------------------------------------------
# Power estimation
# ---------------------------------------------------------------------------


class TestEstimatePower:
    def test_zero_below_cut_in(self):
        # All turbines have cut-in >= 2.5, so 0 m/s should yield 0
        for name in list_turbines():
            assert estimate_power(name, 0.0) == 0.0

    def test_zero_above_cut_out(self):
        ts = get_turbine("Vestas V90-2.0")
        # Well above cut-out (25 m/s)
        assert estimate_power("Vestas V90-2.0", 50.0) == 0.0

    def test_rated_at_rated_speed(self):
        ts = get_turbine("Vestas V90-2.0")
        power = estimate_power("Vestas V90-2.0", ts.rated_speed_ms)
        # Should be at or very close to rated power
        assert abs(power - ts.rated_power_kw) < 1.0

    def test_between_cutin_and_rated_is_positive(self):
        ts = get_turbine("Vestas V90-2.0")
        mid = (ts.cut_in_speed_ms + ts.rated_speed_ms) / 2
        power = estimate_power("Vestas V90-2.0", mid)
        assert 0 < power < ts.rated_power_kw

    def test_monotonically_increasing_cutin_to_rated(self):
        """Power should increase (or stay equal) between cut-in and rated."""
        ts = get_turbine("Siemens SWT-2.3-93")
        speeds = np.arange(ts.cut_in_speed_ms, ts.rated_speed_ms, 0.5)
        powers = [estimate_power("Siemens SWT-2.3-93", s) for s in speeds]
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 0.01  # small tolerance

    def test_flat_at_rated(self):
        """Power should be constant between rated and cut-out."""
        ts = get_turbine("GE 2.5-120")
        speeds = np.arange(ts.rated_speed_ms, ts.cut_out_speed_ms, 0.5)
        powers = [estimate_power("GE 2.5-120", s) for s in speeds]
        for p in powers:
            assert abs(p - ts.rated_power_kw) < 1.0


class TestEstimatePowerArray:
    def test_returns_ndarray(self):
        ws = np.array([0.0, 5.0, 10.0, 15.0, 30.0])
        result = estimate_power_array("Vestas V90-2.0", ws)
        assert isinstance(result, np.ndarray)
        assert result.shape == ws.shape

    def test_consistent_with_scalar(self):
        name = "Senvion MM92"
        ws = np.arange(0, 30, 0.5)
        arr = estimate_power_array(name, ws)
        for i, v in enumerate(ws):
            scalar = estimate_power(name, float(v))
            assert abs(arr[i] - scalar) < 0.01

    def test_empty_array(self):
        result = estimate_power_array("GE 1.5sle", np.array([]))
        assert len(result) == 0

    def test_all_zeros_outside_range(self):
        ws = np.array([-5.0, 0.0, 1.0, 50.0, 100.0])
        result = estimate_power_array("Nordex N100/2500", ws)
        np.testing.assert_array_equal(result, 0.0)
