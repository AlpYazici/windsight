"""
Unit tests for the Open-Meteo Weather API client.

All HTTP calls are mocked — no network access is required.
"""

import math
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.api.weather import (
    WINDFM_FEATURES,
    R_D,
    air_density,
    wind_shear_exponent,
    extrapolate_wind,
    fetch_historical,
    fetch_forecast,
    geocode,
    get_elevation,
    _hourly_to_windfm,
)

# ---------------------------------------------------------------------------
# Shared mock data builders
# ---------------------------------------------------------------------------

def _make_hourly_response(n_hours: int = 24) -> dict:
    """Build a minimal Open-Meteo hourly JSON response."""
    base_time = pd.Timestamp("2025-01-01T00:00", tz="UTC")
    times = [
        (base_time + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
        for i in range(n_hours)
    ]

    # Deterministic but somewhat realistic values
    v10 = [5.0 + 0.1 * i for i in range(n_hours)]
    v80 = [7.0 + 0.15 * i for i in range(n_hours)]
    dir10 = [180.0 + i for i in range(n_hours)]
    dir80 = [185.0 + i for i in range(n_hours)]
    temp = [10.0 + 0.5 * (i % 12) for i in range(n_hours)]
    pres = [1013.25 - 0.1 * i for i in range(n_hours)]

    return {
        "hourly": {
            "time": times,
            "wind_speed_10m": v10,
            "wind_speed_80m": v80,
            "wind_direction_10m": dir10,
            "wind_direction_80m": dir80,
            "temperature_2m": temp,
            "surface_pressure": pres,
        }
    }


def _mock_response(json_data: dict, status_code: int = 200):
    """Create a ``requests.Response``-like mock."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAirDensity(unittest.TestCase):
    """Test air density calculation: rho = (P_hPa * 100) / (R_d * T_K)."""

    def test_standard_conditions(self):
        """At 1013.25 hPa and 15 degC, density ~ 1.225 kg/m^3."""
        rho = air_density(np.array([1013.25]), np.array([15.0]))
        self.assertAlmostEqual(rho[0], 1.225, places=2)

    def test_hot_low_pressure(self):
        """Higher temperature and lower pressure should give lower density."""
        rho_standard = air_density(np.array([1013.25]), np.array([15.0]))
        rho_hot_low = air_density(np.array([950.0]), np.array([35.0]))
        self.assertGreater(rho_standard[0], rho_hot_low[0])

    def test_vectorised(self):
        """Air density should work on multi-element arrays."""
        p = np.array([1013.25, 950.0, 1050.0])
        t = np.array([15.0, 30.0, -5.0])
        rho = air_density(p, t)
        self.assertEqual(rho.shape, (3,))
        # All densities should be positive
        self.assertTrue(np.all(rho > 0))

    def test_formula_exact(self):
        """Verify the exact formula."""
        p_hpa = 1000.0
        t_c = 20.0
        expected = (p_hpa * 100.0) / (R_D * (t_c + 273.15))
        result = air_density(np.array([p_hpa]), np.array([t_c]))
        self.assertAlmostEqual(result[0], expected, places=10)


class TestWindShearExponent(unittest.TestCase):
    """Test wind shear exponent: alpha = ln(v80/v10) / ln(80/10)."""

    def test_typical_values(self):
        """For v10=5, v80=7, alpha should be reasonable (0.1 – 0.3)."""
        alpha = wind_shear_exponent(np.array([5.0]), np.array([7.0]))
        self.assertGreater(alpha[0], 0.1)
        self.assertLess(alpha[0], 0.3)

    def test_formula_exact(self):
        """Verify exact calculation."""
        v10, v80 = 6.0, 9.0
        expected = math.log(v80 / v10) / math.log(80.0 / 10.0)
        alpha = wind_shear_exponent(np.array([v10]), np.array([v80]))
        self.assertAlmostEqual(alpha[0], expected, places=10)

    def test_zero_v10_uses_default(self):
        """When v10 is 0, alpha should fall back to 1/7."""
        alpha = wind_shear_exponent(np.array([0.0]), np.array([5.0]))
        self.assertAlmostEqual(alpha[0], 1.0 / 7.0, places=5)

    def test_zero_both_uses_default(self):
        """When both speeds are 0, alpha should fall back to 1/7."""
        alpha = wind_shear_exponent(np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(alpha[0], 1.0 / 7.0, places=5)

    def test_vectorised(self):
        """Wind shear should handle arrays correctly."""
        v10 = np.array([5.0, 0.0, 8.0])
        v80 = np.array([7.0, 3.0, 10.0])
        alpha = wind_shear_exponent(v10, v80)
        self.assertEqual(alpha.shape, (3,))
        # First and third should be computed, second should be default
        self.assertAlmostEqual(alpha[1], 1.0 / 7.0, places=5)


class TestExtrapolateWind(unittest.TestCase):
    """Test power-law wind extrapolation: v_hub = v_ref * (h_hub / h_ref) ^ alpha."""

    def test_same_height(self):
        """At the reference height, speed should be unchanged."""
        v_ref = np.array([8.0])
        alpha = np.array([0.15])
        v_hub = extrapolate_wind(v_ref, 10.0, 10.0, alpha)
        self.assertAlmostEqual(v_hub[0], 8.0, places=10)

    def test_higher_hub(self):
        """Higher hub should give higher wind speed (positive alpha)."""
        v_ref = np.array([5.0])
        alpha = np.array([0.2])
        v_hub = extrapolate_wind(v_ref, 10.0, 100.0, alpha)
        self.assertGreater(v_hub[0], 5.0)

    def test_formula_exact(self):
        """Verify exact calculation."""
        v_ref = np.array([6.0])
        alpha = np.array([0.14])
        h_ref, h_hub = 10.0, 80.0
        expected = 6.0 * (80.0 / 10.0) ** 0.14
        result = extrapolate_wind(v_ref, h_ref, h_hub, alpha)
        self.assertAlmostEqual(result[0], expected, places=10)


class TestHourlyToWindFM(unittest.TestCase):
    """Test the internal converter from Open-Meteo JSON to WindFM DataFrame."""

    def test_output_columns(self):
        """DataFrame should have 'time' plus all 6 WindFM feature columns."""
        data = _make_hourly_response(24)
        df = _hourly_to_windfm(data, hub_height=80.0)

        expected_cols = {"time"} | set(WINDFM_FEATURES)
        self.assertEqual(set(df.columns), expected_cols)

    def test_row_count(self):
        """Number of rows should match the number of hourly timestamps."""
        data = _make_hourly_response(48)
        df = _hourly_to_windfm(data, hub_height=80.0)
        self.assertEqual(len(df), 48)

    def test_power_is_nan(self):
        """Power column should be all NaN (filled later by pipeline)."""
        data = _make_hourly_response(12)
        df = _hourly_to_windfm(data, hub_height=80.0)
        self.assertTrue(df["power"].isna().all())

    def test_temperature_in_kelvin(self):
        """Temperature should be in Kelvin (> 200 for Earth conditions)."""
        data = _make_hourly_response(12)
        df = _hourly_to_windfm(data, hub_height=80.0)
        self.assertTrue((df["temperature"] > 200).all())

    def test_pressure_in_pascals(self):
        """Pressure should be in Pascals (order of 100 000)."""
        data = _make_hourly_response(12)
        df = _hourly_to_windfm(data, hub_height=80.0)
        self.assertTrue((df["pressure"] > 80_000).all())
        self.assertTrue((df["pressure"] < 120_000).all())

    def test_timestamps_utc(self):
        """All timestamps should be timezone-aware UTC."""
        data = _make_hourly_response(6)
        df = _hourly_to_windfm(data, hub_height=80.0)
        self.assertIsNotNone(df["time"].dt.tz)
        self.assertEqual(str(df["time"].dt.tz), "UTC")

    def test_density_positive(self):
        """Air density should be positive for all rows."""
        data = _make_hourly_response(24)
        df = _hourly_to_windfm(data, hub_height=80.0)
        self.assertTrue((df["density"] > 0).all())

    def test_hub_height_80_uses_v80(self):
        """When hub height is 80 m, wind_speed should equal v80 directly."""
        data = _make_hourly_response(6)
        df = _hourly_to_windfm(data, hub_height=80.0)
        v80 = np.array(data["hourly"]["wind_speed_80m"][:6], dtype=np.float64)
        np.testing.assert_array_almost_equal(df["wind_speed"].values, v80)

    def test_hub_height_extrapolation(self):
        """When hub height differs from 10 m and 80 m, extrapolation is used."""
        data = _make_hourly_response(6)
        df = _hourly_to_windfm(data, hub_height=120.0)
        # At 120 m, wind should be higher than at 80 m (positive shear)
        v80 = np.array(data["hourly"]["wind_speed_80m"][:6], dtype=np.float64)
        self.assertTrue((df["wind_speed"].values >= v80 - 0.01).all())


class TestFetchHistorical(unittest.TestCase):
    """Test fetch_historical with mocked HTTP."""

    @patch("src.api.weather._get_json")
    def test_returns_dataframe(self, mock_get_json):
        mock_get_json.return_value = _make_hourly_response(24)

        df = fetch_historical(52.0, 13.0, days=30, hub_height=80.0)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 24)
        expected_cols = {"time"} | set(WINDFM_FEATURES)
        self.assertEqual(set(df.columns), expected_cols)

    @patch("src.api.weather._get_json")
    def test_passes_correct_dates(self, mock_get_json):
        mock_get_json.return_value = _make_hourly_response(24)

        fetch_historical(52.0, 13.0, days=10, hub_height=80.0)

        # Verify _get_json was called (may be via the cached wrapper)
        self.assertTrue(mock_get_json.called)

    @patch("src.api.weather._get_json")
    def test_windfm_format(self, mock_get_json):
        """Verify all WindFM features are present and power is NaN."""
        mock_get_json.return_value = _make_hourly_response(12)

        df = fetch_historical(40.0, 29.0, days=7, hub_height=100.0)

        for feat in WINDFM_FEATURES:
            self.assertIn(feat, df.columns)
        self.assertTrue(df["power"].isna().all())


class TestFetchForecast(unittest.TestCase):
    """Test fetch_forecast with mocked HTTP."""

    @patch("src.api.weather._get_json")
    def test_returns_dataframe(self, mock_get_json):
        mock_get_json.return_value = _make_hourly_response(168)  # 7 days

        df = fetch_forecast(52.0, 13.0, hub_height=80.0)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 168)

    @patch("src.api.weather._get_json")
    def test_power_is_nan(self, mock_get_json):
        mock_get_json.return_value = _make_hourly_response(48)

        df = fetch_forecast(41.0, 29.0, hub_height=80.0)

        self.assertTrue(df["power"].isna().all())


class TestGeocode(unittest.TestCase):
    """Test geocode with mocked HTTP."""

    @patch("src.api.weather._get_json")
    def test_returns_results(self, mock_get_json):
        mock_get_json.return_value = {
            "results": [
                {
                    "name": "Istanbul",
                    "latitude": 41.0082,
                    "longitude": 28.9784,
                    "country": "Turkey",
                    "elevation": 39.0,
                }
            ]
        }
        # Clear the lru_cache so our mock is hit
        geocode.cache_clear()

        results = geocode("Istanbul")

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["latitude"], 41.0082)
        self.assertEqual(results[0]["country"], "Turkey")

    @patch("src.api.weather._get_json")
    def test_no_results_raises(self, mock_get_json):
        mock_get_json.return_value = {"results": []}
        geocode.cache_clear()

        with self.assertRaises(ValueError):
            geocode("XYZNONEXISTENT999")


class TestGetElevation(unittest.TestCase):
    """Test get_elevation with mocked HTTP."""

    @patch("src.api.weather._get_json")
    def test_returns_float(self, mock_get_json):
        mock_get_json.return_value = {"elevation": [542.0]}
        get_elevation.cache_clear()

        elev = get_elevation(47.0, 8.0)

        self.assertIsInstance(elev, float)
        self.assertAlmostEqual(elev, 542.0)


if __name__ == "__main__":
    unittest.main()
