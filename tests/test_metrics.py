"""Tests for src.evaluation.metrics with known inputs/outputs."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    mae,
    rmse,
    r_squared,
    mape,
    crps,
    aql,
    coverage,
    pi_width,
    reliability_diagram,
    compute_all_metrics,
    compute_percentiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect_samples(y_true, n_samples=100):
    """Samples that exactly equal the true values (zero spread)."""
    return np.tile(y_true[:, None], (1, n_samples))


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

class TestMAE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 5.0])
        # errors: 1, 1, 2 => mean = 4/3
        assert np.isclose(mae(y_true, y_pred), 4.0 / 3.0)

    def test_symmetric(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        assert np.isclose(mae(y_true, y_pred), 1.0)


class TestRMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 5.0])
        # squared errors: 1, 1, 4 => mean = 2 => rmse = sqrt(2)
        assert np.isclose(rmse(y_true, y_pred), np.sqrt(2.0))

    def test_rmse_ge_mae(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 200)
        y_pred = rng.normal(0, 1, 200)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestRSquared:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(r_squared(y, y), 1.0)

    def test_mean_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, 2.0)  # predict the mean
        assert np.isclose(r_squared(y_true, y_pred), 0.0)

    def test_negative_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])
        assert r_squared(y_true, y_pred) < 0.0

    def test_constant_true(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        # ss_tot = 0, function returns 0.0 by convention
        assert r_squared(y_true, y_pred) == 0.0


class TestMAPE:
    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # pct errors: 10%, 10% => mean = 10%
        assert np.isclose(mape(y_true, y_pred), 10.0)

    def test_zero_true_excluded(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        # only second element counted: 10%
        assert np.isclose(mape(y_true, y_pred), 10.0)

    def test_all_zero_returns_nan(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        assert np.isnan(mape(y_true, y_pred))


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

class TestCRPS:
    def test_perfect_deterministic_samples(self):
        """When all samples equal y_true, CRPS should be 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_samples = _make_perfect_samples(y_true)
        assert np.isclose(crps(y_true, y_samples), 0.0, atol=1e-10)

    def test_known_two_sample(self):
        """Manual calculation with 2 samples per observation.

        y_true = [0], samples = [-1, 1]
        term1 = mean(|(-1)-0|, |1-0|) = 1.0
        sorted: [-1, 1], weights = [2*0+1-2, 2*1+1-2] = [-1, 1]
        term2 = ((-1)*(-1) + 1*1) / 4 = 2/4 = 0.5
        CRPS = 1.0 - 0.5 = 0.5
        """
        y_true = np.array([0.0])
        y_samples = np.array([[-1.0, 1.0]])
        assert np.isclose(crps(y_true, y_samples), 0.5)

    def test_wider_spread_higher_crps(self):
        """Wider sample spread around truth should give higher CRPS."""
        rng = np.random.default_rng(0)
        y_true = np.zeros(500)
        narrow = rng.normal(0, 0.1, (500, 200))
        wide = rng.normal(0, 10.0, (500, 200))
        assert crps(y_true, narrow) < crps(y_true, wide)

    def test_biased_higher_crps(self):
        """Biased samples should score worse than centred samples."""
        rng = np.random.default_rng(1)
        y_true = np.zeros(500)
        centred = rng.normal(0, 1, (500, 200))
        biased = rng.normal(5, 1, (500, 200))
        assert crps(y_true, centred) < crps(y_true, biased)


class TestAQL:
    def test_perfect_samples(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_samples = _make_perfect_samples(y_true)
        assert np.isclose(aql(y_true, y_samples), 0.0, atol=1e-10)

    def test_positive(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(0, 1, 100)
        y_samples = rng.normal(0, 1, (100, 50))
        assert aql(y_true, y_samples) > 0.0

    def test_custom_quantiles(self):
        rng = np.random.default_rng(3)
        y_true = rng.normal(0, 1, 100)
        y_samples = rng.normal(0, 1, (100, 50))
        result = aql(y_true, y_samples, quantiles=[0.5])
        # With only the median quantile, result equals 2 * MAE of median
        y_median = np.median(y_samples, axis=1)
        expected = 2.0 * np.mean(
            np.where(
                y_true >= y_median,
                0.5 * (y_true - y_median),
                0.5 * (y_median - y_true),
            )
        )
        assert np.isclose(result, expected)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_full_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = np.array([0.0, 0.0, 0.0])
        y_upper = np.array([5.0, 5.0, 5.0])
        assert coverage(y_true, y_lower, y_upper) == 1.0

    def test_no_coverage(self):
        y_true = np.array([10.0, 20.0])
        y_lower = np.array([0.0, 0.0])
        y_upper = np.array([1.0, 1.0])
        assert coverage(y_true, y_lower, y_upper) == 0.0

    def test_partial_coverage(self):
        y_true = np.array([1.0, 5.0, 10.0, 15.0])
        y_lower = np.array([0.0, 0.0, 0.0, 0.0])
        y_upper = np.array([2.0, 6.0, 8.0, 12.0])
        # first two inside, last two outside => 50%
        assert np.isclose(coverage(y_true, y_lower, y_upper), 0.5)

    def test_boundary_included(self):
        y_true = np.array([0.0, 5.0])
        y_lower = np.array([0.0, 3.0])
        y_upper = np.array([3.0, 5.0])
        assert coverage(y_true, y_lower, y_upper) == 1.0


class TestPIWidth:
    def test_known_width(self):
        y_lower = np.array([1.0, 2.0, 3.0])
        y_upper = np.array([3.0, 4.0, 9.0])
        # widths: 2, 2, 6 => mean = 10/3
        assert np.isclose(pi_width(y_lower, y_upper), 10.0 / 3.0)

    def test_zero_width(self):
        v = np.array([1.0, 2.0])
        assert pi_width(v, v) == 0.0


class TestReliabilityDiagram:
    def test_perfect_calibration(self):
        """Samples drawn from the same distribution as y_true should be
        approximately well-calibrated (with enough data)."""
        rng = np.random.default_rng(42)
        n = 10000
        # Each observation comes from N(mu_i, 1) where mu_i varies
        mus = rng.uniform(-3, 3, n)
        y_true = mus + rng.normal(0, 1, n)
        # Samples are drawn from the same conditional distribution
        y_samples = mus[:, None] + rng.normal(0, 1, (n, 500))

        nominal, observed = reliability_diagram(y_true, y_samples)
        # Each observed level should be within ~3 percentage points of nominal
        assert np.allclose(observed, nominal, atol=0.03)

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 50)
        y_samples = rng.normal(0, 1, (50, 30))
        nominal, observed = reliability_diagram(y_true, y_samples, n_bins=5)
        assert nominal.shape == (4,)
        assert observed.shape == (4,)

    def test_nominal_levels_values(self):
        nominal, _ = reliability_diagram(
            np.zeros(10), np.zeros((10, 20)), n_bins=10
        )
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        assert np.allclose(nominal, expected)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestComputePercentiles:
    def test_keys_and_shapes(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 1, (20, 100))
        result = compute_percentiles(samples)
        assert set(result.keys()) == {"P5", "P25", "P50", "P75", "P95"}
        for v in result.values():
            assert v.shape == (20,)

    def test_ordering(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(0, 1, (50, 200))
        p = compute_percentiles(samples)
        assert np.all(p["P5"] <= p["P25"])
        assert np.all(p["P25"] <= p["P50"])
        assert np.all(p["P50"] <= p["P75"])
        assert np.all(p["P75"] <= p["P95"])

    def test_custom_percentiles(self):
        rng = np.random.default_rng(2)
        samples = rng.normal(0, 1, (10, 50))
        result = compute_percentiles(samples, percentiles=[10, 90])
        assert set(result.keys()) == {"P10", "P90"}


class TestComputeAllMetrics:
    def test_keys_present(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 100)
        y_samples = rng.normal(0, 1, (100, 50))
        result = compute_all_metrics(y_true, y_samples)
        expected_keys = {
            "mae", "rmse", "r_squared", "mape",
            "crps", "aql",
            "coverage_90", "coverage_50", "pi_width_90", "pi_width_50",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(0, 1, 100)
        y_samples = rng.normal(0, 1, (100, 50))
        result = compute_all_metrics(y_true, y_samples)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_perfect_forecast(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_samples = _make_perfect_samples(y_true, n_samples=200)
        result = compute_all_metrics(y_true, y_samples)
        assert np.isclose(result["mae"], 0.0)
        assert np.isclose(result["rmse"], 0.0)
        assert np.isclose(result["crps"], 0.0, atol=1e-10)
        assert result["coverage_90"] == 1.0
        assert result["coverage_50"] == 1.0
        assert np.isclose(result["pi_width_90"], 0.0)
        assert np.isclose(result["pi_width_50"], 0.0)

    def test_coverage_bounds(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(0, 1, 200)
        y_samples = rng.normal(0, 1, (200, 100))
        result = compute_all_metrics(y_true, y_samples)
        assert 0.0 <= result["coverage_90"] <= 1.0
        assert 0.0 <= result["coverage_50"] <= 1.0
        assert result["pi_width_90"] >= result["pi_width_50"]
