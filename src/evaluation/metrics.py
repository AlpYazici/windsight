"""
Evaluation metrics for wind power forecasts.

Covers deterministic, probabilistic, and calibration metrics.
All functions expect numpy arrays:
    y_true   — shape (n_timesteps,)
    y_samples — shape (n_timesteps, n_samples)
"""

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Deterministic metrics (point prediction vs actual)
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Observations where y_true == 0 are excluded to avoid division by zero.
    Returns NaN if all observations are zero.
    """
    mask = y_true != 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# ---------------------------------------------------------------------------
# Probabilistic metrics (distribution of samples vs actual)
# ---------------------------------------------------------------------------

def crps(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score (sample-based).

    For each observation i with sorted samples x_1 ... x_N:
        CRPS_i = (1/N) * sum_j |x_j - y_i| - (1/(2*N^2)) * sum_{j,k} |x_j - x_k|

    Returns the mean CRPS across all observations.
    """
    n_timesteps, n_samples = y_samples.shape
    sorted_samples = np.sort(y_samples, axis=1)

    # Term 1: E|X - y| = (1/N) * sum_j |x_j - y|
    term1 = np.mean(np.abs(sorted_samples - y_true[:, None]), axis=1)

    # Term 2: E|X - X'| = (1/N^2) * sum_{j,k} |x_j - x_k|
    # Efficient computation using sorted samples:
    # sum_{j<k} |x_j - x_k| for sorted values can be computed as:
    # sum_j x_j * (2*j - N) where j is 0-indexed rank
    # Then total pairwise sum = 2 * sum_j x_j * (2*j - N + 1) ... use the identity:
    # (1/N^2) * sum_{j,k} |x_j - x_k| = (2 / N^2) * sum_{j=0}^{N-1} x_j * (2j + 1 - N)
    j_indices = np.arange(n_samples)
    weights = 2.0 * j_indices + 1.0 - n_samples  # shape (n_samples,)
    term2 = np.sum(sorted_samples * weights[None, :], axis=1) / (n_samples ** 2)

    crps_per_obs = term1 - term2
    return float(np.mean(crps_per_obs))


def aql(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    quantiles: List[float] = None,
) -> float:
    """Average Quantile Loss (pinball loss averaged across quantiles).

    For each quantile q with predicted quantile yhat_q:
        QL(q) = 2 * mean(max(q*(y - yhat_q), (1-q)*(yhat_q - y)))
    AQL = mean over all quantiles.
    """
    if quantiles is None:
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

    ql_values = []
    for q in quantiles:
        yhat_q = np.percentile(y_samples, q * 100.0, axis=1)
        diff = y_true - yhat_q
        loss = np.where(diff >= 0, q * diff, (1.0 - q) * (-diff))
        ql_values.append(2.0 * np.mean(loss))

    return float(np.mean(ql_values))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Fraction of actuals that fall within [y_lower, y_upper]."""
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(inside))


def pi_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean prediction interval width (sharpness)."""
    return float(np.mean(y_upper - y_lower))


def reliability_diagram(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reliability diagram data.

    For each nominal level in {0.1, 0.2, ..., 0.9}, compute the fraction
    of observations that fall inside the corresponding central prediction
    interval.

    Returns
    -------
    nominal_levels : ndarray of shape (n_bins-1,)
        The target coverage levels (e.g. 0.1, 0.2, ..., 0.9).
    observed_levels : ndarray of shape (n_bins-1,)
        The empirically observed coverage at each level.
    """
    nominal_levels = np.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1)
    observed_levels = np.empty_like(nominal_levels)

    for i, level in enumerate(nominal_levels):
        alpha = 1.0 - level
        lower_q = alpha / 2.0 * 100.0
        upper_q = (1.0 - alpha / 2.0) * 100.0
        y_lower = np.percentile(y_samples, lower_q, axis=1)
        y_upper = np.percentile(y_samples, upper_q, axis=1)
        observed_levels[i] = coverage(y_true, y_lower, y_upper)

    return nominal_levels, observed_levels


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compute_percentiles(
    y_samples: np.ndarray,
    percentiles: List[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute percentile arrays from forecast samples.

    Parameters
    ----------
    y_samples : ndarray of shape (n_timesteps, n_samples)
    percentiles : list of ints (e.g. [5, 25, 50, 75, 95])

    Returns
    -------
    dict mapping "P{xx}" to ndarray of shape (n_timesteps,)
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    result = {}
    for p in percentiles:
        result[f"P{p}"] = np.percentile(y_samples, p, axis=1)
    return result


def compute_all_metrics(
    y_true: np.ndarray,
    y_samples: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics at once.

    Deterministic metrics use the median (P50) of samples as the point
    prediction.

    Returns
    -------
    dict with keys:
        mae, rmse, r_squared, mape,
        crps, aql,
        coverage_90, coverage_50, pi_width_90, pi_width_50
    """
    # Point prediction = median
    y_pred = np.median(y_samples, axis=1)

    # Prediction intervals
    lower_5 = np.percentile(y_samples, 5, axis=1)
    upper_95 = np.percentile(y_samples, 95, axis=1)
    lower_25 = np.percentile(y_samples, 25, axis=1)
    upper_75 = np.percentile(y_samples, 75, axis=1)

    return {
        # Deterministic
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        # Probabilistic
        "crps": crps(y_true, y_samples),
        "aql": aql(y_true, y_samples),
        # Calibration
        "coverage_90": coverage(y_true, lower_5, upper_95),
        "coverage_50": coverage(y_true, lower_25, upper_75),
        "pi_width_90": pi_width(lower_5, upper_95),
        "pi_width_50": pi_width(lower_25, upper_75),
    }
