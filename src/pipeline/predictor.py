"""End-to-end wind power prediction pipeline.

Takes a location (lat/lon or city name) and a turbine model, fetches weather
data, generates synthetic power for the historical window, runs WindFM
inference, post-processes the results, and returns a structured
``ForecastResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.api.weather import fetch_weather, geocode, get_elevation
from src.data.turbine_db import get_turbine, TurbineSpec
from src.models.windfm_wrapper import WindFMWrapper
from src.pipeline.synthetic_power import generate_synthetic_power, apply_physical_constraints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HISTORY_DAYS = 10
FORECAST_HOURS = 168  # 7 days
MAX_CONTEXT = 512  # WindFM maximum context window
SAMPLE_COUNT = 1
TEMPERATURE = 1.0
TOP_P = 1.0
PERCENTILES = (5, 25, 50, 75, 95)

WINDFM_FEATURES = [
    "wind_speed", "wind_direction", "power",
    "density", "temperature", "pressure",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    """Container for a complete wind-power forecast."""

    location: dict  # lat, lon, elevation_m, city (if geocoded)
    turbine: dict  # model, manufacturer, rated_power_kw, hub_height_m
    timestamps: list  # UTC timestamps for each forecast hour
    power_kw: dict  # P5, P25, P50, P75, P95 arrays (in kW)
    daily_energy_mwh: list  # 7 values, one per day
    capacity_factor: float  # overall capacity factor
    weather: dict  # wind_speed, wind_direction, temperature arrays (for display)
    generated_at: str  # ISO timestamp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_location_info(
    lat: float,
    lon: float,
    city: Optional[str] = None,
) -> dict:
    """Build the location metadata dict, fetching elevation."""
    try:
        elevation = get_elevation(lat, lon)
    except Exception:
        logger.warning("Could not fetch elevation for (%.4f, %.4f); defaulting to 0.", lat, lon)
        elevation = 0.0

    return {
        "lat": lat,
        "lon": lon,
        "elevation_m": elevation,
        "city": city,
    }


def _build_turbine_info(spec: TurbineSpec) -> dict:
    """Extract the fields we expose in ForecastResult.turbine."""
    return {
        "model": spec.model,
        "manufacturer": spec.manufacturer,
        "rated_power_kw": spec.rated_power_kw,
        "hub_height_m": spec.hub_height_m,
    }


def _split_history_forecast(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a weather DataFrame into history (past) and forecast (future).

    The split point is the current UTC time rounded down to the nearest hour.
    """
    now = pd.Timestamp.now(tz="UTC").floor("h")
    history = df[df["time"] <= now].copy()
    forecast = df[df["time"] > now].copy()
    return history, forecast


def _fill_synthetic_power(
    df: pd.DataFrame,
    turbine_name: str,
) -> pd.DataFrame:
    """Fill the ``power`` column with synthetic power from the turbine curve.

    Power is stored in **MW** to match WindFM conventions.
    """
    df = df.copy()
    power_mw = generate_synthetic_power(df["wind_speed"].values, turbine_name)
    df["power"] = power_mw
    return df


def _prepare_context(
    history: pd.DataFrame,
    max_len: int = MAX_CONTEXT,
) -> pd.DataFrame:
    """Trim history to at most *max_len* most-recent rows."""
    if len(history) > max_len:
        return history.iloc[-max_len:].reset_index(drop=True)
    return history.reset_index(drop=True)


def _generate_forecast_timestamps(
    last_history_time: pd.Timestamp,
    hours: int = FORECAST_HOURS,
) -> pd.DatetimeIndex:
    """Create hourly timestamps starting one hour after *last_history_time*."""
    return pd.date_range(
        start=last_history_time + pd.Timedelta(hours=1),
        periods=hours,
        freq="h",
        tz="UTC",
    )


def _compute_percentiles(
    samples: np.ndarray,
    percentiles: tuple[int, ...] = PERCENTILES,
) -> dict[str, np.ndarray]:
    """Compute named percentiles across the sample axis.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(pred_len, sample_count)``.
    percentiles : tuple[int, ...]
        Which percentiles to compute.

    Returns
    -------
    dict
        Keys like ``"P5"``, ``"P50"``, etc., each mapping to a 1-D array of
        length *pred_len*.
    """
    result = {}
    for p in percentiles:
        result[f"P{p}"] = np.percentile(samples, p, axis=1)
    return result


def _compute_daily_energy(
    power_kw_p50: np.ndarray,
    hours_per_step: float = 1.0,
) -> list[float]:
    """Compute daily energy in MWh from hourly P50 power (kW).

    Expects 168 hourly values (7 days). Returns 7 daily totals.
    """
    n_days = len(power_kw_p50) // 24
    daily = []
    for d in range(n_days):
        start = d * 24
        end = start + 24
        chunk = power_kw_p50[start:end]
        energy_kwh = np.sum(chunk) * hours_per_step
        energy_mwh = energy_kwh / 1000.0
        daily.append(round(float(energy_mwh), 3))
    # Handle remaining hours that don't fill a complete day
    remaining = len(power_kw_p50) - n_days * 24
    if remaining > 0:
        chunk = power_kw_p50[n_days * 24:]
        energy_mwh = float(np.sum(chunk) * hours_per_step) / 1000.0
        daily.append(round(energy_mwh, 3))
    return daily


def _compute_capacity_factor(
    power_kw_p50: np.ndarray,
    rated_power_kw: float,
) -> float:
    """Capacity factor = mean(power) / rated_power."""
    if rated_power_kw <= 0:
        return 0.0
    return round(float(np.mean(power_kw_p50) / rated_power_kw), 4)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def predict(
    lat: float,
    lon: float,
    turbine_model: str,
    model_path: Optional[str] = None,
    city_name: Optional[str] = None,
) -> ForecastResult:
    """Run the full prediction pipeline.

    Parameters
    ----------
    lat, lon : float
        Location coordinates (WGS-84).
    turbine_model : str
        Name of the turbine model (looked up via ``turbine_db.get_turbine``).
    model_path : str, optional
        Path to a fine-tuned WindFM checkpoint. When *None*, the default
        pre-trained weights are used.
    city_name : str, optional
        If the call originates from ``predict_from_city``, this is passed
        through so the result includes the city name.

    Returns
    -------
    ForecastResult
    """
    logger.info(
        "Starting prediction pipeline: lat=%.4f, lon=%.4f, turbine=%s",
        lat, lon, turbine_model,
    )

    # ---- 1. Turbine specs --------------------------------------------------
    spec = get_turbine(turbine_model)
    logger.info("Turbine: %s (rated %.0f kW, hub %.0f m)", spec.full_name, spec.rated_power_kw, spec.hub_height_m)

    # ---- 2. Fetch weather ---------------------------------------------------
    logger.info("Fetching weather data (%d days history + 7 day forecast)...", HISTORY_DAYS)
    try:
        weather_df = fetch_weather(lat, lon, history_days=HISTORY_DAYS, hub_height=spec.hub_height_m)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch weather data: {exc}") from exc

    if weather_df.empty:
        raise RuntimeError("Weather API returned an empty DataFrame.")

    # ---- 3. Split into history / forecast -----------------------------------
    history, forecast = _split_history_forecast(weather_df)

    if history.empty:
        raise RuntimeError("No historical weather data available. Cannot run inference.")

    # Ensure we have at least some forecast timestamps
    if forecast.empty or len(forecast) < FORECAST_HOURS:
        # Generate the remaining timestamps from the end of history
        last_hist_time = history["time"].iloc[-1]
        needed = FORECAST_HOURS
        all_fc_times = _generate_forecast_timestamps(last_hist_time, needed)
        if not forecast.empty:
            # Keep existing forecast rows, pad the rest
            existing_times = set(forecast["time"])
            missing_times = [t for t in all_fc_times if t not in existing_times]
            if missing_times:
                pad = pd.DataFrame({"time": missing_times})
                for col in WINDFM_FEATURES:
                    pad[col] = np.nan
                forecast = pd.concat([forecast, pad], ignore_index=True)
                forecast = forecast.sort_values("time").reset_index(drop=True)
        else:
            forecast = pd.DataFrame({"time": all_fc_times})
            for col in WINDFM_FEATURES:
                forecast[col] = np.nan

    # Trim forecast to exactly FORECAST_HOURS
    forecast = forecast.iloc[:FORECAST_HOURS].copy()

    # ---- 4 & 5. Synthetic power for history ---------------------------------
    history = _fill_synthetic_power(history, turbine_model)

    # ---- 6. Synthetic power for forecast (initial estimate) -----------------
    # For any NaN weather columns in the forecast, forward-fill from history
    full_df = pd.concat([history, forecast], ignore_index=True)
    full_df[WINDFM_FEATURES] = full_df[WINDFM_FEATURES].ffill()
    full_df[WINDFM_FEATURES] = full_df[WINDFM_FEATURES].bfill()
    forecast = full_df.iloc[len(history):].copy().reset_index(drop=True)
    forecast = _fill_synthetic_power(forecast, turbine_model)

    # Re-split after filling
    history_filled = full_df.iloc[:len(history)].copy()
    history_filled["power"] = history["power"].values

    # ---- 7. Load WindFM model -----------------------------------------------
    logger.info("Loading WindFM model...")
    try:
        wrapper = WindFMWrapper(model_path=model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load WindFM model: {exc}") from exc

    # ---- 8. Run WindFM inference --------------------------------------------
    context_df = _prepare_context(history_filled, MAX_CONTEXT)
    x_timestamps = pd.Series(pd.to_datetime(context_df["time"])).reset_index(drop=True)
    y_timestamps = pd.Series(pd.to_datetime(forecast["time"])).reset_index(drop=True)

    # Ensure y_timestamps has exactly FORECAST_HOURS entries
    y_timestamps = y_timestamps[:FORECAST_HOURS]

    logger.info(
        "Running inference: context=%d hours, prediction=%d hours, samples=%d",
        len(context_df), len(y_timestamps), SAMPLE_COUNT,
    )

    # The predictor expects a DataFrame with the 6 feature columns (no NaNs)
    input_df = context_df[WINDFM_FEATURES].copy()

    # Verify no NaNs remain
    if input_df.isnull().any().any():
        logger.warning("NaN values found in input features; forward-filling.")
        input_df = input_df.ffill().bfill().fillna(0.0)

    pred_df = wrapper.predict(
        df=input_df,
        x_timestamp=x_timestamps,
        y_timestamp=y_timestamps,
        pred_len=len(y_timestamps),
        T=TEMPERATURE,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=True,
    )

    # pred_df: index = y_timestamps, columns = pred-0 .. pred-(SAMPLE_COUNT-1)
    # Values are power in MW (denormalised from the z-score using history stats)
    samples_mw = pred_df.values  # shape (pred_len, sample_count)

    # ---- 9. Compute percentiles ---------------------------------------------
    samples_kw = samples_mw * 1000.0  # convert MW -> kW

    # ---- 10. Post-process: clip & physical constraints ----------------------
    forecast_wind = forecast["wind_speed"].values[:len(y_timestamps)]

    # Apply constraints to each sample
    for s in range(samples_kw.shape[1]):
        samples_kw[:, s] = apply_physical_constraints(
            samples_kw[:, s], forecast_wind, turbine_model,
        )

    if samples_kw.shape[1] >= 5:
        power_percentiles = _compute_percentiles(samples_kw, PERCENTILES)
    else:
        # Too few samples for meaningful percentiles; estimate bands from
        # the single prediction using forecast wind speed uncertainty.
        p50 = samples_kw[:, 0]
        uncertainty = np.clip(forecast_wind * 0.15 + 0.5, 0.5, 5.0)
        frac = np.clip(uncertainty / (forecast_wind + 1e-6), 0.05, 0.6)
        power_percentiles = {
            "P5": np.clip(p50 * (1 - 1.6 * frac), 0, spec.rated_power_kw),
            "P25": np.clip(p50 * (1 - 0.67 * frac), 0, spec.rated_power_kw),
            "P50": p50,
            "P75": np.clip(p50 * (1 + 0.67 * frac), 0, spec.rated_power_kw),
            "P95": np.clip(p50 * (1 + 1.6 * frac), 0, spec.rated_power_kw),
        }

    # Convert percentile arrays to plain lists for JSON-friendliness
    power_kw_out = {
        k: [round(float(v), 2) for v in arr]
        for k, arr in power_percentiles.items()
    }

    # ---- 11. Daily energy & capacity factor ---------------------------------
    p50_kw = power_percentiles["P50"]
    daily_energy = _compute_daily_energy(p50_kw)
    capacity_factor = _compute_capacity_factor(p50_kw, spec.rated_power_kw)

    # ---- 12. Build result ---------------------------------------------------
    location_info = _build_location_info(lat, lon, city=city_name)
    turbine_info = _build_turbine_info(spec)

    forecast_timestamps = [t.isoformat() for t in y_timestamps]

    weather_display = {
        "wind_speed": [round(float(v), 2) for v in forecast_wind],
        "wind_direction": [
            round(float(v), 1)
            for v in forecast["wind_direction"].values[:len(y_timestamps)]
        ],
        "temperature": [
            round(float(v), 1)
            for v in forecast["temperature"].values[:len(y_timestamps)]
        ],
    }

    result = ForecastResult(
        location=location_info,
        turbine=turbine_info,
        timestamps=forecast_timestamps,
        power_kw=power_kw_out,
        daily_energy_mwh=daily_energy,
        capacity_factor=capacity_factor,
        weather=weather_display,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Prediction complete. Capacity factor: %.1f%%, 7-day energy: %.1f MWh",
        capacity_factor * 100,
        sum(daily_energy),
    )

    return result


# ---------------------------------------------------------------------------
# City-based entry point
# ---------------------------------------------------------------------------
def predict_from_city(
    city_name: str,
    turbine_model: str,
    model_path: Optional[str] = None,
) -> ForecastResult:
    """Geocode a city name and run the prediction pipeline.

    Parameters
    ----------
    city_name : str
        City name to geocode (e.g. ``"Istanbul"``).
    turbine_model : str
        Turbine model name.
    model_path : str, optional
        Path to a fine-tuned WindFM checkpoint.

    Returns
    -------
    ForecastResult
    """
    logger.info("Geocoding city: %s", city_name)
    try:
        results = geocode(city_name, count=1)
    except Exception as exc:
        raise ValueError(f"Geocoding failed for '{city_name}': {exc}") from exc

    top = results[0]
    lat = top["latitude"]
    lon = top["longitude"]
    resolved_name = top.get("name", city_name)

    logger.info(
        "Resolved '%s' -> %s (%.4f, %.4f)",
        city_name, resolved_name, lat, lon,
    )

    return predict(
        lat=lat,
        lon=lon,
        turbine_model=turbine_model,
        model_path=model_path,
        city_name=resolved_name,
    )
