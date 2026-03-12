"""
Open-Meteo Weather API Client — Fetches historical & forecast weather data.

Provides historical and forecast weather data from Open-Meteo for any lat/lon,
with hub-height wind extrapolation and air density calculation.
Returns DataFrames in WindFM 6-feature format.

Endpoints (no API key required):
- Historical: https://archive-api.open-meteo.com/v1/archive
- Forecast:   https://api.open-meteo.com/v1/forecast
- Elevation:  https://api.open-meteo.com/v1/elevation
- Geocoding:  https://geocoding-api.open-meteo.com/v1/search
"""

import math
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDFM_FEATURES = [
    "wind_speed", "wind_direction", "power",
    "density", "temperature", "pressure",
]

HOURLY_VARIABLES = [
    "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m",
    "temperature_2m", "surface_pressure",
]

# Gas constant for dry air (J/(kg·K))
R_D = 287.05

# Reference heights for wind shear calculation (metres)
H_REF_LOW = 10.0
H_REF_HIGH = 80.0

# API base URLs
URL_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
URL_FORECAST = "https://api.open-meteo.com/v1/forecast"
URL_ELEVATION = "https://api.open-meteo.com/v1/elevation"
URL_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"

# Cache: LRU with max 64 entries.  Since lru_cache does not support TTL
# natively, we round the current time to 15-min buckets and include that
# bucket in the cache key so stale entries are automatically evicted once
# a new 15-min window begins.
_CACHE_TTL_SECONDS = 900  # 15 minutes
_CACHE_MAXSIZE = 64


def _cache_bucket() -> int:
    """Return an integer that changes every 15 minutes (UTC epoch // 900)."""
    return int(datetime.now(timezone.utc).timestamp()) // _CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------
def _get_json(url: str, params: dict, timeout: int = 30) -> dict:
    """Perform a GET request and return the JSON response.

    Raises ``requests.HTTPError`` on non-2xx status codes and
    ``ValueError`` if the response body cannot be decoded as JSON.
    """
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Open-Meteo returns errors inside the JSON body in some cases
    if "error" in data and data["error"]:
        reason = data.get("reason", "unknown error")
        raise ValueError(f"Open-Meteo API error: {reason}")
    return data


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
@lru_cache(maxsize=_CACHE_MAXSIZE)
def geocode(city: str, count: int = 1) -> list[dict]:
    """Resolve a city name to geographic coordinates.

    Parameters
    ----------
    city : str
        City name (e.g. ``"Istanbul"``).
    count : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each dict has at least ``name``, ``latitude``, ``longitude``,
        ``country``, and ``elevation`` keys.
    """
    data = _get_json(URL_GEOCODING, {"name": city, "count": count})
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No geocoding results for '{city}'")
    return results


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=_CACHE_MAXSIZE)
def get_elevation(latitude: float, longitude: float) -> float:
    """Fetch terrain elevation (metres above sea level) for a coordinate."""
    data = _get_json(
        URL_ELEVATION,
        {"latitude": latitude, "longitude": longitude},
    )
    elevations = data.get("elevation", [])
    if not elevations:
        raise ValueError("No elevation data returned")
    return float(elevations[0])


# ---------------------------------------------------------------------------
# Wind extrapolation & air density
# ---------------------------------------------------------------------------
def wind_shear_exponent(v_10: np.ndarray, v_80: np.ndarray) -> np.ndarray:
    """Calculate the empirical wind shear exponent (alpha).

    alpha = ln(v_80 / v_10) / ln(80 / 10)

    Where both speeds are <= 0 or identical, a default alpha of 1/7 (0.143)
    is used, which is the commonly accepted value over open terrain.
    """
    alpha_default = 1.0 / 7.0
    v_10 = np.asarray(v_10, dtype=np.float64)
    v_80 = np.asarray(v_80, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = v_80 / v_10
        alpha = np.log(ratio) / math.log(H_REF_HIGH / H_REF_LOW)

    # Replace NaN/inf with default (occurs when v_10 == 0 or ratio <= 0)
    bad = ~np.isfinite(alpha) | (v_10 <= 0) | (v_80 <= 0)
    alpha[bad] = alpha_default
    return alpha


def extrapolate_wind(
    v_ref: np.ndarray,
    h_ref: float,
    h_hub: float,
    alpha: np.ndarray,
) -> np.ndarray:
    """Power-law hub-height wind extrapolation.

    v_hub = v_ref * (h_hub / h_ref) ^ alpha
    """
    v_ref = np.asarray(v_ref, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    return v_ref * (h_hub / h_ref) ** alpha


def air_density(pressure_hpa: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Compute air density (kg/m^3) from pressure and temperature.

    density = (pressure_hPa * 100) / (R_d * (T_C + 273.15))
    """
    p = np.asarray(pressure_hpa, dtype=np.float64)
    t = np.asarray(temperature_c, dtype=np.float64)
    return (p * 100.0) / (R_D * (t + 273.15))


# ---------------------------------------------------------------------------
# Historical weather
# ---------------------------------------------------------------------------
@lru_cache(maxsize=_CACHE_MAXSIZE)
def _fetch_historical_cached(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    _bucket: int,
) -> dict:
    """Cached wrapper around the Open-Meteo Historical API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    return _get_json(URL_HISTORICAL, params)


def fetch_historical(
    latitude: float,
    longitude: float,
    days: int = 365,
    hub_height: float = 80.0,
) -> pd.DataFrame:
    """Fetch historical hourly weather and return WindFM-formatted DataFrame.

    Parameters
    ----------
    latitude, longitude : float
        Location coordinates.
    days : int
        Number of past days to fetch (default 365).
    hub_height : float
        Turbine hub height in metres (default 80 m).

    Returns
    -------
    pd.DataFrame
        Columns: ``time`` (UTC datetime), plus the 6 WindFM features.
    """
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days - 1)

    data = _fetch_historical_cached(
        latitude, longitude,
        start.isoformat(), end.isoformat(),
        _cache_bucket(),
    )

    return _hourly_to_windfm(data, hub_height)


# ---------------------------------------------------------------------------
# Forecast weather
# ---------------------------------------------------------------------------
@lru_cache(maxsize=_CACHE_MAXSIZE)
def _fetch_forecast_cached(
    latitude: float,
    longitude: float,
    past_days: int,
    forecast_days: int,
    _bucket: int,
) -> dict:
    """Cached wrapper around the Open-Meteo Forecast API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "wind_speed_unit": "ms",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    return _get_json(URL_FORECAST, params)


def fetch_forecast(
    latitude: float,
    longitude: float,
    hub_height: float = 80.0,
    past_days: int = 0,
    forecast_days: int = 7,
) -> pd.DataFrame:
    """Fetch forecast (and optionally recent history) as WindFM DataFrame.

    Parameters
    ----------
    latitude, longitude : float
        Location coordinates.
    hub_height : float
        Turbine hub height in metres (default 80 m).
    past_days : int
        Number of past days to include (0–92). Uses the forecast API's
        ``past_days`` parameter which, unlike the archive API, has no
        multi-day delay.
    forecast_days : int
        Number of forecast days (default 7, max 16).

    Returns
    -------
    pd.DataFrame
        Columns: ``time`` (UTC datetime), plus the 6 WindFM features.
    """
    data = _fetch_forecast_cached(
        latitude, longitude, past_days, forecast_days, _cache_bucket(),
    )
    return _hourly_to_windfm(data, hub_height)


# ---------------------------------------------------------------------------
# Combined: historical + forecast
# ---------------------------------------------------------------------------
# Maximum past_days supported by the Open-Meteo forecast API
_FORECAST_API_MAX_PAST_DAYS = 92


def fetch_weather(
    latitude: float,
    longitude: float,
    history_days: int = 365,
    hub_height: float = 80.0,
) -> pd.DataFrame:
    """Fetch both historical and forecast data, concatenated.

    The resulting DataFrame covers *history_days* in the past plus the
    7-day forecast, all in WindFM 6-feature format.

    For short histories (≤ 92 days) the forecast API's ``past_days``
    parameter is used, which avoids the multi-day data delay of the
    archive API and returns data right up to the current hour.

    For longer histories the archive API is used for the bulk of the
    data, and the forecast API with ``past_days`` fills in the most
    recent days that the archive hasn't indexed yet.
    """
    forecast_days = 7

    if history_days <= _FORECAST_API_MAX_PAST_DAYS:
        # Single call — forecast API handles both history and forecast
        df = fetch_forecast(
            latitude, longitude,
            hub_height=hub_height,
            past_days=history_days,
            forecast_days=forecast_days,
        )
        return df

    # Long history: archive API for the bulk + forecast API for recent days
    # Archive API is delayed ~5 days, so stop 7 days ago and let the
    # forecast API's past_days cover the gap.
    archive_gap_days = 7
    archive_days = history_days - archive_gap_days

    hist = fetch_historical(
        latitude, longitude,
        days=archive_days,
        hub_height=hub_height,
    )
    recent = fetch_forecast(
        latitude, longitude,
        hub_height=hub_height,
        past_days=archive_gap_days,
        forecast_days=forecast_days,
    )
    combined = pd.concat([hist, recent], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Internal: convert hourly JSON to WindFM DataFrame
# ---------------------------------------------------------------------------
def _hourly_to_windfm(data: dict, hub_height: float) -> pd.DataFrame:
    """Convert Open-Meteo hourly JSON response to WindFM DataFrame.

    Steps:
    1. Parse timestamps as UTC.
    2. Compute wind shear exponent alpha from 10 m and 80 m wind speeds.
    3. Extrapolate wind speed to hub height via power law.
    4. Interpolate wind direction between 10 m and 80 m using hub height.
    5. Convert temperature to Kelvin, pressure to Pascals.
    6. Compute air density.
    7. Set power to NaN (to be filled later by synthetic generation).
    """
    hourly = data["hourly"]

    time = pd.to_datetime(hourly["time"], utc=True)

    v10 = np.array(hourly["wind_speed_10m"], dtype=np.float64)
    v80 = np.array(hourly["wind_speed_80m"], dtype=np.float64)
    dir10 = np.array(hourly["wind_direction_10m"], dtype=np.float64)
    dir80 = np.array(hourly["wind_direction_80m"], dtype=np.float64)
    temp_c = np.array(hourly["temperature_2m"], dtype=np.float64)
    # Open-Meteo returns surface pressure in hPa
    pressure_hpa = np.array(hourly["surface_pressure"], dtype=np.float64)

    # Wind shear exponent and hub-height extrapolation
    alpha = wind_shear_exponent(v10, v80)

    if hub_height == H_REF_HIGH:
        wind_speed = v80.copy()
    elif hub_height == H_REF_LOW:
        wind_speed = v10.copy()
    else:
        wind_speed = extrapolate_wind(v10, H_REF_LOW, hub_height, alpha)

    # Interpolate wind direction: linear weight between 10 m and 80 m
    if hub_height <= H_REF_LOW:
        wind_dir = dir10.copy()
    elif hub_height >= H_REF_HIGH:
        wind_dir = dir80.copy()
    else:
        weight = (hub_height - H_REF_LOW) / (H_REF_HIGH - H_REF_LOW)
        # Circular interpolation of angles
        wind_dir = _circular_interp(dir10, dir80, weight)

    # Unit conversions
    temperature_k = temp_c + 273.15
    pressure_pa = pressure_hpa * 100.0
    rho = air_density(pressure_hpa, temp_c)

    df = pd.DataFrame({
        "time": time,
        "wind_speed": wind_speed,
        "wind_direction": wind_dir,
        "power": np.nan,
        "density": rho,
        "temperature": temperature_k,
        "pressure": pressure_pa,
    })

    return df


def _circular_interp(
    angle_a: np.ndarray,
    angle_b: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Linearly interpolate between two angle arrays (in degrees).

    Uses the shortest-arc method so that e.g. 350 → 10 interpolates through
    0 rather than going the long way around.
    """
    a = np.deg2rad(angle_a)
    b = np.deg2rad(angle_b)
    sin_interp = (1 - weight) * np.sin(a) + weight * np.sin(b)
    cos_interp = (1 - weight) * np.cos(a) + weight * np.cos(b)
    return np.rad2deg(np.arctan2(sin_interp, cos_interp)) % 360
