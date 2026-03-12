"""WindSight FastAPI service.

Provides wind power forecast endpoints backed by the WindSight prediction
pipeline.  Falls back to physics-based mock data when the pipeline module
is not yet available, so the API and dashboard can be developed independently.

Run:
    uvicorn api:app --reload
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.turbine_db import get_turbine, list_turbines, TurbineSpec
from src.api.weather import geocode, get_elevation, fetch_forecast

# ---------------------------------------------------------------------------
# Try importing the real predictor; fall back to mock if unavailable
# ---------------------------------------------------------------------------
_PIPELINE_AVAILABLE = False
try:
    from src.pipeline.predictor import predict, ForecastResult  # noqa: F401
    _PIPELINE_AVAILABLE = True
except Exception:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WindSight API",
    description="Wind power generation forecast service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    turbine_model: str = Field(..., description="Turbine model name")


class ForecastTimeStep(BaseModel):
    time: str
    power_kw: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class LocationInfo(BaseModel):
    lat: float
    lon: float
    elevation_m: float


class TurbineInfo(BaseModel):
    model: str
    rated_power_kw: float


class ForecastResponse(BaseModel):
    location: LocationInfo
    turbine: TurbineInfo
    forecast: list[ForecastTimeStep]
    daily_energy_mwh: list[float]
    capacity_factor: float
    generated_at: str


class TurbineDetail(BaseModel):
    full_name: str
    manufacturer: str
    model: str
    rated_power_kw: float
    rotor_diameter_m: float
    hub_height_m: float
    cut_in_speed_ms: float
    rated_speed_ms: float
    cut_out_speed_ms: float
    swept_area_m2: float


# ---------------------------------------------------------------------------
# Mock forecast generator
# ---------------------------------------------------------------------------
def _generate_mock_forecast(
    lat: float,
    lon: float,
    turbine: TurbineSpec,
) -> dict:
    """Generate plausible mock forecast data using the turbine power curve.

    Uses real weather forecast data from Open-Meteo when available, otherwise
    falls back to synthetic wind speed patterns.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rated = turbine.rated_power_kw
    hours = 7 * 24  # 7 days

    # Try to get real forecast weather data
    try:
        weather_df = fetch_forecast(lat, lon, hub_height=turbine.hub_height_m)
        wind_speeds = weather_df["wind_speed"].values[:hours]
        # Pad if needed
        if len(wind_speeds) < hours:
            pad = hours - len(wind_speeds)
            wind_speeds = np.concatenate([wind_speeds, np.full(pad, wind_speeds.mean())])
    except Exception:
        # Synthetic wind speeds: diurnal pattern + trend + noise
        base_speed = 5.0 + 3.0 * abs(math.sin(math.radians(lat)))
        t = np.arange(hours, dtype=np.float64)
        diurnal = 1.5 * np.sin(2 * np.pi * t / 24 - np.pi / 3)
        multi_day = 2.0 * np.sin(2 * np.pi * t / (72 + random.uniform(-12, 12)))
        noise = np.random.normal(0, 1.0, hours)
        wind_speeds = np.clip(base_speed + diurnal + multi_day + noise, 0, 30)

    # Estimate power from the turbine's power curve
    from src.data.turbine_db import estimate_power_array

    p50_values = estimate_power_array(turbine.full_name, wind_speeds)

    # Build confidence bands by scaling around p50
    # Uncertainty grows with wind speed variance
    uncertainty = np.clip(wind_speeds * 0.15 + 0.5, 0.5, 5.0)
    uncertainty_frac = uncertainty / (wind_speeds + 1e-6)
    uncertainty_frac = np.clip(uncertainty_frac, 0.05, 0.6)

    p5_values = np.clip(p50_values * (1 - 1.6 * uncertainty_frac), 0, rated)
    p25_values = np.clip(p50_values * (1 - 0.67 * uncertainty_frac), 0, rated)
    p75_values = np.clip(p50_values * (1 + 0.67 * uncertainty_frac), 0, rated)
    p95_values = np.clip(p50_values * (1 + 1.6 * uncertainty_frac), 0, rated)

    timestamps = [now + timedelta(hours=i) for i in range(hours)]

    forecast_steps = []
    for i in range(hours):
        forecast_steps.append(ForecastTimeStep(
            time=timestamps[i].strftime("%Y-%m-%dT%H:%MZ"),
            power_kw=round(float(p50_values[i]), 1),
            p5=round(float(p5_values[i]), 1),
            p25=round(float(p25_values[i]), 1),
            p50=round(float(p50_values[i]), 1),
            p75=round(float(p75_values[i]), 1),
            p95=round(float(p95_values[i]), 1),
        ))

    # Daily energy (MWh): sum of hourly kW -> kWh -> MWh
    daily_energy = []
    for day in range(7):
        start_h = day * 24
        end_h = start_h + 24
        day_kwh = float(np.sum(p50_values[start_h:end_h]))
        daily_energy.append(round(day_kwh / 1000.0, 2))

    # Capacity factor
    mean_power = float(np.mean(p50_values))
    cf = mean_power / rated if rated > 0 else 0.0

    # Elevation
    try:
        elev = get_elevation(lat, lon)
    except Exception:
        elev = 0.0

    return ForecastResponse(
        location=LocationInfo(lat=round(lat, 4), lon=round(lon, 4), elevation_m=round(elev, 1)),
        turbine=TurbineInfo(model=turbine.full_name, rated_power_kw=rated),
        forecast=forecast_steps,
        daily_energy_mwh=daily_energy,
        capacity_factor=round(cf, 4),
        generated_at=now.strftime("%Y-%m-%dT%H:%MZ"),
    ).model_dump()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "pipeline_available": _PIPELINE_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/turbines", response_model=list[TurbineDetail])
async def turbines():
    """List all available turbine models with specifications."""
    names = list_turbines()
    results = []
    for name in names:
        spec = get_turbine(name)
        results.append(TurbineDetail(
            full_name=spec.full_name,
            manufacturer=spec.manufacturer,
            model=spec.model,
            rated_power_kw=spec.rated_power_kw,
            rotor_diameter_m=spec.rotor_diameter_m,
            hub_height_m=spec.hub_height_m,
            cut_in_speed_ms=spec.cut_in_speed_ms,
            rated_speed_ms=spec.rated_speed_ms,
            cut_out_speed_ms=spec.cut_out_speed_ms,
            swept_area_m2=spec.swept_area_m2,
        ))
    return results


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    """Generate a 7-day wind power forecast for a location and turbine model."""
    # Validate turbine model
    try:
        turbine = get_turbine(req.turbine_model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Use real pipeline if available, otherwise mock
    if _PIPELINE_AVAILABLE:
        try:
            result = predict(req.lat, req.lon, req.turbine_model)
            # Convert ForecastResult to response format
            timestamps = result.timestamps
            power_dict = result.power_kw  # dict with P5, P25, P50, P75, P95

            forecast_steps = []
            for i, ts in enumerate(timestamps):
                ts_str = ts.strftime("%Y-%m-%dT%H:%MZ") if hasattr(ts, "strftime") else str(ts)
                forecast_steps.append(ForecastTimeStep(
                    time=ts_str,
                    power_kw=round(float(power_dict["P50"][i]), 1),
                    p5=round(float(power_dict["P5"][i]), 1),
                    p25=round(float(power_dict["P25"][i]), 1),
                    p50=round(float(power_dict["P50"][i]), 1),
                    p75=round(float(power_dict["P75"][i]), 1),
                    p95=round(float(power_dict["P95"][i]), 1),
                ))

            try:
                elev = get_elevation(req.lat, req.lon)
            except Exception:
                elev = 0.0

            return ForecastResponse(
                location=LocationInfo(
                    lat=round(req.lat, 4),
                    lon=round(req.lon, 4),
                    elevation_m=round(elev, 1),
                ),
                turbine=TurbineInfo(
                    model=turbine.full_name,
                    rated_power_kw=turbine.rated_power_kw,
                ),
                forecast=forecast_steps,
                daily_energy_mwh=[round(e, 2) for e in result.daily_energy_mwh],
                capacity_factor=round(result.capacity_factor, 4),
                generated_at=result.generated_at if isinstance(result.generated_at, str)
                    else result.generated_at.strftime("%Y-%m-%dT%H:%MZ"),
            )
        except Exception as exc:
            logger.warning("Pipeline prediction failed, falling back to mock: %s", exc)

    # Mock / fallback
    try:
        data = _generate_mock_forecast(req.lat, req.lon, turbine)
        return ForecastResponse(**data)
    except Exception as exc:
        logger.exception("Mock forecast generation failed")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {exc}")
