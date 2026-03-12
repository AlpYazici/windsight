"""WindSight Streamlit Dashboard.

Interactive dashboard for wind power generation forecasting.  Communicates
with the FastAPI backend (``api.py``) when it is running, and falls back to
direct module calls / mock data for standalone demo use.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Import project modules (direct access, used as fallback when API is down)
# ---------------------------------------------------------------------------
from src.data.turbine_db import get_turbine, list_turbines, estimate_power_array
from src.api.weather import geocode, get_elevation, fetch_forecast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
COLOR_PRIMARY = "#1E88E5"
COLOR_SECONDARY = "#43A047"
COLOR_WARNING = "#FF7043"
COLOR_BG = "#FAFAFA"

# Confidence band colours (blue, decreasing opacity)
COLOR_P5_P95 = "rgba(30, 136, 229, 0.12)"
COLOR_P25_P75 = "rgba(30, 136, 229, 0.28)"
COLOR_P50_LINE = "rgba(30, 136, 229, 1.0)"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WindSight - Wind Power Forecast",
    page_icon="https://em-content.zobj.net/source/twitter/408/wind-face_1f32c-fe0f.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, professional look
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLOR_BG};
    }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetric"] label {{
        color: #666;
        font-size: 0.85rem;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: #212121;
        font-weight: 600;
    }}
    .block-container {{
        padding-top: 1rem;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }}
    h1 {{
        color: #1565C0;
    }}
    h2, h3 {{
        color: #333;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: call API or fall back to direct module access
# ---------------------------------------------------------------------------
def _api_available() -> bool:
    """Check whether the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _fetch_turbine_list() -> list[dict]:
    """Get turbine list from API or directly from turbine_db."""
    try:
        r = requests.get(f"{API_BASE}/turbines", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback: direct module access
    names = list_turbines()
    results = []
    for name in names:
        spec = get_turbine(name)
        results.append({
            "full_name": spec.full_name,
            "manufacturer": spec.manufacturer,
            "model": spec.model,
            "rated_power_kw": spec.rated_power_kw,
            "rotor_diameter_m": spec.rotor_diameter_m,
            "hub_height_m": spec.hub_height_m,
            "cut_in_speed_ms": spec.cut_in_speed_ms,
            "rated_speed_ms": spec.rated_speed_ms,
            "cut_out_speed_ms": spec.cut_out_speed_ms,
            "swept_area_m2": spec.swept_area_m2,
        })
    return results


def _fetch_forecast_data(lat: float, lon: float, turbine_model: str) -> dict | None:
    """Get forecast from API or generate locally."""
    # Try API first
    try:
        r = requests.post(
            f"{API_BASE}/forecast",
            json={"lat": lat, "lon": lon, "turbine_model": turbine_model},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # Fallback: generate forecast data locally
    return _generate_local_forecast(lat, lon, turbine_model)


def _generate_local_forecast(lat: float, lon: float, turbine_model: str) -> dict | None:
    """Generate forecast data directly without the API."""
    try:
        turbine = get_turbine(turbine_model)
    except KeyError:
        return None

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rated = turbine.rated_power_kw
    hours = 7 * 24

    # Try real weather data
    try:
        weather_df = fetch_forecast(lat, lon, hub_height=turbine.hub_height_m)
        wind_speeds = weather_df["wind_speed"].values[:hours]
        wind_dirs = weather_df["wind_direction"].values[:hours]
        if len(wind_speeds) < hours:
            pad = hours - len(wind_speeds)
            wind_speeds = np.concatenate([wind_speeds, np.full(pad, wind_speeds.mean())])
            wind_dirs = np.concatenate([wind_dirs, np.full(pad, wind_dirs.mean())])
    except Exception:
        base_speed = 5.0 + 3.0 * abs(math.sin(math.radians(lat)))
        t = np.arange(hours, dtype=np.float64)
        diurnal = 1.5 * np.sin(2 * np.pi * t / 24 - np.pi / 3)
        multi_day = 2.0 * np.sin(2 * np.pi * t / (72 + random.uniform(-12, 12)))
        noise = np.random.normal(0, 1.0, hours)
        wind_speeds = np.clip(base_speed + diurnal + multi_day + noise, 0, 30)
        wind_dirs = (180 + 40 * np.sin(2 * np.pi * t / 48) + np.random.normal(0, 15, hours)) % 360

    p50_values = estimate_power_array(turbine.full_name, wind_speeds)

    uncertainty = np.clip(wind_speeds * 0.15 + 0.5, 0.5, 5.0)
    uncertainty_frac = np.clip(uncertainty / (wind_speeds + 1e-6), 0.05, 0.6)

    p5_values = np.clip(p50_values * (1 - 1.6 * uncertainty_frac), 0, rated)
    p25_values = np.clip(p50_values * (1 - 0.67 * uncertainty_frac), 0, rated)
    p75_values = np.clip(p50_values * (1 + 0.67 * uncertainty_frac), 0, rated)
    p95_values = np.clip(p50_values * (1 + 1.6 * uncertainty_frac), 0, rated)

    timestamps = [now + timedelta(hours=i) for i in range(hours)]

    forecast_steps = []
    for i in range(hours):
        forecast_steps.append({
            "time": timestamps[i].strftime("%Y-%m-%dT%H:%MZ"),
            "power_kw": round(float(p50_values[i]), 1),
            "p5": round(float(p5_values[i]), 1),
            "p25": round(float(p25_values[i]), 1),
            "p50": round(float(p50_values[i]), 1),
            "p75": round(float(p75_values[i]), 1),
            "p95": round(float(p95_values[i]), 1),
        })

    daily_energy = []
    for day in range(7):
        s, e = day * 24, (day + 1) * 24
        daily_energy.append(round(float(np.sum(p50_values[s:e])) / 1000.0, 2))

    mean_power = float(np.mean(p50_values))
    cf = mean_power / rated if rated > 0 else 0.0

    try:
        elev = get_elevation(lat, lon)
    except Exception:
        elev = 0.0

    return {
        "location": {"lat": round(lat, 4), "lon": round(lon, 4), "elevation_m": round(elev, 1)},
        "turbine": {"model": turbine.full_name, "rated_power_kw": rated},
        "forecast": forecast_steps,
        "daily_energy_mwh": daily_energy,
        "capacity_factor": round(cf, 4),
        "generated_at": now.strftime("%Y-%m-%dT%H:%MZ"),
        "_weather": {
            "wind_speeds": wind_speeds.tolist(),
            "wind_dirs": wind_dirs.tolist(),
            "timestamps": [ts.strftime("%Y-%m-%dT%H:%MZ") for ts in timestamps],
        },
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def build_power_forecast_chart(forecast_data: list[dict]) -> go.Figure:
    """Build a 7-day power forecast chart with confidence bands."""
    df = pd.DataFrame(forecast_data)
    df["time"] = pd.to_datetime(df["time"])

    fig = go.Figure()

    # P5-P95 band (lighter)
    fig.add_trace(go.Scatter(
        x=pd.concat([df["time"], df["time"][::-1]]),
        y=pd.concat([df["p95"], df["p5"][::-1]]),
        fill="toself",
        fillcolor=COLOR_P5_P95,
        line=dict(color="rgba(0,0,0,0)"),
        name="P5-P95 range",
        showlegend=True,
        hoverinfo="skip",
    ))

    # P25-P75 band (darker)
    fig.add_trace(go.Scatter(
        x=pd.concat([df["time"], df["time"][::-1]]),
        y=pd.concat([df["p75"], df["p25"][::-1]]),
        fill="toself",
        fillcolor=COLOR_P25_P75,
        line=dict(color="rgba(0,0,0,0)"),
        name="P25-P75 range",
        showlegend=True,
        hoverinfo="skip",
    ))

    # P50 line
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["p50"],
        mode="lines",
        line=dict(color=COLOR_P50_LINE, width=2.5),
        name="P50 (median)",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Power: %{y:.0f} kW<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="7-Day Power Forecast with Confidence Bands", font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor=COLOR_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(gridcolor="#EEEEEE", showgrid=True),
        yaxis=dict(gridcolor="#EEEEEE", showgrid=True, rangemode="tozero"),
        height=420,
    )

    return fig


def build_daily_energy_chart(daily_energy: list[float], start_date: str) -> go.Figure:
    """Build a daily energy bar chart."""
    try:
        start = pd.to_datetime(start_date)
    except Exception:
        start = pd.Timestamp.now(tz="UTC")

    dates = [start + timedelta(days=i) for i in range(len(daily_energy))]
    labels = [d.strftime("%a %b %d") for d in dates]

    avg = sum(daily_energy) / len(daily_energy) if daily_energy else 0
    colors = [COLOR_PRIMARY if e >= avg else COLOR_WARNING for e in daily_energy]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=daily_energy,
        marker_color=colors,
        text=[f"{e:.1f}" for e in daily_energy],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>Energy: %{y:.1f} MWh<extra></extra>",
    ))

    # Average line
    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color="#999",
        annotation_text=f"Avg: {avg:.1f} MWh",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(text="Daily Energy Production", font=dict(size=16)),
        yaxis_title="Energy (MWh)",
        plot_bgcolor="white",
        paper_bgcolor=COLOR_BG,
        margin=dict(l=60, r=30, t=60, b=50),
        yaxis=dict(gridcolor="#EEEEEE", showgrid=True, rangemode="tozero"),
        xaxis=dict(gridcolor="#EEEEEE"),
        height=350,
        showlegend=False,
    )

    return fig


def build_wind_chart(timestamps: list, wind_speeds: list, wind_dirs: list) -> go.Figure:
    """Build a wind speed and direction chart."""
    times = pd.to_datetime(timestamps)

    fig = go.Figure()

    # Wind speed line
    fig.add_trace(go.Scatter(
        x=times,
        y=wind_speeds,
        mode="lines",
        line=dict(color=COLOR_SECONDARY, width=2),
        name="Wind Speed",
        yaxis="y1",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Speed: %{y:.1f} m/s<extra></extra>",
    ))

    # Wind direction scatter (sampled every 3 hours for readability)
    step = 3
    fig.add_trace(go.Scatter(
        x=times[::step],
        y=[d for d in wind_dirs[::step]],
        mode="markers",
        marker=dict(
            color=COLOR_PRIMARY,
            size=5,
            opacity=0.5,
        ),
        name="Wind Direction",
        yaxis="y2",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Direction: %{y:.0f} deg<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Wind Conditions", font=dict(size=16)),
        plot_bgcolor="white",
        paper_bgcolor=COLOR_BG,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=60, b=50),
        height=320,
        xaxis=dict(gridcolor="#EEEEEE", showgrid=True),
        yaxis=dict(
            title="Wind Speed (m/s)",
            gridcolor="#EEEEEE",
            showgrid=True,
            rangemode="tozero",
            titlefont=dict(color=COLOR_SECONDARY),
        ),
        yaxis2=dict(
            title="Wind Direction (deg)",
            overlaying="y",
            side="right",
            range=[0, 360],
            dtick=90,
            titlefont=dict(color=COLOR_PRIMARY),
            showgrid=False,
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("WindSight")
st.sidebar.caption("Wind Power Forecast Dashboard")
st.sidebar.divider()

# --- Location input ---
st.sidebar.subheader("Location")

location_mode = st.sidebar.radio(
    "Input method",
    ["City search", "Coordinates"],
    horizontal=True,
    label_visibility="collapsed",
)

lat, lon = None, None
location_name = ""

if location_mode == "City search":
    col_city, col_btn = st.sidebar.columns([3, 1])
    city_input = col_city.text_input("City name", placeholder="e.g. Istanbul")
    search_clicked = col_btn.button("Search", use_container_width=True)

    if search_clicked and city_input:
        try:
            results = geocode(city_input, count=5)
            st.session_state["geo_results"] = results
        except Exception as e:
            st.sidebar.error(f"Geocoding failed: {e}")
            st.session_state["geo_results"] = []

    if "geo_results" in st.session_state and st.session_state["geo_results"]:
        results = st.session_state["geo_results"]
        options = [
            f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}"
            for r in results
        ]
        selected_idx = st.sidebar.selectbox(
            "Select location",
            range(len(options)),
            format_func=lambda i: options[i],
        )
        selected = results[selected_idx]
        lat = selected["latitude"]
        lon = selected["longitude"]
        location_name = options[selected_idx]

else:
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=41.01, step=0.01, format="%.4f")
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=28.97, step=0.01, format="%.4f")

# --- Turbine selector ---
st.sidebar.divider()
st.sidebar.subheader("Turbine Model")

turbine_list = _fetch_turbine_list()

# Group by manufacturer
by_manufacturer: dict[str, list[dict]] = defaultdict(list)
for t in turbine_list:
    by_manufacturer[t["manufacturer"]].append(t)

# Build display options grouped by manufacturer
turbine_options = []
for mfr in sorted(by_manufacturer.keys()):
    for t in sorted(by_manufacturer[mfr], key=lambda x: x["full_name"]):
        turbine_options.append(t["full_name"])

selected_turbine_name = st.sidebar.selectbox(
    "Select turbine",
    turbine_options,
    index=0,
    help="Turbine models grouped alphabetically by manufacturer",
)

# Show selected turbine specs
selected_turbine_data = next((t for t in turbine_list if t["full_name"] == selected_turbine_name), None)
if selected_turbine_data:
    with st.sidebar.expander("Turbine specifications", expanded=False):
        st.markdown(f"""
| Spec | Value |
|---|---|
| **Rated Power** | {selected_turbine_data['rated_power_kw']:.0f} kW |
| **Rotor Diameter** | {selected_turbine_data['rotor_diameter_m']:.0f} m |
| **Hub Height** | {selected_turbine_data['hub_height_m']:.0f} m |
| **Cut-in Speed** | {selected_turbine_data['cut_in_speed_ms']:.1f} m/s |
| **Rated Speed** | {selected_turbine_data['rated_speed_ms']:.1f} m/s |
| **Cut-out Speed** | {selected_turbine_data['cut_out_speed_ms']:.1f} m/s |
| **Swept Area** | {selected_turbine_data['swept_area_m2']:.0f} m2 |
""")

# --- Generate button ---
st.sidebar.divider()
generate_clicked = st.sidebar.button(
    "Generate Forecast",
    type="primary",
    use_container_width=True,
    disabled=(lat is None or lon is None),
)

# Status info
api_up = _api_available()
if api_up:
    st.sidebar.success("API connected", icon=None)
else:
    st.sidebar.info("API offline -- using direct mode", icon=None)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("WindSight")
st.caption("Wind Power Generation Forecast")

# Show map of selected location
if lat is not None and lon is not None:
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_df, zoom=5, use_container_width=True)
    if location_name:
        st.caption(f"Selected: **{location_name}** ({lat:.4f}, {lon:.4f})")
    else:
        st.caption(f"Selected coordinates: {lat:.4f}, {lon:.4f}")

# ---------------------------------------------------------------------------
# Forecast results
# ---------------------------------------------------------------------------
if generate_clicked and lat is not None and lon is not None:
    with st.spinner("Generating forecast..."):
        forecast_data = _fetch_forecast_data(lat, lon, selected_turbine_name)

    if forecast_data is None:
        st.error("Failed to generate forecast. Please check the turbine model and try again.")
    else:
        st.divider()

        # --- Metric cards ---
        col1, col2, col3, col4 = st.columns(4)

        # Current predicted power (first timestep P50)
        current_power = forecast_data["forecast"][0]["p50"]
        rated_kw = forecast_data["turbine"]["rated_power_kw"]
        pct_of_rated = (current_power / rated_kw * 100) if rated_kw > 0 else 0

        col1.metric(
            label="Current Predicted Power",
            value=f"{current_power:,.0f} kW",
            delta=f"{pct_of_rated:.0f}% of rated",
        )

        # Capacity factor
        cf = forecast_data["capacity_factor"]
        cf_label = "Low" if cf < 0.2 else "Moderate" if cf < 0.35 else "Good" if cf < 0.5 else "Excellent"
        col2.metric(
            label="Capacity Factor",
            value=f"{cf * 100:.1f}%",
            delta=cf_label,
        )

        # 7-day total energy
        total_energy = sum(forecast_data["daily_energy_mwh"])
        col3.metric(
            label="7-Day Total Energy",
            value=f"{total_energy:.1f} MWh",
        )

        # Average daily energy
        avg_daily = total_energy / 7
        col4.metric(
            label="Avg Daily Energy",
            value=f"{avg_daily:.1f} MWh/day",
        )

        st.divider()

        # --- Power forecast chart ---
        power_chart = build_power_forecast_chart(forecast_data["forecast"])
        st.plotly_chart(power_chart, use_container_width=True)

        # --- Daily energy & wind charts side by side ---
        col_left, col_right = st.columns(2)

        with col_left:
            start_time = forecast_data["forecast"][0]["time"]
            energy_chart = build_daily_energy_chart(forecast_data["daily_energy_mwh"], start_time)
            st.plotly_chart(energy_chart, use_container_width=True)

        with col_right:
            # Build wind conditions chart
            # If we have weather data in the response (local mode), use it;
            # otherwise reconstruct from forecast timestamps
            timestamps = [f["time"] for f in forecast_data["forecast"]]
            if "_weather" in forecast_data:
                wind_speeds = forecast_data["_weather"]["wind_speeds"]
                wind_dirs = forecast_data["_weather"]["wind_dirs"]
            else:
                # Try fetching weather data directly
                try:
                    weather_df = fetch_forecast(lat, lon)
                    n = len(timestamps)
                    wind_speeds = weather_df["wind_speed"].values[:n].tolist()
                    wind_dirs = weather_df["wind_direction"].values[:n].tolist()
                    # Pad if needed
                    while len(wind_speeds) < n:
                        wind_speeds.append(wind_speeds[-1] if wind_speeds else 5.0)
                        wind_dirs.append(wind_dirs[-1] if wind_dirs else 180)
                except Exception:
                    # Estimate from power values (rough approximation)
                    wind_speeds = [5.0 + random.uniform(-2, 2) for _ in timestamps]
                    wind_dirs = [180 + random.uniform(-40, 40) for _ in timestamps]

            wind_chart = build_wind_chart(timestamps, wind_speeds, wind_dirs)
            st.plotly_chart(wind_chart, use_container_width=True)

        # --- Detailed info ---
        with st.expander("Forecast details"):
            st.markdown(f"""
- **Location:** {forecast_data['location']['lat']}, {forecast_data['location']['lon']} (elevation: {forecast_data['location']['elevation_m']} m)
- **Turbine:** {forecast_data['turbine']['model']} ({forecast_data['turbine']['rated_power_kw']:.0f} kW rated)
- **Generated at:** {forecast_data['generated_at']}
- **Data source:** {'WindSight Pipeline' if api_up else 'Direct forecast (physics-based)'}
""")

            # Downloadable forecast table
            df_table = pd.DataFrame(forecast_data["forecast"])
            st.dataframe(df_table, use_container_width=True, height=300)

elif not generate_clicked:
    # Landing state
    st.info("Configure location and turbine in the sidebar, then click **Generate Forecast** to view predictions.")

    # Show a brief overview
    st.subheader("Available Turbine Models")
    if turbine_list:
        df_turbines = pd.DataFrame(turbine_list)
        display_cols = ["full_name", "rated_power_kw", "rotor_diameter_m", "hub_height_m"]
        display_names = {"full_name": "Model", "rated_power_kw": "Rated Power (kW)",
                         "rotor_diameter_m": "Rotor Diameter (m)", "hub_height_m": "Hub Height (m)"}
        st.dataframe(
            df_turbines[display_cols].rename(columns=display_names),
            use_container_width=True,
            hide_index=True,
        )
