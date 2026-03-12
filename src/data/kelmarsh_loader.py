"""
Kelmarsh Wind Farm Loader — Preprocesses Kelmarsh SCADA to WindFM 6-feature format.

Source: Zenodo (CC-BY 4.0)
Turbines: 6 × Senvion MM92 (2.05 MW, 92.5m rotor, 68.5-78.5m hub)
Location: Northamptonshire, England (~52.40°N, -0.94°W, elevation ~135-157m)
Terrain: Flat farmland
Resolution: 10-min → resampled to 1-hour
"""

import pandas as pd
import numpy as np
import zipfile
import io
from pathlib import Path

RAW_DIR = Path("data/kelmarsh/raw")
PROCESSED_DIR = Path("data/kelmarsh/processed")

WINDFM_FEATURES = ["wind_speed", "wind_direction", "power", "density", "temperature", "pressure"]
R_D = 287.05

# Kelmarsh turbine metadata (from kelmarsh_static.csv)
TURBINES = {
    "Kelmarsh 1": {"lat": 52.4027, "lon": -0.9417, "elevation": 157, "hub_height": 78.5},
    "Kelmarsh 2": {"lat": 52.4010, "lon": -0.9363, "elevation": 148, "hub_height": 78.5},
    "Kelmarsh 3": {"lat": 52.3989, "lon": -0.9307, "elevation": 140, "hub_height": 68.5},
    "Kelmarsh 4": {"lat": 52.3988, "lon": -0.9473, "elevation": 153, "hub_height": 78.5},
    "Kelmarsh 5": {"lat": 52.3970, "lon": -0.9417, "elevation": 145, "hub_height": 78.5},
    "Kelmarsh 6": {"lat": 52.3950, "lon": -0.9363, "elevation": 135, "hub_height": 68.5},
}

RATED_POWER_KW = 2050


def pressure_from_elevation(elevation_m: float) -> float:
    """Estimate atmospheric pressure from elevation using barometric formula.
    P = P_sea × (1 - 2.25577e-5 × h)^5.25588
    """
    return 101325.0 * (1 - 2.25577e-5 * elevation_m) ** 5.25588


def parse_turbine_csv(zf: zipfile.ZipFile, csv_name: str) -> pd.DataFrame:
    """Parse a single turbine CSV from a Kelmarsh zip file.
    Kelmarsh CSVs have comment lines (starting with #) then data.
    The last comment line contains the header (starts with '# Date and time,...').
    The actual data rows follow without a separate header line.
    """
    with zf.open(csv_name) as f:
        content = f.read().decode("utf-8")

    lines = content.split("\n")

    # Find the header line (last comment line with column names)
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#") and ("Date and time" in line or "Wind speed" in line):
            header_line = line.lstrip("# ").strip()
        if not line.startswith("#"):
            data_start = i
            break

    if header_line is None:
        # Fallback: first non-comment line is header
        data_str = "\n".join(lines[data_start:])
        return pd.read_csv(io.StringIO(data_str))

    # Combine header with data lines
    data_str = header_line + "\n" + "\n".join(lines[data_start:])
    return pd.read_csv(io.StringIO(data_str))


def extract_turbine_name(csv_name: str) -> str:
    """Extract turbine name from CSV filename like 'Turbine_Data_Kelmarsh_1_...'"""
    parts = csv_name.replace(".csv", "").split("_")
    # Find 'Kelmarsh' and take it + next number
    for i, p in enumerate(parts):
        if p == "Kelmarsh" and i + 1 < len(parts):
            return f"Kelmarsh {parts[i+1]}"
    return csv_name


def load_year(zip_path: Path) -> pd.DataFrame:
    """Load all turbine data from a single year zip file."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if "Turbine_Data" in n and n.endswith(".csv")]
        for csv_name in csv_files:
            turbine_name = extract_turbine_name(csv_name.split("/")[-1])
            if turbine_name not in TURBINES:
                continue

            df = parse_turbine_csv(zf, csv_name)

            # Find relevant columns (position varies across years)
            col_map = {}
            for col in df.columns:
                cl = col.strip().lower()
                if cl in ("date and time", "# date and time"):
                    col_map["time"] = col
                elif cl == "wind speed (m/s)":
                    if "wind_speed" not in col_map:
                        col_map["wind_speed"] = col
                elif "wind direction" in cl and ("°" in col or "(deg)" in cl):
                    if "wind_direction" not in col_map and "standard deviation" not in cl and "min" not in cl and "max" not in cl:
                        col_map["wind_direction"] = col
                elif cl == "power (kw)":
                    if "power" not in col_map:
                        col_map["power"] = col
                elif cl == "nacelle ambient temperature (°c)" or cl == "nacelle ambient temperature (deg c)":
                    if "temperature" not in col_map:
                        col_map["temperature"] = col

            # Try the first column for timestamp if not found
            if "time" not in col_map:
                col_map["time"] = df.columns[0]

            if not all(k in col_map for k in ["time", "wind_speed", "power"]):
                print(f"  Warning: Missing columns for {turbine_name} in {zip_path.name}, found: {list(col_map.keys())}")
                continue

            meta = TURBINES[turbine_name]
            pressure_pa = pressure_from_elevation(meta["elevation"])

            mapped = pd.DataFrame()
            mapped["time"] = pd.to_datetime(df[col_map["time"]], utc=True, format="mixed")
            mapped["TurbID"] = turbine_name
            mapped["wind_speed"] = pd.to_numeric(df[col_map["wind_speed"]], errors="coerce")
            mapped["wind_direction"] = pd.to_numeric(df.get(col_map.get("wind_direction", ""), pd.Series(dtype=float)), errors="coerce")
            mapped["power"] = pd.to_numeric(df[col_map["power"]], errors="coerce") / 1000.0  # kW → MW

            if "temperature" in col_map:
                temp_c = pd.to_numeric(df[col_map["temperature"]], errors="coerce")
            else:
                temp_c = pd.Series(10.0, index=df.index)  # default 10°C if missing

            mapped["temperature"] = temp_c + 273.15  # °C → K
            mapped["pressure"] = pressure_pa
            mapped["density"] = pressure_pa / (R_D * mapped["temperature"])

            frames.append(mapped)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_years() -> pd.DataFrame:
    """Load all Kelmarsh data across all available year zips."""
    zip_files = sorted(RAW_DIR.glob("kelmarsh*.zip"))
    print(f"Found {len(zip_files)} zip files")

    frames = []
    for zp in zip_files:
        print(f"  Processing {zp.name}...")
        yr_df = load_year(zp)
        if len(yr_df) > 0:
            frames.append(yr_df)
            print(f"    {len(yr_df):,} rows")

    if not frames:
        raise ValueError("No data loaded from Kelmarsh zips")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Kelmarsh data."""
    df = df.copy()
    df = df.dropna(subset=["wind_speed", "power", "time"])
    df["power"] = df["power"].clip(lower=0, upper=RATED_POWER_KW / 1000.0)
    df = df[df["wind_speed"] <= 40]
    df = df[df["wind_speed"] >= 0]
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-min to hourly per turbine."""
    df = df.set_index("time")
    resampled = (
        df.groupby("TurbID")
        .resample("1h")
        .agg({
            "wind_speed": "mean",
            "wind_direction": "mean",
            "power": "mean",
            "density": "mean",
            "temperature": "mean",
            "pressure": "mean",
        })
        .dropna()
        .reset_index()
    )
    return resampled


def process_and_save():
    """Full pipeline."""
    print("Loading Kelmarsh data...")
    df = load_all_years()
    print(f"  Total raw: {len(df):,} rows")

    print("Cleaning...")
    df = clean(df)
    print(f"  After cleaning: {len(df):,} rows")

    print("Resampling to hourly...")
    hourly = resample_hourly(df)
    print(f"  Hourly: {len(hourly):,} rows")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "eval.parquet"
    hourly.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(hourly):,} rows)")

    print("\nFeature statistics:")
    print(hourly[WINDFM_FEATURES].describe().round(4))

    return hourly


if __name__ == "__main__":
    process_and_save()
