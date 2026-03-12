"""
Penmanshiel Wind Farm Loader — Preprocesses to WindFM 6-feature format.

Source: Zenodo (CC-BY 4.0)
Turbines: 14 × Senvion MM82 (2.05 MW, 82m rotor, 59m hub)
Location: Scottish Borders (~55.87°N, -2.35°W, elevation ~180-228m)
Terrain: Hilly
Resolution: 10-min → resampled to 1-hour
Note: Same Greenbyte format as Kelmarsh
"""

import pandas as pd
import numpy as np
import zipfile
import io
from pathlib import Path

RAW_DIR = Path("data/penmanshiel/raw")
PROCESSED_DIR = Path("data/penmanshiel/processed")

WINDFM_FEATURES = ["wind_speed", "wind_direction", "power", "density", "temperature", "pressure"]
R_D = 287.05
RATED_POWER_KW = 2050

# Average elevation for Penmanshiel (from penmanshiel_static.csv)
AVG_ELEVATION = 200  # ~180-228m range


def pressure_from_elevation(elevation_m: float) -> float:
    """Barometric formula: P = P_sea × (1 - 2.25577e-5 × h)^5.25588"""
    return 101325.0 * (1 - 2.25577e-5 * elevation_m) ** 5.25588


def parse_greenbyte_csv(zf: zipfile.ZipFile, csv_name: str) -> pd.DataFrame:
    """Parse a Greenbyte-format CSV (same format as Kelmarsh).
    The last comment line contains the header (starts with '# Date and time,...').
    """
    with zf.open(csv_name) as f:
        content = f.read().decode("utf-8")

    lines = content.split("\n")

    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#") and ("Date and time" in line or "Wind speed" in line):
            header_line = line.lstrip("# ").strip()
        if not line.startswith("#"):
            data_start = i
            break

    if header_line is None:
        data_str = "\n".join(lines[data_start:])
        return pd.read_csv(io.StringIO(data_str))

    data_str = header_line + "\n" + "\n".join(lines[data_start:])
    return pd.read_csv(io.StringIO(data_str))


def extract_turbine_id(csv_name: str) -> str:
    """Extract turbine ID from filename like 'Turbine_Data_Penmanshiel_01_...'"""
    parts = csv_name.replace(".csv", "").split("_")
    for i, p in enumerate(parts):
        if p in ("Penmanshiel", "penmanshiel") and i + 1 < len(parts):
            return f"T{parts[i+1].zfill(2)}"
    return csv_name


def load_zip(zip_path: Path) -> pd.DataFrame:
    """Load all turbine data from a zip file."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if "Turbine_Data" in n and n.endswith(".csv")]
        print(f"    Found {len(csv_files)} turbine files")

        for csv_name in csv_files:
            turb_id = extract_turbine_id(csv_name.split("/")[-1])
            df = parse_greenbyte_csv(zf, csv_name)

            # Find columns (same Greenbyte structure as Kelmarsh)
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

            if "time" not in col_map:
                col_map["time"] = df.columns[0]

            if not all(k in col_map for k in ["time", "wind_speed", "power"]):
                print(f"    Warning: Missing columns for {turb_id}, found: {list(col_map.keys())}")
                continue

            pressure_pa = pressure_from_elevation(AVG_ELEVATION)

            mapped = pd.DataFrame()
            mapped["time"] = pd.to_datetime(df[col_map["time"]], utc=True, format="mixed")
            mapped["TurbID"] = turb_id
            mapped["wind_speed"] = pd.to_numeric(df[col_map["wind_speed"]], errors="coerce")
            mapped["wind_direction"] = pd.to_numeric(df.get(col_map.get("wind_direction", ""), pd.Series(dtype=float)), errors="coerce")
            mapped["power"] = pd.to_numeric(df[col_map["power"]], errors="coerce") / 1000.0

            if "temperature" in col_map:
                temp_c = pd.to_numeric(df[col_map["temperature"]], errors="coerce")
            else:
                temp_c = pd.Series(8.0, index=df.index)  # Scottish average

            mapped["temperature"] = temp_c + 273.15
            mapped["pressure"] = pressure_pa
            mapped["density"] = pressure_pa / (R_D * mapped["temperature"])

            frames.append(mapped)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data."""
    df = df.copy()
    df = df.dropna(subset=["wind_speed", "power", "time"])
    df["power"] = df["power"].clip(lower=0, upper=RATED_POWER_KW / 1000.0)
    df = df[df["wind_speed"].between(0, 40)]
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to hourly."""
    df = df.set_index("time")
    return (
        df.groupby("TurbID")
        .resample("1h")
        .agg({f: "mean" for f in WINDFM_FEATURES})
        .dropna()
        .reset_index()
    )


def process_and_save():
    """Full pipeline."""
    print("Loading Penmanshiel data...")
    zip_files = sorted(RAW_DIR.glob("penmanshiel*.zip"))
    print(f"  Found {len(zip_files)} zip files")

    frames = []
    for zp in zip_files:
        print(f"  Processing {zp.name}...")
        df = load_zip(zp)
        if len(df) > 0:
            frames.append(df)
            print(f"    {len(df):,} rows")

    if not frames:
        raise ValueError("No data loaded")

    df = pd.concat(frames, ignore_index=True)
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
