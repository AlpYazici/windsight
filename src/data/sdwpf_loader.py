"""
SDWPF Dataset Loader — Preprocesses SDWPF to WindFM 6-feature format.

Source: China Longyuan Power Group / Baidu KDD Cup 2022
Turbines: 134 × Sinovel SL1500/82 (1.5 MW)
Resolution: 10-min → resampled to 1-hour
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/sdwpf/raw/sdwpf_full.parquet")
PROCESSED_DIR = Path("data/sdwpf/processed")

WINDFM_FEATURES = ["wind_speed", "wind_direction", "power", "density", "temperature", "pressure"]

# Gas constant for dry air (J/(kg·K))
R_D = 287.05


def load_raw() -> pd.DataFrame:
    """Load raw SDWPF parquet."""
    return pd.read_parquet(RAW_PATH)


def clean_and_map(df: pd.DataFrame) -> pd.DataFrame:
    """Clean SDWPF and map to WindFM 6-feature format.

    Cleaning steps:
    - Drop rows with NaN in critical SCADA columns (Wspd, Wdir, Patv)
    - Drop rows with NaN in ERA5 columns (T2m, Sp)
    - Clip negative power to 0
    - Cap power at 1500 kW (rated capacity)
    - Remove physically impossible wind speeds (> 40 m/s)
    - Remove Wdir outliers (outside [-180, 360])

    Feature mapping:
    - wind_speed ← Wspd (m/s)
    - wind_direction ← Wdir (degrees)
    - power ← Patv / 1000 (kW → MW)
    - temperature ← T2m + 273.15 (°C → K) [ERA5, more reliable than Etmp]
    - pressure ← Sp (Pa, ERA5 surface pressure)
    - density ← Sp / (R_d × (T2m + 273.15))
    """
    df = df.copy()

    # Drop rows missing critical columns
    critical_scada = ["Wspd", "Wdir", "Patv"]
    critical_era5 = ["T2m", "Sp"]
    df = df.dropna(subset=critical_scada + critical_era5)

    # Clean power
    df["Patv"] = df["Patv"].clip(lower=0, upper=1500)

    # Clean wind speed
    df = df[df["Wspd"] <= 40]

    # Clean wind direction
    df = df[(df["Wdir"] >= -180) & (df["Wdir"] <= 360)]

    # Map to WindFM features
    temp_k = df["T2m"].astype(float) + 273.15

    mapped = pd.DataFrame({
        "TurbID": df["TurbID"],
        "time": pd.to_datetime(df["Tmstamp"], utc=True),
        "wind_speed": df["Wspd"].astype(float),
        "wind_direction": df["Wdir"].astype(float),
        "power": df["Patv"].astype(float) / 1000.0,  # kW → MW
        "density": df["Sp"].astype(float) / (R_D * temp_k),
        "temperature": temp_k,
        "pressure": df["Sp"].astype(float),
    })

    return mapped


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-min data to hourly per turbine."""
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


def temporal_split(df: pd.DataFrame):
    """Split by date: train 70%, val 15%, test 15%.

    Train: Jan 2020 – Aug 2021
    Val:   Sep 2021 – Oct 2021
    Test:  Nov 2021 – Dec 2021
    """
    train = df[df["time"] < "2021-09-01"]
    val = df[(df["time"] >= "2021-09-01") & (df["time"] < "2021-11-01")]
    test = df[df["time"] >= "2021-11-01"]
    return train, val, test


def process_and_save():
    """Full pipeline: load → clean → map → resample → split → save."""
    print("Loading raw SDWPF...")
    raw = load_raw()
    print(f"  Raw shape: {raw.shape}")

    print("Cleaning and mapping to WindFM format...")
    mapped = clean_and_map(raw)
    print(f"  Mapped shape: {mapped.shape}")

    print("Resampling to hourly...")
    hourly = resample_hourly(mapped)
    print(f"  Hourly shape: {hourly.shape}")

    print("Splitting train/val/test...")
    train, val, test = temporal_split(hourly)
    print(f"  Train: {train.shape} ({train['time'].min()} to {train['time'].max()})")
    print(f"  Val:   {val.shape} ({val['time'].min()} to {val['time'].max()})")
    print(f"  Test:  {test.shape} ({test['time'].min()} to {test['time'].max()})")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = PROCESSED_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(split_df):,} rows)")

    # Print summary statistics
    print("\nFeature statistics (train):")
    print(train[WINDFM_FEATURES].describe().round(4))

    return train, val, test


if __name__ == "__main__":
    process_and_save()
