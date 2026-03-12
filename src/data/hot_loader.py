"""
Hill of Towie Wind Farm Loader — Preprocesses to WindFM 6-feature format.

Source: Zenodo (CC-BY 4.0)
Turbines: 21 × Siemens SWT-2.3-VS-82 (2.3 MW, 82m rotor, 59m hub)
Location: Aberdeenshire, Scotland
Terrain: Hilly
Resolution: 10-min → resampled to 1-hour

Note: Different format from Kelmarsh/Penmanshiel (Siemens SCADA tables).
Wind direction column is missing from actual data — using nacelle yaw position as proxy.
No atmospheric pressure — derived from elevation.
"""

import pandas as pd
import numpy as np
import zipfile
import io
from pathlib import Path

RAW_DIR = Path("data/hill_of_towie/raw")
PROCESSED_DIR = Path("data/hill_of_towie/processed")

WINDFM_FEATURES = ["wind_speed", "wind_direction", "power", "density", "temperature", "pressure"]
R_D = 287.05
RATED_POWER_KW = 2300
AVG_ELEVATION = 350  # Hill of Towie is elevated terrain in Aberdeenshire


def pressure_from_elevation(elevation_m: float) -> float:
    """Barometric formula."""
    return 101325.0 * (1 - 2.25577e-5 * elevation_m) ** 5.25588


def load_metadata() -> dict:
    """Load turbine metadata: Station ID → turbine name mapping."""
    meta_path = RAW_DIR / "hot_metadata.csv"
    meta = pd.read_csv(meta_path)
    mapping = {}
    for _, row in meta.iterrows():
        mapping[row["Station ID"]] = row["Turbine Name"]
    return mapping


def load_table_from_zip(zf: zipfile.ZipFile, table_prefix: str, month_file: str) -> pd.DataFrame:
    """Load a specific table type CSV from the zip."""
    matching = [n for n in zf.namelist() if table_prefix in n and month_file in n]
    if not matching:
        return pd.DataFrame()

    with zf.open(matching[0]) as f:
        return pd.read_csv(f)


def load_zip(zip_path: Path) -> pd.DataFrame:
    """Load and merge relevant tables from the HoT zip file.

    We need:
    - tblSCTurbine: wind speed, nacelle position (proxy for wind direction)
    - tblSCTurGrid: active power
    - tblSCTurTemp: ambient temperature
    """
    station_map = load_metadata()

    with zipfile.ZipFile(zip_path) as zf:
        all_files = zf.namelist()

        # Group files by month suffix (format: tblXxx_2020_01.csv)
        import re
        months = set()
        for f in all_files:
            m = re.search(r'_(\d{4}_\d{2})\.csv$', f)
            if m:
                months.add(m.group(1))

        frames = []
        for month in sorted(months):
            month_suffix = f"_{month}.csv"
            turbine_files = [f for f in all_files if "tblSCTurbine" in f and f.endswith(month_suffix)]
            grid_files = [f for f in all_files if "tblSCTurGrid" in f and f.endswith(month_suffix)]
            temp_files = [f for f in all_files if "tblSCTurTemp" in f and f.endswith(month_suffix)]

            if not turbine_files or not grid_files:
                continue

            # Load tables
            with zf.open(turbine_files[0]) as f:
                df_turb = pd.read_csv(f)
            with zf.open(grid_files[0]) as f:
                df_grid = pd.read_csv(f)

            df_temp = None
            if temp_files:
                with zf.open(temp_files[0]) as f:
                    df_temp = pd.read_csv(f)

            # Merge on timestamp + StationId
            time_col = "PCTimeStamp" if "PCTimeStamp" in df_turb.columns else df_turb.columns[0]
            station_col = "StationId" if "StationId" in df_turb.columns else "stationid"

            # Extract wind speed and nacelle position from tblSCTurbine
            turb_cols = [time_col, station_col]
            ws_col = None
            for c in df_turb.columns:
                if "AcWindSp_mean" in c or "PrWindSp_mean" in c:
                    ws_col = c
                    break
            if ws_col is None:
                for c in df_turb.columns:
                    if "WindSp" in c and "mean" in c:
                        ws_col = c
                        break

            nacel_col = None
            for c in df_turb.columns:
                if "NacelPos_mean" in c or "YawPos_mean" in c or "ScYawPos_mean" in c:
                    nacel_col = c
                    break

            if ws_col:
                turb_cols.append(ws_col)
            if nacel_col:
                turb_cols.append(nacel_col)

            df_turb_slim = df_turb[turb_cols].copy()
            df_turb_slim.rename(columns={
                time_col: "time",
                station_col: "StationId",
            }, inplace=True)
            if ws_col:
                df_turb_slim.rename(columns={ws_col: "wind_speed_raw"}, inplace=True)
            if nacel_col:
                df_turb_slim.rename(columns={nacel_col: "nacelle_pos"}, inplace=True)

            # Extract power from tblSCTurGrid
            pwr_col = None
            for c in df_grid.columns:
                if "ActPower_mean" in c:
                    pwr_col = c
                    break
            if pwr_col is None:
                for c in df_grid.columns:
                    if "Power" in c and "mean" in c:
                        pwr_col = c
                        break

            grid_time_col = "PCTimeStamp" if "PCTimeStamp" in df_grid.columns else df_grid.columns[0]
            grid_station_col = "StationId" if "StationId" in df_grid.columns else "stationid"

            if pwr_col:
                df_grid_slim = df_grid[[grid_time_col, grid_station_col, pwr_col]].copy()
                df_grid_slim.rename(columns={
                    grid_time_col: "time",
                    grid_station_col: "StationId",
                    pwr_col: "power_kw",
                }, inplace=True)
            else:
                continue

            # Merge turbine + grid
            merged = df_turb_slim.merge(df_grid_slim, on=["time", "StationId"], how="inner")

            # Extract temperature if available
            if df_temp is not None:
                temp_col = None
                for c in df_temp.columns:
                    if "AmbieTmp_mean" in c:
                        temp_col = c
                        break
                if temp_col:
                    temp_time_col = "PCTimeStamp" if "PCTimeStamp" in df_temp.columns else df_temp.columns[0]
                    temp_station_col = "StationId" if "StationId" in df_temp.columns else "stationid"
                    df_temp_slim = df_temp[[temp_time_col, temp_station_col, temp_col]].copy()
                    df_temp_slim.rename(columns={
                        temp_time_col: "time",
                        temp_station_col: "StationId",
                        temp_col: "ambient_temp_c",
                    }, inplace=True)
                    merged = merged.merge(df_temp_slim, on=["time", "StationId"], how="left")

            frames.append(merged)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Map to WindFM format
    pressure_pa = pressure_from_elevation(AVG_ELEVATION)

    result = pd.DataFrame()
    result["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    result["TurbID"] = df["StationId"].map(station_map).fillna(df["StationId"].astype(str))
    result["wind_speed"] = pd.to_numeric(df.get("wind_speed_raw", pd.Series(dtype=float)), errors="coerce")
    result["wind_direction"] = pd.to_numeric(df.get("nacelle_pos", pd.Series(dtype=float)), errors="coerce")
    result["power"] = pd.to_numeric(df["power_kw"], errors="coerce") / 1000.0  # kW → MW

    if "ambient_temp_c" in df.columns:
        temp_c = pd.to_numeric(df["ambient_temp_c"], errors="coerce")
    else:
        temp_c = pd.Series(7.0, index=df.index)  # Scottish average

    result["temperature"] = temp_c + 273.15
    result["pressure"] = pressure_pa
    result["density"] = pressure_pa / (R_D * result["temperature"])

    return result


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
    print("Loading Hill of Towie data...")
    zip_files = sorted(RAW_DIR.glob("hot*.zip"))
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
