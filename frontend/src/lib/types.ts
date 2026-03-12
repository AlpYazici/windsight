export interface TurbineDetail {
  full_name: string;
  manufacturer: string;
  model: string;
  rated_power_kw: number;
  rotor_diameter_m: number;
  hub_height_m: number;
  cut_in_speed_ms: number;
  rated_speed_ms: number;
  cut_out_speed_ms: number;
  swept_area_m2: number;
}

export interface ForecastTimeStep {
  time: string;
  power_kw: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

export interface LocationInfo {
  lat: number;
  lon: number;
  elevation_m: number;
}

export interface TurbineInfo {
  model: string;
  rated_power_kw: number;
}

export interface ForecastResponse {
  location: LocationInfo;
  turbine: TurbineInfo;
  forecast: ForecastTimeStep[];
  daily_energy_mwh: number[];
  capacity_factor: number;
  generated_at: string;
}

export interface SelectedLocation {
  lat: number;
  lon: number;
  name?: string;
}
