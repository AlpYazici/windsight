import type { TurbineDetail, ForecastResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchTurbines(): Promise<TurbineDetail[]> {
  const res = await fetch(`${API_BASE}/turbines`);
  if (!res.ok) throw new Error("Failed to fetch turbines");
  return res.json();
}

export async function fetchForecast(
  lat: number,
  lon: number,
  turbineModel: string
): Promise<ForecastResponse> {
  const res = await fetch(`${API_BASE}/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lon, turbine_model: turbineModel }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Forecast request failed");
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
