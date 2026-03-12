"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Wind, Loader2, AlertCircle, Globe, CheckCircle2 } from "lucide-react";
import type { TurbineDetail, ForecastResponse, SelectedLocation } from "@/lib/types";
import { fetchTurbines, fetchForecast, checkHealth } from "@/lib/api";

import LocationSearch from "@/components/LocationSearch";
import TurbineSelector from "@/components/TurbineSelector";
import MetricCards from "@/components/MetricCards";
import ForecastChart from "@/components/ForecastChart";
import EnergyChart from "@/components/EnergyChart";
import WindChart from "@/components/WindChart";

const Map = dynamic(() => import("@/components/Map"), { ssr: false });

export default function Home() {
  const [turbines, setTurbines] = useState<TurbineDetail[]>([]);
  const [selectedTurbine, setSelectedTurbine] = useState("");
  const [location, setLocation] = useState<SelectedLocation>({
    lat: 55.95,
    lon: -3.19,
    name: "Edinburgh, Scotland",
  });
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth().then(setApiOnline);
    fetchTurbines()
      .then((t) => {
        setTurbines(t);
        if (t.length > 0) {
          const vestas = t.find((x) => x.full_name.includes("V90"));
          setSelectedTurbine(vestas?.full_name || t[0].full_name);
        }
      })
      .catch(() => setApiOnline(false));
  }, []);

  const handleMapClick = useCallback((lat: number, lon: number) => {
    setLocation({ lat, lon, name: `${lat}, ${lon}` });
  }, []);

  const handleCitySelect = useCallback((lat: number, lon: number, name: string) => {
    setLocation({ lat, lon, name });
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!selectedTurbine) return;
    setLoading(true);
    setError(null);
    setForecast(null);
    try {
      const result = await fetchForecast(location.lat, location.lon, selectedTurbine);
      setForecast(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Forecast failed");
    } finally {
      setLoading(false);
    }
  }, [location, selectedTurbine]);

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* ---- Header ---- */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-zinc-100">
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
              <Wind className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-zinc-800 leading-tight">
                WindSight
              </h1>
              <p className="text-[11px] text-zinc-400 leading-tight">
                AI-Powered Wind Power Forecasting
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs">
            {apiOnline === null ? (
              <span className="text-zinc-400">Checking API...</span>
            ) : apiOnline ? (
              <span className="flex items-center gap-1.5 text-emerald-600 bg-emerald-50 px-3 py-1.5 rounded-full">
                <CheckCircle2 className="w-3.5 h-3.5" /> API Connected
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-red-500 bg-red-50 px-3 py-1.5 rounded-full">
                <AlertCircle className="w-3.5 h-3.5" /> API Offline
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-6 py-6">
        {/* ---- Map + Controls ---- */}
        <div className="relative">
          {/* Full-width map */}
          <div className="h-[560px] rounded-2xl overflow-hidden border border-zinc-200 shadow-sm">
            <Map
              lat={location.lat}
              lon={location.lon}
              onLocationSelect={handleMapClick}
            />
          </div>

          {/* Floating control panel */}
          <div className="absolute top-4 left-4 w-[340px] bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg border border-zinc-200/50 p-5 space-y-4 z-10">
            {/* Location search */}
            <div>
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Location
              </label>
              <LocationSearch onSelect={handleCitySelect} />
              <div className="mt-2 flex items-center gap-1.5 text-xs text-zinc-500">
                <Globe className="w-3 h-3" />
                <span className="font-medium">
                  {location.name || `${location.lat}, ${location.lon}`}
                </span>
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5 ml-4">
                or click anywhere on the map
              </div>
            </div>

            {/* Turbine selector */}
            <div>
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Turbine Model
              </label>
              {turbines.length > 0 ? (
                <TurbineSelector
                  turbines={turbines}
                  selected={selectedTurbine}
                  onSelect={setSelectedTurbine}
                />
              ) : (
                <div className="py-6 text-center text-sm text-zinc-400">
                  {apiOnline === false
                    ? "Start API: uvicorn api:app"
                    : "Loading turbines..."}
                </div>
              )}
            </div>

            {/* Generate button */}
            <button
              onClick={handleGenerate}
              disabled={loading || !selectedTurbine || apiOnline === false}
              className="w-full py-3.5 bg-blue-600 text-white text-sm font-semibold rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-md shadow-blue-200/50"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Forecast"
              )}
            </button>

            {loading && (
              <div className="text-xs text-zinc-400 text-center">
                WindFM inference running...
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-100 rounded-xl text-xs text-red-600">
                <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                {error}
              </div>
            )}
          </div>

          {/* Coordinates badge */}
          <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-sm text-white text-xs px-3 py-1.5 rounded-lg z-10">
            {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
          </div>
        </div>

        {/* ---- Results ---- */}
        {forecast && (
          <div className="mt-8 space-y-5">
            <MetricCards data={forecast} />

            <ForecastChart
              data={forecast.forecast}
              ratedPower={forecast.turbine.rated_power_kw}
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <EnergyChart
                dailyEnergy={forecast.daily_energy_mwh}
                startTime={forecast.forecast[0]?.time || ""}
              />
              <WindChart data={forecast.forecast} />
            </div>

            <div className="text-xs text-zinc-400 text-center py-3">
              Generated {new Date(forecast.generated_at).toLocaleString()} &middot;{" "}
              {forecast.forecast.length}h forecast &middot; Powered by WindFM
            </div>
          </div>
        )}

        {/* ---- Empty state ---- */}
        {!forecast && !loading && (
          <div className="mt-16 text-center">
            <Wind className="w-16 h-16 text-zinc-200 mx-auto mb-4" />
            <p className="text-zinc-400">
              Select a location and turbine, then click{" "}
              <strong className="text-zinc-600">Generate Forecast</strong>
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
