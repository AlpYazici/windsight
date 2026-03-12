"use client";

import { Zap, TrendingUp, Battery, BarChart3 } from "lucide-react";
import type { ForecastResponse } from "@/lib/types";

interface MetricCardsProps {
  data: ForecastResponse;
}

export default function MetricCards({ data }: MetricCardsProps) {
  const currentPower = data.forecast[0]?.p50 ?? 0;
  const rated = data.turbine.rated_power_kw;
  const pctOfRated = rated > 0 ? (currentPower / rated) * 100 : 0;
  const totalEnergy = data.daily_energy_mwh.reduce((a, b) => a + b, 0);
  const avgDaily = totalEnergy / 7;
  const cf = data.capacity_factor * 100;

  const cfLabel =
    cf < 15 ? "Low" : cf < 25 ? "Moderate" : cf < 40 ? "Good" : "Excellent";
  const cfColor =
    cf < 15
      ? "text-orange-500"
      : cf < 25
        ? "text-yellow-500"
        : cf < 40
          ? "text-emerald-500"
          : "text-blue-500";

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <Card
        icon={<Zap className="w-5 h-5 text-amber-500" />}
        label="Current Predicted"
        value={`${currentPower.toLocaleString("en-US", { maximumFractionDigits: 0 })} kW`}
        sub={`${pctOfRated.toFixed(0)}% of rated`}
      />
      <Card
        icon={<TrendingUp className="w-5 h-5 text-blue-500" />}
        label="Capacity Factor"
        value={`${cf.toFixed(1)}%`}
        sub={<span className={cfColor}>{cfLabel}</span>}
      />
      <Card
        icon={<Battery className="w-5 h-5 text-emerald-500" />}
        label="7-Day Energy"
        value={`${totalEnergy.toFixed(1)} MWh`}
        sub={`${data.turbine.model}`}
      />
      <Card
        icon={<BarChart3 className="w-5 h-5 text-violet-500" />}
        label="Avg Daily"
        value={`${avgDaily.toFixed(1)} MWh/d`}
        sub={`${data.forecast.length}h forecast`}
      />
    </div>
  );
}

function Card({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-zinc-100 p-4 shadow-sm">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-zinc-400 font-medium uppercase tracking-wide">
          {label}
        </span>
      </div>
      <div className="text-2xl font-bold text-zinc-800">{value}</div>
      <div className="text-xs text-zinc-400 mt-0.5">{sub}</div>
    </div>
  );
}
