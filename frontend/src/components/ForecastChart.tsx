"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ForecastTimeStep } from "@/lib/types";

interface ForecastChartProps {
  data: ForecastTimeStep[];
  ratedPower: number;
}

export default function ForecastChart({ data, ratedPower }: ForecastChartProps) {
  const chartData = data.map((d) => ({
    time: new Date(d.time).getTime(),
    p5: d.p5,
    p25: d.p25,
    p50: d.p50,
    p75: d.p75,
    p95: d.p95,
  }));

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
  };

  const formatHour = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  };

  return (
    <div className="bg-white rounded-xl border border-zinc-100 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-zinc-700 mb-4">
        7-Day Power Forecast
      </h3>
      <ResponsiveContainer width="100%" height={360}>
        <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="gradP5P95" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.08} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="gradP25P75" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f1f1" />
          <XAxis
            dataKey="time"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={formatTime}
            tick={{ fontSize: 11, fill: "#999" }}
            tickCount={7}
            stroke="#e5e5e5"
          />
          <YAxis
            tick={{ fontSize: 11, fill: "#999" }}
            tickFormatter={(v: number) => `${v}`}
            label={{
              value: "Power (kW)",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 11, fill: "#999" },
            }}
            domain={[0, Math.ceil(ratedPower * 1.05)]}
            stroke="#e5e5e5"
          />
          <Tooltip
            labelFormatter={(ts) => {
              const d = new Date(Number(ts));
              return d.toLocaleString("en-US", {
                weekday: "short",
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              });
            }}
            formatter={(value, name) => [
              `${Number(value).toFixed(0)} kW`,
              String(name).toUpperCase(),
            ]}
            contentStyle={{
              borderRadius: 10,
              border: "1px solid #e5e5e5",
              fontSize: 12,
              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
            }}
          />
          <Legend
            verticalAlign="top"
            height={36}
            iconType="plainline"
            wrapperStyle={{ fontSize: 12 }}
          />
          {/* P5-P95 band */}
          <Area
            type="monotone"
            dataKey="p95"
            stroke="none"
            fill="url(#gradP5P95)"
            name="p95"
            legendType="none"
          />
          <Area
            type="monotone"
            dataKey="p5"
            stroke="none"
            fill="#fff"
            name="p5"
            legendType="none"
          />
          {/* P25-P75 band */}
          <Area
            type="monotone"
            dataKey="p75"
            stroke="none"
            fill="url(#gradP25P75)"
            name="P25-P75"
          />
          <Area
            type="monotone"
            dataKey="p25"
            stroke="none"
            fill="#fff"
            name="p25"
            legendType="none"
          />
          {/* P50 line */}
          <Area
            type="monotone"
            dataKey="p50"
            stroke="#2563eb"
            strokeWidth={2}
            fill="none"
            name="P50 (median)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
