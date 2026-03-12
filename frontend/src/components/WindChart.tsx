"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ForecastTimeStep } from "@/lib/types";

interface WindChartProps {
  data: ForecastTimeStep[];
}

export default function WindChart({ data }: WindChartProps) {
  // We don't have direct wind data from the forecast endpoint,
  // but we can show power percentiles spread as a proxy for wind variability.
  // For a real wind chart we'd need wind data from a separate endpoint.
  // For now, show the P5-P95 spread ratio as "uncertainty"
  const chartData = data.map((d) => {
    const spread = d.p95 - d.p5;
    const uncertainty = d.p50 > 0 ? (spread / d.p50) * 100 : 0;
    return {
      time: new Date(d.time).getTime(),
      p50: d.p50,
      spread,
      uncertainty: Math.min(uncertainty, 500),
    };
  });

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="bg-white rounded-xl border border-zinc-100 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-zinc-700 mb-4">
        Forecast Uncertainty
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
            yAxisId="power"
            tick={{ fontSize: 11, fill: "#999" }}
            label={{
              value: "kW",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 11, fill: "#999" },
            }}
            stroke="#e5e5e5"
          />
          <YAxis
            yAxisId="spread"
            orientation="right"
            tick={{ fontSize: 11, fill: "#999" }}
            label={{
              value: "P95-P5 spread (kW)",
              angle: 90,
              position: "insideRight",
              style: { fontSize: 11, fill: "#999" },
            }}
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
            formatter={(value, name) => {
              const v = Number(value);
              if (name === "spread") return [`${v.toFixed(0)} kW`, "P95-P5 Spread"];
              return [`${v.toFixed(0)} kW`, "P50 Power"];
            }}
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
          <Line
            yAxisId="power"
            type="monotone"
            dataKey="p50"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            name="P50 Power"
          />
          <Line
            yAxisId="spread"
            type="monotone"
            dataKey="spread"
            stroke="#f59e0b"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            name="P95-P5 Spread"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
