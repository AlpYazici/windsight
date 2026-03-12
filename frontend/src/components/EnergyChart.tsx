"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

interface EnergyChartProps {
  dailyEnergy: number[];
  startTime: string;
}

export default function EnergyChart({ dailyEnergy, startTime }: EnergyChartProps) {
  const start = new Date(startTime);
  const avg = dailyEnergy.reduce((a, b) => a + b, 0) / dailyEnergy.length;

  const chartData = dailyEnergy.map((e, i) => {
    const d = new Date(start);
    d.setDate(d.getDate() + i);
    return {
      day: d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }),
      energy: e,
      aboveAvg: e >= avg,
    };
  });

  return (
    <div className="bg-white rounded-xl border border-zinc-100 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-zinc-700 mb-4">
        Daily Energy Production
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 15, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f1f1" vertical={false} />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 11, fill: "#999" }}
            stroke="#e5e5e5"
          />
          <YAxis
            tick={{ fontSize: 11, fill: "#999" }}
            label={{
              value: "MWh",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 11, fill: "#999" },
            }}
            stroke="#e5e5e5"
          />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(2)} MWh`, "Energy"]}
            contentStyle={{
              borderRadius: 10,
              border: "1px solid #e5e5e5",
              fontSize: 12,
              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
            }}
          />
          <ReferenceLine
            y={avg}
            stroke="#94a3b8"
            strokeDasharray="4 4"
            label={{
              value: `Avg: ${avg.toFixed(1)}`,
              position: "right",
              style: { fontSize: 10, fill: "#94a3b8" },
            }}
          />
          <Bar dataKey="energy" radius={[6, 6, 0, 0]} maxBarSize={48}>
            {chartData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.aboveAvg ? "#2563eb" : "#94a3b8"}
                opacity={entry.aboveAvg ? 1 : 0.6}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
