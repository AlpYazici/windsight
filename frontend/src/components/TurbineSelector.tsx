"use client";

import { useState } from "react";
import { ChevronDown, Wind, Gauge, ArrowUpDown, Circle } from "lucide-react";
import type { TurbineDetail } from "@/lib/types";

interface TurbineSelectorProps {
  turbines: TurbineDetail[];
  selected: string;
  onSelect: (name: string) => void;
}

export default function TurbineSelector({
  turbines,
  selected,
  onSelect,
}: TurbineSelectorProps) {
  const [open, setOpen] = useState(false);
  const current = turbines.find((t) => t.full_name === selected);

  const grouped = turbines.reduce(
    (acc, t) => {
      if (!acc[t.manufacturer]) acc[t.manufacturer] = [];
      acc[t.manufacturer].push(t);
      return acc;
    },
    {} as Record<string, TurbineDetail[]>
  );

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-white border border-zinc-200 rounded-xl hover:border-blue-300 transition-colors text-sm"
      >
        <div className="flex items-center gap-2">
          <Wind className="w-4 h-4 text-blue-600" />
          <span className="font-medium text-zinc-800">
            {current?.full_name || "Select turbine"}
          </span>
        </div>
        <ChevronDown
          className={`w-4 h-4 text-zinc-400 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="absolute z-50 mt-1 w-full bg-white border border-zinc-200 rounded-xl shadow-lg max-h-80 overflow-y-auto">
          {Object.entries(grouped)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([mfr, models]) => (
              <div key={mfr}>
                <div className="px-3 py-1.5 text-xs font-semibold text-zinc-400 uppercase tracking-wide bg-zinc-50">
                  {mfr}
                </div>
                {models
                  .sort((a, b) => a.model.localeCompare(b.model))
                  .map((t) => (
                    <button
                      key={t.full_name}
                      onClick={() => {
                        onSelect(t.full_name);
                        setOpen(false);
                      }}
                      className={`w-full text-left px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors flex items-center justify-between ${
                        t.full_name === selected
                          ? "bg-blue-50 text-blue-700"
                          : "text-zinc-700"
                      }`}
                    >
                      <span>{t.model}</span>
                      <span className="text-xs text-zinc-400">
                        {t.rated_power_kw} kW
                      </span>
                    </button>
                  ))}
              </div>
            ))}
        </div>
      )}

      {current && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          <Spec icon={<Gauge className="w-3.5 h-3.5" />} label="Rated" value={`${current.rated_power_kw} kW`} />
          <Spec icon={<ArrowUpDown className="w-3.5 h-3.5" />} label="Hub" value={`${current.hub_height_m} m`} />
          <Spec icon={<Wind className="w-3.5 h-3.5" />} label="Cut-in" value={`${current.cut_in_speed_ms} m/s`} />
          <Spec icon={<Circle className="w-3.5 h-3.5" />} label="Rotor" value={`${current.rotor_diameter_m} m`} />
        </div>
      )}
    </div>
  );
}

function Spec({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-zinc-50 rounded-lg text-xs">
      <span className="text-zinc-400">{icon}</span>
      <span className="text-zinc-400">{label}</span>
      <span className="font-medium text-zinc-700 ml-auto">{value}</span>
    </div>
  );
}
