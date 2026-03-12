"use client";

import { useState, useRef, useEffect } from "react";
import { Search, MapPin, X } from "lucide-react";

interface LocationSearchProps {
  onSelect: (lat: number, lon: number, name: string) => void;
}

interface GeoResult {
  name: string;
  latitude: number;
  longitude: number;
  country?: string;
  admin1?: string;
}

export default function LocationSearch({ onSelect }: LocationSearchProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<GeoResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  async function search() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=5&language=en`
      );
      const data = await res.json();
      setResults(data.results || []);
      setOpen(true);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") search();
  }

  function handleSelect(r: GeoResult) {
    const label = [r.name, r.admin1, r.country].filter(Boolean).join(", ");
    onSelect(r.latitude, r.longitude, label);
    setQuery(label);
    setOpen(false);
  }

  return (
    <div ref={wrapperRef} className="relative">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search city..."
            className="w-full pl-9 pr-8 py-2.5 bg-white border border-zinc-200 rounded-xl text-sm text-zinc-800 placeholder:text-zinc-400 focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
          />
          {query && (
            <button
              onClick={() => {
                setQuery("");
                setResults([]);
                setOpen(false);
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-300 hover:text-zinc-500"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <button
          onClick={search}
          disabled={loading || !query.trim()}
          className="px-4 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "..." : "Go"}
        </button>
      </div>

      {open && results.length > 0 && (
        <div className="absolute z-50 mt-1 w-full bg-white border border-zinc-200 rounded-xl shadow-lg overflow-hidden">
          {results.map((r, i) => (
            <button
              key={i}
              onClick={() => handleSelect(r)}
              className="w-full flex items-start gap-2.5 px-4 py-3 text-left hover:bg-blue-50 transition-colors border-b border-zinc-50 last:border-0"
            >
              <MapPin className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
              <div>
                <div className="text-sm font-medium text-zinc-800">
                  {r.name}
                </div>
                <div className="text-xs text-zinc-400">
                  {[r.admin1, r.country].filter(Boolean).join(", ")} &middot;{" "}
                  {r.latitude.toFixed(2)}, {r.longitude.toFixed(2)}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
