"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

interface MapProps {
  lat: number;
  lon: number;
  onLocationSelect: (lat: number, lon: number) => void;
}

export default function Map({ lat, lon, onLocationSelect }: MapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      zoomControl: false,
    }).setView([lat, lon], 5);

    L.control.zoom({ position: "bottomright" }).addTo(map);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    const icon = L.divIcon({
      html: `<div style="
        width: 20px; height: 20px;
        background: #2563eb;
        border: 3px solid white;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      "></div>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
      className: "",
    });

    const marker = L.marker([lat, lon], { icon }).addTo(map);
    markerRef.current = marker;

    map.on("click", (e: L.LeafletMouseEvent) => {
      const { lat: newLat, lng: newLon } = e.latlng;
      marker.setLatLng([newLat, newLon]);
      onLocationSelect(
        Math.round(newLat * 10000) / 10000,
        Math.round(newLon * 10000) / 10000
      );
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.setLatLng([lat, lon]);
    }
    if (mapRef.current) {
      mapRef.current.setView([lat, lon], mapRef.current.getZoom(), {
        animate: true,
      });
    }
  }, [lat, lon]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full rounded-xl overflow-hidden"
      style={{ minHeight: 400 }}
    />
  );
}
