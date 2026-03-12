# WindSight

## Project Goal
Wind power generation forecasting for any location worldwide using WindFM foundation model + Open-Meteo weather API.

User selects a location on the map, picks a turbine model → gets a 7-day hourly power forecast with confidence intervals.

## Architecture
```
Location (lat/lon) + Turbine selection
  → Open-Meteo Forecast API (10-day history via past_days + 7-day forecast)
  → Synthetic power history via turbine power curve
  → WindFM inference (1 sample, autoregressive)
  → Post-processing (cut-in/cut-out, capacity limits, synthetic confidence bands)
  → FastAPI backend + React/Next.js frontend
```

## Key Decisions
- WindFM used **zero-shot** — no fine-tuning needed. Fine-tuning was tested (SDWPF, 28 epochs) but degraded performance due to teacher-forcing/autoregressive mismatch. LoRA showed only marginal improvement (+2% R²).
- Single sample inference (SAMPLE_COUNT=1) to keep memory usage minimal (~2 GB). Confidence bands (P5/P25/P75/P95) are synthetically estimated from wind speed uncertainty.
- Synthetic power history: no real power data at new locations, so it's generated from the turbine's power curve.
- Weather data fetched via Open-Meteo's forecast API `past_days` parameter (up to 92 days) to avoid archive API's multi-day data delay that caused NaN issues.

## WindFM Technical
- Model: `NeoQuasar/WindFM` (16 MB, 8.1M params)
- Tokenizer: `NeoQuasar/WindFM-Tokenizer` (15 MB)
- Repo: https://github.com/shiyu-coder/WindFM
- Input: 6 features [wind_speed, wind_direction, power, density, temperature, pressure] + UTC timestamp
- Units: m/s, degrees, MW, kg/m³, Kelvin, Pascal
- Normalization: per-window z-score (automatic)
- Power = feature index 2
- Max context: 512 timesteps
- Device: CPU, CUDA, Apple Silicon MPS (2.6x faster than CPU)

## Model Files
- WindFM code: `WindFM/` (git submodule)
- Model weights: `models/windfm/model.safetensors`
- Tokenizer weights: `models/windfm-tokenizer/model.safetensors`

## Stack
- **Backend**: FastAPI (`api.py`) on port 8000
- **Frontend**: Next.js 16 + React + TypeScript + Tailwind CSS (`frontend/`) on port 3001
- **Map**: react-leaflet with CARTO light tiles (no API key)
- **Charts**: recharts (forecast, energy, uncertainty)
- **Weather**: Open-Meteo (free, no API key)

## Evaluation Datasets
| Dataset | Turbines | Location | Purpose |
|---|---|---|---|
| SDWPF | 134 x Sinovel 1.5MW | China | Training/validation |
| Kelmarsh | 6 x Senvion MM92 | England | Evaluation |
| Penmanshiel | 14 x Senvion MM82 | Scotland | Evaluation |
| Hill of Towie | 21 x Siemens SWT-2.3 | Scotland | Held-out test |
