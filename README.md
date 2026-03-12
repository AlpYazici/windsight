# WindSight

Wind power generation forecasting for any location worldwide, powered by the [WindFM](https://github.com/shiyu-coder/WindFM) foundation model.

Given a coordinate and a turbine model, WindSight fetches real-time weather data from Open-Meteo, generates a synthetic power history from the turbine's power curve, runs probabilistic autoregressive inference through WindFM, and returns a 7-day hourly power forecast with confidence intervals.

## Architecture

```
Location (lat/lon) + Turbine selection
  -> Open-Meteo API (10-day history + 7-day forecast)
  -> Synthetic power history via turbine power curve
  -> WindFM inference (autoregressive, probabilistic)
  -> Post-processing (cut-in/cut-out, capacity limits)
  -> FastAPI backend + React frontend
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- ~2 GB disk space (model weights + dependencies)

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/AlpYazici/windsight.git
cd windsight
```

### 2. Download Model Weights

WindFM weights are not included in the repository. Download them from HuggingFace:

```bash
mkdir -p models/windfm models/windfm-tokenizer

# Model (~16 MB)
huggingface-cli download NeoQuasar/WindFM --local-dir models/windfm

# Tokenizer (~15 MB)
huggingface-cli download NeoQuasar/WindFM-Tokenizer --local-dir models/windfm-tokenizer
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn api:app --port 8000
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev -- -p 3001
```

Open http://localhost:3001 in your browser.

## Usage

### Web Interface

1. Search for a city or click on the map to select a location
2. Choose a turbine model from the dropdown (13 models available)
3. Click **Generate Forecast**
4. View 7-day hourly power forecast with P5-P95 confidence bands

### API

```bash
# Health check
curl http://localhost:8000/health

# List available turbines
curl http://localhost:8000/turbines

# Generate forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"lat": 55.95, "lon": -3.19, "turbine_model": "Vestas V90"}'
```

The forecast response includes:
- 168 hourly timesteps with P5/P25/P50/P75/P95 percentiles (kW) — confidence bands are estimated from wind speed uncertainty, not from multiple model samples. Set `SAMPLE_COUNT >= 20` in `src/pipeline/predictor.py` for true probabilistic bands.
- Daily energy production (MWh)
- Capacity factor
- Location metadata (elevation)

### Python

```python
from src.pipeline.predictor import predict

result = predict(lat=55.95, lon=-3.19, turbine_model="Vestas V90")
print(f"Capacity factor: {result.capacity_factor:.1%}")
print(f"7-day energy: {sum(result.daily_energy_mwh):.1f} MWh")
```

## WindFM Model

WindSight uses [WindFM](https://github.com/shiyu-coder/WindFM) (Shi et al., 2025), an 8.1M parameter foundation model for wind power forecasting.

- **Architecture**: BSQ tokenizer (3.96M params) + autoregressive Transformer (4.1M params)
- **Input**: 6 features per timestep — wind speed (m/s), wind direction (deg), power (MW), air density (kg/m3), temperature (K), pressure (Pa)
- **Normalization**: Per-window z-score (automatic)
- **Context**: Up to 512 hourly timesteps
- **Inference**: Autoregressive token-by-token generation with nucleus sampling
- **Device**: Supports CPU, CUDA, and Apple Silicon MPS (2.6x faster than CPU)

The model is used **zero-shot** — no fine-tuning required. It generalizes to any location worldwide using weather data from Open-Meteo.

## Evaluation

Evaluated on held-out datasets not seen during WindFM's original training:

### Hill of Towie (Scotland, 21 x Siemens SWT-2.3, 2020)

| Horizon | MAE (MW) | RMSE (MW) | R² | CRPS | Cov90 |
|---------|----------|-----------|------|------|-------|
| 24h | 0.323 | 0.460 | 0.578 | 0.231 | 89.6% |
| 48h | 0.387 | 0.535 | 0.382 | 0.275 | 89.7% |
| 72h | 0.429 | 0.602 | 0.308 | 0.303 | 89.1% |
| 168h | 0.506 | 0.683 | 0.200 | 0.345 | 87.3% |

### SDWPF (China, 134 x Sinovel 1.5MW, 2020-2021)

| Horizon | MAE (MW) | RMSE (MW) | R² | CRPS | Cov90 |
|---------|----------|-----------|------|------|-------|
| 24h | 0.264 | 0.387 | 0.107 | 0.186 | 77.2% |
| 48h | 0.245 | 0.388 | 0.061 | 0.179 | 78.7% |
| 72h | 0.244 | 0.392 | -0.139 | 0.179 | 83.7% |
| 168h | 0.243 | 0.415 | -0.147 | 0.186 | 83.6% |

*SDWPF R² is low because 37% of timesteps have zero power output and 57% are below 0.1 MW — typical for wind data with frequent low-wind periods.*

## Available Turbines

13 turbine models with full power curves:

| Manufacturer | Model | Rated Power | Rotor | Hub Height |
|-------------|-------|-------------|-------|------------|
| Enercon | E-126 EP4 | 4,200 kW | 127 m | 135 m |
| GE | 1.5sle | 1,500 kW | 77 m | 80 m |
| GE | 2.5-120 | 2,500 kW | 120 m | 85 m |
| Goldwind | GW121/2500 | 2,500 kW | 121 m | 90 m |
| Nordex | N100/2500 | 2,500 kW | 100 m | 100 m |
| Senvion | MM82 | 2,050 kW | 82 m | 80 m |
| Senvion | MM92 | 2,050 kW | 92 m | 80 m |
| Siemens | SWT-2.3-93 | 2,300 kW | 93 m | 80 m |
| Sinovel | SL1500/82 | 1,500 kW | 82 m | 65 m |
| Vestas | V90-2.0 | 2,000 kW | 90 m | 80 m |
| Vestas | V110-2.0 | 2,000 kW | 110 m | 80 m |
| Vestas | V126-3.45 | 3,450 kW | 126 m | 117 m |

## Project Structure

```
windsight/
├── api.py                    # FastAPI backend
├── frontend/                 # Next.js + React frontend
│   └── src/
│       ├── app/page.tsx      # Main dashboard
│       ├── components/       # Map, charts, selectors
│       └── lib/              # API client, types
├── src/
│   ├── api/weather.py        # Open-Meteo client
│   ├── data/turbine_db.py    # Turbine power curves
│   ├── models/windfm_wrapper.py  # WindFM inference wrapper
│   ├── pipeline/
│   │   ├── predictor.py      # End-to-end prediction pipeline
│   │   └── synthetic_power.py
│   ├── training/             # Fine-tuning scripts (optional)
│   └── evaluation/metrics.py # MAE, RMSE, CRPS, calibration
├── scripts/evaluate.py       # Cross-geography evaluation
├── data/turbine_specs.json   # 13 turbine power curves
├── WindFM/                   # WindFM model code (submodule)
└── models/                   # Model weights (not in repo)
```

## Data Sources

- **Weather**: [Open-Meteo](https://open-meteo.com/) — free, no API key required
- **Model**: [WindFM](https://huggingface.co/NeoQuasar/WindFM) — pre-trained foundation model
- **Turbine specs**: Manufacturer datasheets, IEC standards

### Training/Evaluation Datasets (not included)

| Dataset | Turbines | Location | Source |
|---------|----------|----------|--------|
| SDWPF | 134 x Sinovel 1.5MW | China | Baidu KDD Cup 2022 |
| Kelmarsh | 6 x Senvion MM92 | England | Zenodo |
| Penmanshiel | 14 x Senvion MM82 | Scotland | Zenodo |
| Hill of Towie | 21 x Siemens SWT-2.3 | Scotland | Zenodo |

## Fine-Tuning (Optional)

WindSight includes scripts for fine-tuning WindFM on custom datasets:

```bash
# Full fine-tuning
python -m src.training.finetune \
  --train_data data/sdwpf/processed/train.parquet \
  --val_data data/sdwpf/processed/val.parquet \
  --output_dir outputs/finetuned

# LoRA fine-tuning (fewer parameters, preserves base model)
python -m src.training.finetune_lora \
  --train_data data/sdwpf/processed/train.parquet \
  --val_data data/sdwpf/processed/val.parquet \
  --rank 8 --alpha 16 --output_dir outputs/lora
```

**Note**: In our experiments, the zero-shot model performed comparably to fine-tuned variants. Full fine-tuning degraded performance due to teacher-forcing / autoregressive mismatch. LoRA showed marginal improvement (+2% R² on Hill of Towie 24h) but the difference is within noise.

## License

MIT
