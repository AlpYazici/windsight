# PRD: WindSight — Wind Power Forecasting API & Dashboard

## Introduction

WindSight is a wind power forecasting system that uses the WindFM foundation model to predict wind turbine power output at any given location. Users select a location and turbine model, and the system returns hourly power forecasts with confidence intervals for the next 7 days.

The system operates as a two-layer architecture:
1. **WindFM Foundation Model** — A compact 8.1M-parameter decoder-only Transformer pre-trained on 126,000+ sites, fine-tuned on the SDWPF dataset (134 turbines, 24 months of SCADA data)
2. **Live Prediction Service** — A FastAPI backend + Streamlit dashboard integrating Open-Meteo weather API for real-time, location-based forecasting

**Project type:** MVP product with B2B SaaS potential. Sell accurate wind power forecasts via API subscription to wind farm operators, energy traders, and grid operators.

**Hardware target:** Apple Silicon M3 Ultra with 256GB unified memory, using PyTorch MPS backend.

---

## Goals

- Fine-tune WindFM on the SDWPF dataset (134 turbines, 11.4M records) to improve prediction accuracy
- Validate fine-tuned model on separate open-source SCADA datasets (Kelmarsh, Penmanshiel) to confirm cross-geography generalization
- Build a FastAPI service: location + turbine → 7-day hourly power forecast with confidence intervals
- Build a Streamlit dashboard for interactive exploration
- Full prediction pipeline completes in < 30 seconds for any location
- Run entirely on Apple Silicon M3 Ultra without CUDA dependencies

---

## How It Works

### Prediction Flow

```
1. User inputs: location (lat/lon or city) + turbine model
                    │
2. Open-Meteo API ──┤── Son 10 gün gerçek hava durumu (rüzgar, sıcaklık, basınç)
                    ├── 7 gün hava tahmini
                    └── Yükseklik
                    │
3. Sentetik Güç ────┤── Türbin güç eğrisinden geçmiş güç üretimi hesapla
   Geçmişi Üret     └── WindFM'in beklediği 6 özellik formatına çevir
                    │
4. WindFM ──────────┤── Geçmiş (10 gün hava + sentetik güç) alır
   İnference        ├── Gelecek 7 gün güç tahmini üretir
                    └── 100 örnek → güven aralıkları (P5, P25, P50, P75, P95)
                    │
5. Son İşleme ──────┤── Türbin limitlerini uygula (cut-in, cut-out, rated power)
                    ├── Günlük MWh hesapla
                    └── Kapasite faktörü hesapla
                    │
6. Çıktı: JSON API response + Dashboard
```

### Sentetik Güç Geçmişi (Kritik Detay)

WindFM'in girdisi 6 özellik: `[wind_speed, wind_direction, power, density, temperature, pressure]`

Yeni bir konum için geçmiş `power` verisi yok. Çözüm:
1. Open-Meteo Historical API'den son 10 günün hava durumunu çek
2. Türbin güç eğrisinden her saat için "bu rüzgarda bu türbin şu kadar üretirdi" hesapla
3. Bu sentetik güç verisini diğer 5 hava verisiyle birleştirip WindFM'e girdi olarak ver

Bu şekilde herhangi bir konum için, geçmiş SCADA verisi olmadan tahmin yapılabiliyor.

---

## WindFM Technical Reference

*Paper: "WindFM: An Open-Source Foundation Model for Zero-Shot Wind Power Forecasting" (arXiv:2509.06311)*
*Repository: https://github.com/shiyu-coder/WindFM*

### Architecture
| Component | Specification |
|---|---|
| **Model type** | Decoder-only Transformer (autoregressive) |
| **Total parameters** | 8.1M |
| **Input features (D=6)** | wind_speed, wind_direction, power, density, temperature, pressure |
| **AR Transformer layers** | 4 |
| **Model dimension** | 256 |
| **FFN dimension** | 512 (SwiGLU) |
| **Attention heads** | 8 (head_dim=32) |
| **Dropout** | attn=0.1, ffn=0.2, residual=0.2 |
| **Positional encoding** | RoPE (Rotary Position Embeddings) |
| **Normalization** | Pre-LN with RMSNorm |
| **Max sequence length** | 512 time steps |

### Tokenizer
| Component | Specification |
|---|---|
| **Type** | Transformer autoencoder with BSQ (Binary Spherical Quantization) |
| **Encoder/Decoder layers** | 4 each |
| **Dimension** | 256, FFN 512, 4 heads |
| **Total bit length (k)** | 20 |
| **Coarse subtoken (s1)** | 10 bits → vocab 1,024 |
| **Fine subtoken (s2)** | 10 bits → vocab 1,024 |
| **BSQ commitment weight** | β=0.05 |

### Input/Output Format

**Input DataFrame columns (exact names, hardcoded):**
```python
feature_cols = ['wind_speed', 'wind_direction', 'power', 'density', 'temperature', 'pressure']
```
Plus a `time` column with **UTC timestamps**.

**Units:**
| Feature | Unit | Example Range |
|---|---|---|
| `wind_speed` | m/s | 3–16 |
| `wind_direction` | degrees (0-360) | 0–360 |
| `power` | MW | 1–20 |
| `density` | kg/m³ | 1.18–1.25 |
| `temperature` | K (Kelvin) | 255–310 |
| `pressure` | Pa (Pascals) | 93,000–101,000 |

**Normalization (per-sample z-score, applied automatically):**
```python
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-6)
x = np.clip(x, -5, 5)
```
Predictions are de-normalized back. Only power (feature index 2) is extracted from output.

### Predictor API
```python
from model import WindFM, WindFMTokenizer, WindFMPredictor

tokenizer = WindFMTokenizer.from_pretrained("NeoQuasar/WindFM-Tokenizer")
model = WindFM.from_pretrained("NeoQuasar/WindFM")
predictor = WindFMPredictor(model, tokenizer, device="mps", max_context=512)

pred_df = predictor.predict(
    df=input_dataframe,          # DataFrame with 6 feature columns
    x_timestamp=history_times,   # pd.Series of UTC timestamps (lookback)
    y_timestamp=future_times,    # pd.Series of UTC timestamps (forecast horizon)
    pred_len=168,                # 7 days × 24 hours
    T=1.0,                       # sampling temperature
    top_p=0.9,                   # nucleus sampling
    sample_count=100             # number of probabilistic samples
)
# Output: DataFrame (pred_len × sample_count), columns "pred-0" to "pred-99"
```

### Temporal Encoding
5 features extracted from timestamps, each mapped to 10-dim Fourier features:
- minute (/59), hour (/23), weekday (/6), day ((-1)/30), month ((-1)/11)

### Inference Hyperparameters
| Task | Temperature | Top-p | Samples | Post-processing |
|---|---|---|---|---|
| Deterministic (MAE) | 0.6 | 0.9 | 20 | Mean of samples |
| Deterministic (RMSE) | 0.9 | 0.9 | 20 | Mean of samples |
| Probabilistic | 1.0 | 1.0 | 100 | Empirical distribution → percentiles |

### HuggingFace Assets
| Asset | Path | Size |
|---|---|---|
| Model (public) | `NeoQuasar/WindFM` | 16.4 MB |
| Tokenizer (public) | `NeoQuasar/WindFM-Tokenizer` | 15.8 MB |

Note: `shiyu-coder/WindFM-8.1M` and `shiyu-coder/WindFM-Tokenizer` exist but are private/gated. Use the NeoQuasar paths.

### Fine-Tuning Approach (No existing code in repo — must implement)
1. Freeze tokenizer completely (encoder + decoder + BSQ)
2. Encode training data with frozen tokenizer → s1_ids, s2_ids sequences
3. Train autoregressive Transformer with teacher forcing
4. Loss: `model.head.compute_loss(s1_logits, s2_logits, s1_targets, s2_targets)` — cross-entropy on both subtokens
5. Low learning rate (1e-5 to 5e-5) to avoid destabilizing pre-trained weights
6. Cosine annealing LR scheduler with linear warmup (5% of steps)

### MPS Compatibility
- No MPS-specific code exists in repo (hardcodes `cuda:0`)
- All operations are standard PyTorch (linear, attention, softmax, multinomial) — should work on MPS
- `torch.multinomial` works on MPS
- `scatter_reduce` (used in BSQ entropy loss) may have issues — but only needed during training, not inference
- **Risk:** Must verify in US-001 before proceeding

---

## Data

### Fine-Tuning Dataset: SDWPF
| Field | Details |
|---|---|
| **Source** | China Longyuan Power Group / Baidu KDD Cup 2022 |
| **Download** | https://figshare.com/articles/dataset/SDWPF_dataset/24798654 |
| **Turbines** | 134 (Sinovel SL1500/82: 1.5 MW, 82m rotor, 70m hub) |
| **Resolution** | 10 minutes |
| **Duration** | Full version: Jan 2020 – Dec 2021 (24 months, 11.4M records) |
| **Size** | ~2 GB |
| **License** | CC-BY 4.0 |

**SDWPF Columns:**
| Column | Description |
|---|---|
| TurbID | Turbine ID (1-134) |
| Wspd | Wind speed (m/s) |
| Wdir | Wind direction (degrees) |
| Etmp | Environment temperature (°C) |
| Itmp | Internal nacelle temperature (°C) |
| Ndir | Nacelle direction (degrees) |
| Pab1/2/3 | Pitch angle of blades (degrees) |
| Prtv | Reactive power (kW) |
| Patv | **Active power — target** (kW) |

**Full version extras:** ERA5 weather (T2m, Sp, RelH, Wspd_w, Wdir_w, Tp) + ASTER elevation data.

**WindFM Feature Mapping:**
| WindFM Feature | SDWPF Source | Conversion |
|---|---|---|
| `wind_speed` | `Wspd` | Direct (m/s) |
| `wind_direction` | `Wdir` | Direct (degrees) |
| `power` | `Patv` | ÷ 1000 (kW → MW) |
| `density` | Derived | `Sp / (287.05 × (Etmp + 273.15))` |
| `temperature` | `Etmp` | + 273.15 (°C → K) |
| `pressure` | `Sp` | Direct (Pa) — from ERA5 full version |

### Validation Datasets (Cross-Geography Test)

#### Kelmarsh Wind Farm (England)
| Field | Details |
|---|---|
| **Download** | https://zenodo.org/records/5841834 |
| **Turbines** | 6 (Senvion MM92: 2.05 MW, 92.5m rotor, 69m hub) |
| **Duration** | 2016 – mid-2021 (~5.5 years) |
| **Location** | Northamptonshire, England (~52.40°N, -0.94°W) |
| **Terrain** | Flat farmland |
| **License** | CC-BY 4.0 |
| **Role** | Validate that fine-tuning improves accuracy on an unseen European site with a different turbine model |

#### Penmanshiel Wind Farm (Scotland)
| Field | Details |
|---|---|
| **Download** | https://zenodo.org/records/8253010 |
| **Turbines** | 14 (Senvion MM82: 2.05 MW, 82m rotor) |
| **Duration** | 2016 – mid-2022 (~6.5 years) |
| **Location** | Scottish Borders (~55.87°N, -2.35°W) |
| **Terrain** | Hilly |
| **License** | CC-BY 4.0 |
| **Role** | Validate cross-geography generalization on hilly terrain |

#### Hill of Towie Wind Farm (Scotland) — Stretch Goal
| Field | Details |
|---|---|
| **Download** | https://zenodo.org/records/14870023 |
| **Turbines** | 21 (Siemens SWT-2.3-VS-82: 2.3 MW) |
| **Duration** | Jan 2016 – Aug 2024 (8.7 years) |
| **License** | CC-BY 4.0 |
| **Role** | Additional validation with a third turbine manufacturer (Siemens) |

---

## User Stories

### Phase 1: Foundation (US-001 to US-003)

#### US-001: Set Up WindFM on Apple Silicon
**Description:** Clone WindFM and verify it runs on MPS backend.

**Acceptance Criteria:**
- [ ] WindFM repo cloned, dependencies installed (Python ≥ 3.11)
- [ ] Model + tokenizer downloaded from HuggingFace (`NeoQuasar/WindFM`, `NeoQuasar/WindFM-Tokenizer`)
- [ ] `device="mps"` works for inference
- [ ] Example prediction runs successfully with bundled sample data (`examples/data/121522.csv`)
- [ ] Output is valid (non-NaN, reasonable power values)
- [ ] If MPS fails on specific ops: document which ones, implement CPU fallback
- [ ] Benchmark: inference time for 240-step lookback → 80-step prediction on MPS vs CPU

**Hard gate:** Do not proceed until inference works on MPS (or CPU fallback is acceptable).

#### US-002: Prepare SDWPF for Fine-Tuning
**Description:** Download SDWPF and transform to WindFM format.

**Acceptance Criteria:**
- [ ] SDWPF full dataset downloaded from Figshare
- [ ] Data cleaned: negative power → 0, missing values handled, outlier turbines flagged
- [ ] Mapped to WindFM 6-feature format (see mapping table above)
- [ ] Air density derived: `density = Sp / (287.05 × (Etmp + 273.15))`
- [ ] Temporal split: 70% train, 15% validation, 15% test (by date, not random)
- [ ] Saved as Parquet files in `data/sdwpf/processed/`
- [ ] EDA notebook at `notebooks/01_eda.ipynb`: power curves, wind rose, distributions, missing data report

#### US-003: Fine-Tune WindFM on SDWPF
**Description:** Fine-tune the autoregressive Transformer (freeze tokenizer) on SDWPF training data.

**Acceptance Criteria:**
- [ ] Tokenizer frozen — only AR Transformer weights updated
- [ ] Training loop: encode data → shift tokens → forward with teacher forcing → cross-entropy loss on s1 + s2
- [ ] Hyperparameters: LR=1e-5, cosine annealing, warmup 5%, batch size 64, max 50 epochs, early stopping patience=5
- [ ] Validation loss monitored per epoch; best checkpoint saved
- [ ] Fine-tuned model achieves lower RMSE than zero-shot WindFM on SDWPF test set
- [ ] Training metrics logged (loss, RMSE, MAE per epoch)
- [ ] Checkpoint saved to `outputs/windfm-finetuned/`
- [ ] Training completes within 24 hours on M3 Ultra

### Phase 2: Live Prediction (US-004 to US-007)

#### US-004: Open-Meteo API Client
**Description:** Weather data fetcher for any location.

**Acceptance Criteria:**
- [ ] Module at `src/api/weather.py`
- [ ] Fetches historical weather (last 10 days): wind speed at 10m + 80m, wind direction, temperature, pressure, humidity
- [ ] Fetches forecast (next 7 days): same variables
- [ ] Fetches elevation for coordinates
- [ ] Geocoding: city name → lat/lon
- [ ] Hub-height wind extrapolation via power law: `v_hub = v_ref × (h_hub / h_ref) ^ alpha` where `alpha = ln(v_80/v_10) / ln(80/10)`
- [ ] Air density calculation: `density = (pressure × 100) / (287.05 × (temp + 273.15))`
- [ ] Rate limiting: respect 10,000 calls/day free tier
- [ ] Response caching: 15-minute TTL
- [ ] Returns DataFrame in WindFM 6-feature format (correct units: K, Pa, MW, kg/m³)
- [ ] Unit tests with mocked responses

#### US-005: Turbine Database
**Description:** Turbine specifications and power curves from windpowerlib.

**Acceptance Criteria:**
- [ ] Power curves extracted from windpowerlib/OEDB for common turbine models
- [ ] For Sinovel SL1500/82 (not in OEDB): derive empirical power curve from SDWPF data (Patv vs Wspd)
- [ ] Minimum 12 turbine models:

| Turbine | Rated Power | Rotor | Cut-in | Cut-out |
|---|---|---|---|---|
| Sinovel SL1500/82 | 1.5 MW | 82m | 3.0 m/s | 25.0 m/s |
| Senvion MM82 | 2.05 MW | 82m | 3.5 m/s | 25.0 m/s |
| Senvion MM92 | 2.05 MW | 92.5m | 3.0 m/s | 24.0 m/s |
| Vestas V90-2.0 | 2.0 MW | 90m | 4.0 m/s | 25.0 m/s |
| Vestas V110-2.0 | 2.0 MW | 110m | 3.0 m/s | 20.0 m/s |
| Vestas V126-3.45 | 3.45 MW | 126m | 3.0 m/s | 22.5 m/s |
| Siemens SWT-2.3-93 | 2.3 MW | 93m | 4.0 m/s | 25.0 m/s |
| GE 1.5sle | 1.5 MW | 77m | 3.5 m/s | 25.0 m/s |
| GE 2.5-120 | 2.5 MW | 120m | 3.0 m/s | 25.0 m/s |
| Enercon E-126 EP4 | 4.2 MW | 127m | 3.0 m/s | 34.0 m/s |
| Nordex N100/2500 | 2.5 MW | 100m | 3.0 m/s | 20.0 m/s |
| Goldwind GW121/2500 | 2.5 MW | 121m | 2.8 m/s | 22.0 m/s |

- [ ] Each entry: manufacturer, model, rated_power_kw, rotor_diameter_m, hub_height_m, cut_in/rated/cut_out speeds, swept_area, power_curve (array of wind_speed → power_kw)
- [ ] Stored at `data/turbine_specs.json`
- [ ] Python API:
  - `get_turbine(name) → TurbineSpec`
  - `list_turbines() → List[str]`
  - `estimate_power(name, wind_speed) → float` (interpolated from curve)

#### US-006: Prediction Pipeline
**Description:** End-to-end: location + turbine → forecast.

**Acceptance Criteria:**
- [ ] Module at `src/pipeline/predictor.py`
- [ ] Input: `latitude, longitude, turbine_model_name`
- [ ] Step 1 — Fetch weather (history + forecast) via US-004
- [ ] Step 2 — Extrapolate wind to hub height
- [ ] Step 3 — Generate synthetic power history from turbine power curve
- [ ] Step 4 — Assemble 6-feature DataFrame in WindFM format (correct units)
- [ ] Step 5 — Run WindFM inference (100 samples, T=1.0, top_p=1.0)
- [ ] Step 6 — Extract power predictions, compute percentiles (P5, P25, P50, P75, P95)
- [ ] Step 7 — Clip to [0, rated_power], apply cut-in/cut-out
- [ ] Step 8 — Compute daily energy (MWh) and capacity factor
- [ ] Returns `ForecastResult` dataclass with timestamps, percentiles, daily energy, metadata
- [ ] Total latency < 30 seconds
- [ ] Handles edge cases: zero wind, extreme wind, API failures

#### US-007: FastAPI Service + Streamlit Dashboard
**Description:** API endpoint and visual dashboard.

**Acceptance Criteria:**

**FastAPI (`api.py`):**
- [ ] `POST /forecast` — accepts `{lat, lon, turbine_model}`, returns forecast JSON
- [ ] `GET /turbines` — lists available turbine models
- [ ] `GET /health` — health check
- [ ] Response format:
```json
{
  "location": {"lat": 41.01, "lon": 28.97, "city": "Istanbul", "elevation_m": 40},
  "turbine": {"model": "Vestas V110-2.0", "rated_power_kw": 2000},
  "forecast": [
    {"time": "2026-03-12T00:00Z", "power_kw": 847, "p5": 320, "p25": 520, "p50": 847, "p75": 1240, "p95": 1680},
    ...
  ],
  "daily_energy_mwh": [18.2, 22.1, 15.4, 19.8, 24.3, 20.1, 16.7],
  "capacity_factor": 0.42,
  "generated_at": "2026-03-11T12:00Z"
}
```

**Streamlit (`app.py`):**
- [ ] Location input: city search + lat/lon fields + interactive map (click-to-select)
- [ ] Turbine dropdown (grouped by manufacturer), shows specs + power curve plot
- [ ] Current predicted power (large number + capacity factor %)
- [ ] 7-day forecast chart with 50% and 90% confidence bands (Plotly)
- [ ] Daily energy bar chart (MWh)
- [ ] Current weather panel (wind speed, direction, temperature, pressure)
- [ ] Page load < 5 seconds, new prediction < 30 seconds

**Color palette:** Primary `#1E88E5` (blue), secondary `#43A047` (green), confidence bands blue with decreasing opacity, warning `#FF7043`, background `#FAFAFA`

### Phase 3: Validation (US-008)

#### US-008: Cross-Geography Validation
**Description:** Validate on Kelmarsh and Penmanshiel to confirm model works on unseen sites.

**Acceptance Criteria:**
- [ ] Kelmarsh and Penmanshiel data downloaded and preprocessed to WindFM format
- [ ] Run zero-shot WindFM on both → baseline metrics (RMSE, MAE, R²)
- [ ] Run fine-tuned WindFM on both → compare metrics
- [ ] Results table:

| Model | SDWPF Test | Kelmarsh | Penmanshiel |
|---|---|---|---|
| WindFM zero-shot | | | |
| WindFM fine-tuned | | | |
| Improvement (%) | | | |

- [ ] Prediction vs actual plots for representative turbines
- [ ] Probabilistic calibration check (90% CI contains ~90% of actuals?)
- [ ] Results saved to `outputs/evaluation/`

---

## User Story Dependency Graph

```
US-001 (WindFM Setup) ───┐
                          ├──► US-003 (Fine-tune) ──────────────────────────┐
US-002 (SDWPF Prep) ─────┘                                                 │
                                                                            ├──► US-006 (Pipeline) ──► US-007 (API + Dashboard)
US-004 (Weather API) ───────────────────────────────────────────────────────┤
                                                                            │
US-005 (Turbine DB) ────────────────────────────────────────────────────────┘
                                                                            │
                                                                            └──► US-008 (Validation)
```

**Critical path:** US-001 → US-002 → US-003 → US-006 → US-007

**Parallel work** (start immediately):
- US-004 (Weather API) — no dependencies
- US-005 (Turbine DB) — no dependencies

---

## Technical Considerations

### Dependencies
```
# Core ML
Python >= 3.11
torch >= 2.0              # MPS support
huggingface_hub >= 0.33
einops == 0.8.1
safetensors >= 0.6.2

# Data
pandas >= 2.2
numpy >= 1.26
pyarrow >= 15.0           # Parquet support

# Turbine data
windpowerlib >= 0.2.2     # Power curves from OEDB

# API & Web
fastapi >= 0.110
uvicorn >= 0.27
streamlit >= 1.30
plotly >= 5.18
requests >= 2.31
cachetools >= 5.3         # API response caching

# Evaluation
scikit-learn >= 1.4
matplotlib >= 3.8
```

### Project Structure
```
windsight/
├── api.py                          # FastAPI service
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── sdwpf/
│   │   ├── raw/                    # Downloaded SDWPF data
│   │   └── processed/             # Parquet in WindFM format
│   └── turbine_specs.json         # Turbine database with power curves
├── src/
│   ├── api/
│   │   └── weather.py             # Open-Meteo client
│   ├── data/
│   │   ├── sdwpf_loader.py        # SDWPF preprocessing
│   │   └── turbine_db.py          # Turbine spec utilities
│   ├── models/
│   │   └── windfm_wrapper.py      # WindFM loading, inference, fine-tuning
│   ├── pipeline/
│   │   └── predictor.py           # End-to-end prediction
│   └── evaluation/
│       └── metrics.py             # RMSE, MAE, R², CRPS
├── notebooks/
│   ├── 01_eda.ipynb               # SDWPF Exploratory Data Analysis
│   ├── 02_finetune.ipynb          # Fine-tuning experiments
│   └── 03_validation.ipynb        # Cross-geography validation
├── outputs/
│   ├── windfm-finetuned/          # Model checkpoints
│   └── evaluation/                # Results & figures
├── tests/
│   ├── test_weather_api.py
│   ├── test_turbine_db.py
│   └── test_pipeline.py
├── scripts/
│   ├── download_sdwpf.py
│   ├── preprocess_sdwpf.py
│   ├── finetune_windfm.py
│   └── validate.py
└── docs/
    └── mps_compatibility.md
```

### Key Risks

| Risk | Severity | Mitigation |
|---|---|---|
| MPS incompatibility | High | US-001 is hard gate. Worst case: CPU inference (~2-5x slower, still viable on M3 Ultra) |
| SDWPF missing pressure/density | Medium | Full version includes ERA5 surface pressure. Derive density from pressure + temperature |
| Fine-tuning instability | Medium | Freeze tokenizer. Low LR (1e-5). Early stopping. |
| Sentetik güç geçmişi kalitesi | Medium | Validate by comparing synthetic vs actual power for SDWPF (where we have both) |
| Open-Meteo rate limits | Low | 10K calls/day free. Cache 15-min TTL. Typical usage < 100 calls/hour |

---

## Non-Goals (Out of Scope for MVP)

- ~~Geographic correction layer~~ — Dropped. Insufficient data for meaningful training.
- No real-time SCADA integration
- No multi-turbine farm optimization or wake modeling
- No electricity price/trading features
- No mobile layout
- No authentication (single-user local)
- No cloud deployment (runs locally)
- No custom turbine input (preset list only)

---

## B2B Future Roadmap (Post-MVP)

| Phase | Feature | Value |
|---|---|---|
| **MVP** | Zero-shot + fine-tuned forecasting API | "Any location, 7-day forecast" |
| **v2** | Client SCADA fine-tuning | Client shares historical data → custom model → higher accuracy |
| **v3** | Multi-turbine farm forecasting | Whole-farm output prediction |
| **v4** | Market integration | Electricity price overlay, trading signals |
| **v5** | Geographic correction (with enough client data) | Learn terrain/elevation effects from accumulated client sites |

---

## Success Metrics

### Model
- Fine-tuned WindFM achieves **≥ 5% RMSE reduction** over zero-shot on SDWPF test set
- Fine-tuned model performs **equal or better** than zero-shot on Kelmarsh and Penmanshiel (no regression)

### Product
- Full prediction: **< 30 seconds** any location
- Dashboard load: **< 5 seconds**
- API uptime: works locally on M3 Ultra without issues

### Validation
- Predictions are physically plausible (0 ≤ power ≤ rated, respects cut-in/cut-out)
- Probabilistic forecasts are calibrated (90% CI ≈ 90% coverage)
- Works for at least 3 different turbine models on at least 3 different continents
