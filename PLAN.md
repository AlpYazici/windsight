# WindSight — Research & Development Plan

## Research Thesis

The WindFM paper demonstrates strong zero-shot wind power forecasting but **never explores fine-tuning**. We hypothesize that domain-adaptive fine-tuning on real SCADA data (SDWPF, 134 turbines) improves both deterministic and probabilistic accuracy — and that this improvement **transfers cross-geography** to unseen European wind farms (Kelmarsh, Penmanshiel, Hill of Towie) with different turbine manufacturers and terrain types.

Secondary contribution: a **synthetic power history** method that enables forecasting at any location without historical SCADA data, using only weather API data + turbine power curves.

## Research Contributions (vs. Original Paper)

| Gap in WindFM Paper | Our Contribution |
|---|---|
| Zero-shot only — no fine-tuning explored | Fine-tune on real SCADA (SDWPF), measure improvement |
| Pre-trained on US NREL data only | Cross-geography validation on 3 European sites (UK, Scotland) |
| Pre-trained on simulated NREL data | Fine-tune on real-world SCADA with sensor noise, curtailment, outages |
| No method for locations without SCADA history | Synthetic power history via turbine power curves + weather API |
| Limited turbine diversity in evaluation | Test across 3 manufacturers: Sinovel, Senvion, Siemens |
| No terrain diversity analysis | Flat farmland (Kelmarsh) vs hilly (Penmanshiel, Hill of Towie) |

## Evaluation Metrics

| Category | Metric | Description |
|---|---|---|
| Deterministic | MAE | Mean Absolute Error |
| Deterministic | RMSE | Root Mean Squared Error |
| Deterministic | R² | Coefficient of determination |
| Probabilistic | CRPS | Continuous Ranked Probability Score |
| Probabilistic | AQL | Average Quantile Loss |
| Calibration | Coverage | % of actuals within 90% CI (target: ~90%) |
| Calibration | Sharpness | Average width of prediction intervals |

---

## Phase 0: Environment & Model Verification

**Goal:** Confirm WindFM runs on Apple Silicon MPS and produces valid predictions.

### 0.1 — Environment Setup
- [ ] Create Python 3.11+ virtual environment
- [ ] Install dependencies: torch (MPS), einops, safetensors, huggingface_hub, pandas, numpy, pyarrow
- [ ] Verify `torch.backends.mps.is_available()` returns True

### 0.2 — WindFM Inference on MPS
- [ ] Modify `WindFMPredictor` device from `cuda:0` to `mps`
- [ ] Run bundled example (`WindFM/examples/data/121522.csv`) on MPS
- [ ] Validate output: non-NaN, physically reasonable power values
- [ ] Document any MPS-incompatible ops (likely `scatter_reduce` in BSQ entropy — training only, not inference)
- [ ] Benchmark: inference time for 240→80 step prediction on MPS vs CPU

### 0.3 — Load Local Weights
- [ ] Verify loading from `models/windfm/model.safetensors` and `models/windfm-tokenizer/model.safetensors` (instead of HuggingFace download)
- [ ] Confirm identical output to HuggingFace-loaded model

**Hard gate:** Phase 1 cannot start until inference works.

**Deliverables:** `docs/mps_compatibility.md`, working inference script

---

## Phase 1: Data Preparation

**Goal:** All datasets cleaned, mapped to WindFM 6-feature format, split, and ready for training/evaluation.

### 1.1 — SDWPF Processing (Fine-Tuning Data)
- [ ] Load `data/sdwpf/raw/sdwpf_full.parquet` (11.4M rows, 134 turbines)
- [ ] Quality cleaning:
  - Negative Patv → 0 (curtailment/consumption)
  - Remove rows where Wspd > 0 but Patv = 0 for extended periods (outage detection)
  - Flag outlier turbines with abnormal power curves
  - Handle NaN/missing values (forward fill for short gaps, drop for long gaps)
- [ ] Map to WindFM 6-feature format:
  - `wind_speed` ← Wspd (m/s, direct)
  - `wind_direction` ← Wdir (degrees, direct)
  - `power` ← Patv / 1000 (kW → MW)
  - `density` ← Sp / (287.05 × (Etmp + 273.15)) (derived from ERA5 surface pressure)
  - `temperature` ← Etmp + 273.15 (°C → K)
  - `pressure` ← Sp (Pa, direct from ERA5 column)
- [ ] Resample to hourly (10-min → 1-hour mean) to match forecast use case
- [ ] Temporal split by date (no leakage):
  - Train: Jan 2020 – Aug 2021 (70%)
  - Validation: Sep 2021 – Oct 2021 (15%)
  - Test: Nov 2021 – Dec 2021 (15%)
- [ ] Save to `data/sdwpf/processed/{train,val,test}.parquet`

### 1.2 — Kelmarsh Processing (Validation — Flat Terrain)
- [ ] Extract zip files (2017–2021, ~1.4 GB total)
- [ ] Parse SCADA signals using `kelmarsh_signal_mapping.csv`
- [ ] Map to WindFM 6-feature format:
  - `wind_speed` ← anemometer wind speed (m/s)
  - `wind_direction` ← wind vane (degrees)
  - `power` ← active power / 1000 (kW → MW)
  - `density` ← derive from temperature + pressure if available, else use standard atmosphere at elevation
  - `temperature` ← ambient temp + 273.15 (°C → K)
  - `pressure` ← barometric pressure or derive from elevation
- [ ] Resample to hourly
- [ ] Use 2020–2021 as evaluation period (overlap with SDWPF training era)
- [ ] Save to `data/kelmarsh/processed/eval.parquet`

### 1.3 — Penmanshiel Processing (Validation — Hilly Terrain)
- [ ] Extract `penmanshiel_2020_wt01-10.zip` (~694 MB)
- [ ] Parse using `penmanshiel_signal_mapping.xlsx`
- [ ] Map to WindFM 6-feature format (same approach as Kelmarsh)
- [ ] Resample to hourly, save to `data/penmanshiel/processed/eval.parquet`

### 1.4 — Hill of Towie Processing (Validation — Different Manufacturer)
- [ ] Extract `hot_2020.zip` (~1.3 GB)
- [ ] Parse using `hot_fields.csv` and `hot_metadata.csv`
- [ ] Map to WindFM 6-feature format
- [ ] Resample to hourly, save to `data/hill_of_towie/processed/eval.parquet`

### 1.5 — Exploratory Data Analysis
- [ ] EDA notebook: `notebooks/01_eda.ipynb`
  - Per-turbine power curves (Patv vs Wspd scatter)
  - Wind roses per dataset
  - Feature distributions comparison across all 4 datasets
  - Missing data heatmaps
  - Temporal patterns (diurnal, seasonal)
  - Dataset summary statistics table (for paper)

**Deliverables:** Processed parquet files, EDA notebook, data summary table for paper

---

## Phase 2: Fine-Tuning

**Goal:** Fine-tune WindFM's autoregressive Transformer on SDWPF, demonstrating improvement over zero-shot baseline.

### 2.1 — Fine-Tuning Infrastructure
- [ ] Create `src/training/finetune.py`
- [ ] Implement training loop:
  1. Freeze tokenizer (encoder + decoder + BSQ) — `requires_grad = False`
  2. Encode training sequences with frozen tokenizer → (s1_ids, s2_ids) pairs
  3. Shift tokens for teacher forcing: input = tokens[:-1], target = tokens[1:]
  4. Forward pass: `model(s1_ids, s2_ids, stamps, use_teacher_forcing=True, s1_targets=s1_targets)`
  5. Loss: `model.head.compute_loss(s1_logits, s2_logits, s1_targets, s2_targets)`
- [ ] Sliding window data loader:
  - Window size: 512 steps (max context)
  - Stride: configurable (e.g., 256 for 50% overlap)
  - Per-window z-score normalization (matching inference behavior)
  - Random turbine sampling per batch

### 2.2 — Training Configuration
- [ ] Hyperparameters:
  - Learning rate: 1e-5 (low to preserve pre-trained weights)
  - Scheduler: cosine annealing with 5% linear warmup
  - Batch size: 64
  - Max epochs: 50
  - Early stopping: patience=5 on validation loss
  - Gradient clipping: max_norm=1.0
- [ ] MPS training verification (test `scatter_reduce` workaround if needed)
- [ ] Checkpoint best model by validation loss → `outputs/windfm-finetuned/`

### 2.3 — Training Execution & Monitoring
- [ ] Log per-epoch: train loss, val loss, ce_s1, ce_s2
- [ ] Training notebook: `notebooks/02_finetune.ipynb` (experiment tracking)
- [ ] Target: training completes within 24 hours on M3 Ultra
- [ ] Save training curves for paper

### 2.4 — Ablation Studies (For Paper)
- [ ] Effect of learning rate: {5e-6, 1e-5, 5e-5}
- [ ] Effect of training data size: {25%, 50%, 100% of turbines}
- [ ] Effect of context length: {128, 256, 512 steps}
- [ ] Frozen tokenizer vs full fine-tune (expect full fine-tune to degrade)

**Deliverables:** Fine-tuned checkpoint, training curves, ablation results

---

## Phase 3: Evaluation & Validation

**Goal:** Comprehensive comparison of zero-shot vs fine-tuned WindFM across all datasets. This is the core of the paper.

### 3.1 — SDWPF Test Set Evaluation
- [ ] Run zero-shot WindFM on SDWPF test set
- [ ] Run fine-tuned WindFM on SDWPF test set
- [ ] Prediction horizons: 24h, 48h, 72h, 168h (7 days)
- [ ] Deterministic metrics: MAE, RMSE, R² (mean of 20 samples, T=0.9)
- [ ] Probabilistic metrics: CRPS, AQL (100 samples, T=1.0)
- [ ] Per-turbine breakdown (identify which turbines benefit most from fine-tuning)

### 3.2 — Cross-Geography Evaluation
- [ ] For each validation dataset (Kelmarsh, Penmanshiel, Hill of Towie):
  - Zero-shot WindFM → metrics
  - Fine-tuned WindFM → metrics
  - Compare: does fine-tuning on SDWPF (China) help or hurt on European sites?
- [ ] Same prediction horizons: 24h, 48h, 72h, 168h
- [ ] Same metric suite: MAE, RMSE, R², CRPS, AQL

### 3.3 — Probabilistic Calibration Analysis
- [ ] Reliability diagrams: predicted quantile vs observed frequency
- [ ] Coverage analysis: does 90% CI actually contain ~90% of actuals?
- [ ] Sharpness: average PI width (narrower is better at same coverage)
- [ ] Compare calibration: zero-shot vs fine-tuned

### 3.4 — Synthetic Power History Evaluation
- [ ] For SDWPF turbines (where we have real power data):
  - Generate synthetic power from power curve + wind speed
  - Run WindFM with synthetic history vs real history
  - Measure accuracy degradation → quantifies the "synthetic gap"
- [ ] This validates the approach used for arbitrary-location forecasting

### 3.5 — Results Compilation
- [ ] Main results table (for paper):

| Model | Dataset | MAE | RMSE | R² | CRPS | AQL |
|---|---|---|---|---|---|---|
| WindFM zero-shot | SDWPF test | | | | | |
| WindFM fine-tuned | SDWPF test | | | | | |
| WindFM zero-shot | Kelmarsh | | | | | |
| WindFM fine-tuned | Kelmarsh | | | | | |
| WindFM zero-shot | Penmanshiel | | | | | |
| WindFM fine-tuned | Penmanshiel | | | | | |
| WindFM zero-shot | Hill of Towie | | | | | |
| WindFM fine-tuned | Hill of Towie | | | | | |

- [ ] Prediction vs actual time-series plots (representative turbines per dataset)
- [ ] Confidence band visualizations
- [ ] Save all to `outputs/evaluation/`

**Deliverables:** Results tables, figures, evaluation notebook `notebooks/03_validation.ipynb`

---

## Phase 4: Open-Source Pipeline

**Goal:** Build the reusable prediction pipeline that works for any location, package for open-source release.

### 4.1 — Weather API Client
- [ ] `src/api/weather.py` — Open-Meteo integration
  - Historical weather (last 10 days): wind speed 10m + 80m, direction, temp, pressure
  - Forecast (next 7 days): same variables
  - Elevation lookup
  - Hub-height wind extrapolation: `v_hub = v_ref × (h_hub/h_ref)^α`
  - Air density calculation: `ρ = P / (R_d × T)`
  - Output: DataFrame in WindFM 6-feature format
- [ ] Response caching (15-min TTL)
- [ ] Unit tests with mocked API responses

### 4.2 — Turbine Database
- [ ] `data/turbine_specs.json` — Power curves for 12+ turbine models
  - Extract from windpowerlib/OEDB where available
  - Derive Sinovel SL1500/82 empirical curve from SDWPF data
  - Include: rated power, rotor diameter, hub height, cut-in/rated/cut-out speeds, power curve array
- [ ] `src/data/turbine_db.py` — lookup API: `get_turbine()`, `list_turbines()`, `estimate_power()`

### 4.3 — End-to-End Prediction Pipeline
- [ ] `src/pipeline/predictor.py`
  - Input: lat, lon, turbine model name
  - Fetch weather → extrapolate to hub height → synthetic power → WindFM inference → post-processing
  - Output: hourly forecast with percentiles (P5, P25, P50, P75, P95), daily energy, capacity factor
  - Physical constraints: clip to [0, rated_power], apply cut-in/cut-out
  - Target latency: < 30 seconds

### 4.4 — Web Interface (Demo for Open-Source)
- [ ] FastAPI backend (`api.py`): `/forecast`, `/turbines`, `/health`
- [ ] Streamlit dashboard (`app.py`): map, turbine selector, forecast chart, confidence bands
- [ ] Minimal but functional — serves as a demo for the open-source release

### 4.5 — Repository Packaging
- [ ] Clean project structure, README with usage examples
- [ ] `requirements.txt` with pinned versions
- [ ] Example notebooks showing: inference, fine-tuning, evaluation
- [ ] License selection (MIT or Apache 2.0)
- [ ] Pre-trained + fine-tuned model weights on HuggingFace

**Deliverables:** Complete open-source repository, demo app, HuggingFace model release

---

## Phase 5: Paper Writing

**Goal:** Write and submit the research paper.

### 5.1 — Paper Structure (Draft Outline)
1. **Abstract** — Fine-tuning WindFM + cross-geography generalization + synthetic power history
2. **Introduction** — Wind power forecasting landscape, foundation models, gap (no fine-tuning study)
3. **Related Work** — Time-series foundation models, wind power forecasting, transfer learning in energy
4. **Method**
   - WindFM architecture overview (brief, cite original)
   - Fine-tuning strategy (frozen tokenizer, AR Transformer only)
   - Synthetic power history for arbitrary-location forecasting
5. **Experimental Setup**
   - Datasets: SDWPF (training), Kelmarsh, Penmanshiel, Hill of Towie (validation)
   - Metrics: MAE, RMSE, CRPS, AQL, calibration
   - Baselines: zero-shot WindFM, (optional: persistence, climatology)
6. **Results**
   - Fine-tuning improvement on SDWPF
   - Cross-geography transfer
   - Probabilistic calibration
   - Synthetic vs real power history gap
   - Ablation studies
7. **Discussion** — When fine-tuning helps, terrain/manufacturer effects, limitations
8. **Conclusion** — Summary + open-source release announcement

### 5.2 — Figures & Tables
- [ ] Architecture diagram (fine-tuning pipeline)
- [ ] Main results table (all datasets × all metrics)
- [ ] Prediction plots with confidence bands
- [ ] Reliability diagrams
- [ ] Training curves
- [ ] Wind farm location map
- [ ] Power curve comparison across datasets

### 5.3 — Target Venues
- Applied Energy, Renewable Energy, or Energy AI (journal)
- NeurIPS / ICML Climate Change AI workshop (conference)
- arXiv preprint first

**Deliverables:** Paper draft, figures, submission-ready manuscript

---

## Execution Order & Dependencies

```
Phase 0 (Environment)
    │
    ▼
Phase 1 (Data Prep) ─────────────────────────────┐
    │                                              │
    ▼                                              │ (parallel)
Phase 2 (Fine-Tuning)                             │
    │                                    Phase 4.1 (Weather API)
    ▼                                    Phase 4.2 (Turbine DB)
Phase 3 (Evaluation)                              │
    │                                              │
    ├──────────────────────────────────────────────┘
    ▼
Phase 4.3–4.5 (Pipeline + Packaging)
    │
    ▼
Phase 5 (Paper)
```

**Critical path:** Phase 0 → Phase 1.1 → Phase 2 → Phase 3 → Phase 5

**Parallel work (start during Phase 1):**
- Phase 4.1 (Weather API) — no dependencies
- Phase 4.2 (Turbine DB) — no dependencies
- Phase 1.2–1.4 (validation data prep) — independent of SDWPF

---

## Project Structure

```
windsight/
├── PLAN.md                              # This file
├── CLAUDE.md                            # Project context
├── prd-windsight.md                     # Original PRD
├── WindFM/                              # Cloned WindFM repo (upstream code)
├── models/
│   ├── windfm/                          # Pre-trained weights
│   └── windfm-tokenizer/               # Tokenizer weights
├── data/
│   ├── sdwpf/
│   │   ├── raw/                         # Original SDWPF parquet + metadata
│   │   └── processed/                   # WindFM-format train/val/test splits
│   ├── kelmarsh/
│   │   ├── raw/                         # Zenodo downloads
│   │   └── processed/                   # WindFM-format eval set
│   ├── penmanshiel/
│   │   ├── raw/
│   │   └── processed/
│   ├── hill_of_towie/
│   │   ├── raw/
│   │   └── processed/
│   └── turbine_specs.json               # Power curves database
├── src/
│   ├── api/
│   │   └── weather.py                   # Open-Meteo client
│   ├── data/
│   │   ├── sdwpf_loader.py             # SDWPF preprocessing
│   │   ├── kelmarsh_loader.py          # Kelmarsh preprocessing
│   │   ├── penmanshiel_loader.py       # Penmanshiel preprocessing
│   │   ├── hot_loader.py               # Hill of Towie preprocessing
│   │   └── turbine_db.py               # Turbine specifications
│   ├── models/
│   │   └── windfm_wrapper.py           # WindFM loading & inference wrapper
│   ├── training/
│   │   └── finetune.py                 # Fine-tuning loop
│   ├── pipeline/
│   │   └── predictor.py                # End-to-end prediction
│   └── evaluation/
│       └── metrics.py                   # RMSE, MAE, R², CRPS, AQL, calibration
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory data analysis
│   ├── 02_finetune.ipynb               # Fine-tuning experiments
│   └── 03_validation.ipynb             # Cross-geography validation
├── outputs/
│   ├── windfm-finetuned/               # Model checkpoints
│   └── evaluation/                     # Results, figures, tables
├── api.py                               # FastAPI service
├── app.py                               # Streamlit dashboard
├── literature-review/                   # Papers
├── tests/
└── requirements.txt
```
