# WindSight

## Proje Hedefi
WindFM foundation modeli + Open-Meteo hava durumu API'si kullanarak herhangi bir konum icin ruzgar turbini guc tahmini yapan bir SaaS urun.

Kullanici haritadan konum secer, turbin modeli secer → 7 gunluk saatlik guc tahmini + guven araliklari alir.

## Mimari
```
Konum + Turbin secimi
    → Open-Meteo API (hava durumu gecmisi + 7 gun tahmin)
    → Guc egrisi ile sentetik guc gecmisi uret
    → WindFM inference (100 ornek → olasiliksal tahmin)
    → Son isleme (cut-in/cut-out, kapasite siniri)
    → FastAPI + Streamlit dashboard
```

## Temel Kararlar
- Cografi duzeltme katmani MVP'den cikarildi (yeterli veri yok, musteriler biriktikce v5'te eklenir)
- WindFM zero-shot calisir ama SDWPF ile fine-tune yapilacak
- XGBoost ensemble olarak eklenecek (SDWPF + diger verilerle egitilir)
- Sentetik guc gecmisi: yeni konumlarda gecmis power verisi yok, turbin guc egrisinden uretilir
- Tokenizer freeze edilir, sadece AR Transformer fine-tune edilir

## WindFM Teknik
- Model: `NeoQuasar/WindFM` (16 MB, 8.1M parametre)
- Tokenizer: `NeoQuasar/WindFM-Tokenizer` (15 MB)
- Repo: https://github.com/shiyu-coder/WindFM
- Girdi: 6 ozellik [wind_speed, wind_direction, power, density, temperature, pressure] + UTC timestamp
- Birimler: m/s, derece, MW, kg/m3, Kelvin, Pascal
- Normalizasyon: per-sample z-score (otomatik)
- Power = feature index 2
- Max context: 512 adim

## Veri Setleri

### Egitim
| Veri Seti | Turbin | Lokasyon | Durum |
|---|---|---|---|
| SDWPF | 134 x Sinovel 1.5MW | Cin | data/sdwpf/raw/ ✅ indirildi (278 MB parquet) |
| Kelmarsh | 6 x Senvion MM92 | Ingiltere | data/kelmarsh/raw/ ✅ indirildi (2016-2021) |
| Penmanshiel | 14 x Senvion MM82 | Iskocya | data/penmanshiel/raw/ ✅ 2020 yili indirildi (662 MB) |

### Test (held-out)
| Veri Seti | Turbin | Lokasyon | Durum |
|---|---|---|---|
| Hill of Towie | 21 x Siemens SWT-2.3 | Iskocya | data/hill_of_towie/raw/ ✅ 2020 yili indirildi (1.3 GB) |

### La Haute Borne (4 turbin, Fransa) — ENGIE portali kapali, erisilemiyor.

## Model Dosyalari
- WindFM repo: `WindFM/` (kod)
- Model agirliklari: `models/windfm/model.safetensors`
- Tokenizer agirliklari: `models/windfm-tokenizer/model.safetensors`

## User Story Durumu
| US | Aciklama | Durum |
|---|---|---|
| US-001 | WindFM'i MPS'te calistir | ✅ Tamamlandi (MPS 2.6x hizli, docs/mps_compatibility.md) |
| US-002 | SDWPF'yi WindFM formatina cevir | ✅ Tamamlandi (train/val/test parquet) |
| US-002b | Kelmarsh/Penmanshiel/HoT islemek | ✅ Tamamlandi (eval.parquet) |
| US-003 | WindFM fine-tune (SDWPF) | ✅ 28 epoch, val_loss 4.21→3.15 (25.2% reduction) |
| US-004 | Open-Meteo API client | ✅ src/api/weather.py (29 tests) |
| US-005 | Turbin veritabani | ✅ data/turbine_specs.json (13 model, 22 tests) |
| US-006 | Tahmin pipeline'i | ✅ src/pipeline/predictor.py |
| US-007 | FastAPI + Streamlit | ✅ api.py + app.py |
| US-008 | Cross-geography dogrulama | 🔄 Evaluation running |

## Islenmis Veri Ozeti
| Dataset | Rows | Turbines | Period |
|---|---|---|---|
| SDWPF train | 1,779,125 | 134 | Jan 2020 – Aug 2021 |
| SDWPF val | 195,854 | 134 | Sep – Oct 2021 |
| SDWPF test | 189,164 | 134 | Nov – Dec 2021 |
| Kelmarsh eval | 265,134 | 6 | May 2016 – Jun 2021 |
| Penmanshiel eval | 74,533 | 9 | Jan – Dec 2020 |
| Hill of Towie eval | 183,650 | 21 | Jan 2020 – Jan 2021 |

## Veri Isleme Notlari
- SDWPF: T2m (ERA5) kullanildi, Etmp degil (Etmp=28°C ortalam guvenilmez)
- Kelmarsh/Penmanshiel/HoT: Atmosferik basinc yok → barometrik formulle turetildi
- Hill of Towie: Ruzgar yonu yok → nacelle yaw pozisyonu proxy olarak kullanildi
- Tum veriler saatlik ortalamaya donusturuldu

## Fine-Tuning Sonuclari
- Epochs: 28 (early stopping, best at epoch 23)
- Val loss: 4.2120 → 3.1506 (25.2% reduction)
- Frozen: Tokenizer (3.96M params), Trained: AR Transformer (4.1M params)
- Checkpoint: outputs/windfm-finetuned/best_model.pt (16 MB)

## PRD
Detayli PRD: `prd-windsight.md`
Detayli Plan: `PLAN.md`
