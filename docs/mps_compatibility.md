# MPS Compatibility Report — WindFM on Apple Silicon

## Environment
- Hardware: Apple M3 Ultra, 256 GB unified memory
- Python: 3.13.5
- PyTorch: 2.10.0
- MPS available: True

## Inference Results
- **Status: WORKING** on MPS backend
- Device: `mps` (no code changes needed beyond replacing `cuda:0`)
- Output: Valid (non-NaN), physically reasonable power predictions

## Benchmark (240-step lookback → 80-step prediction, 20 samples)
| Device | Time (s) | Speedup |
|--------|----------|---------|
| MPS    | 10.5     | 1.0×    |
| CPU    | 27.1     | 0.39×   |

MPS is **2.6× faster** than CPU for inference.

## Known Issues
- `scatter_reduce` in BSQ entropy loss (`DifferentiableEntropyFunction`) may have MPS issues — only needed during **training**, not inference.
- If training on MPS fails, fallback: compute BSQ loss on CPU, rest on MPS.

## Model Sizes
- WindFM AR Transformer: 4,100,096 parameters (16 MB)
- WindFM Tokenizer: 3,958,042 parameters (15 MB)
- Total: ~8.1M parameters
