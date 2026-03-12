#!/usr/bin/env python3
"""
Evaluation runner for the WindSight project.

Evaluates a WindFM model (zero-shot or fine-tuned) across multiple datasets
and prediction horizons, producing deterministic + probabilistic metrics.

Usage examples:
    # Zero-shot evaluation on all datasets
    python3 scripts/evaluate.py --model_path models/windfm

    # Fine-tuned model, specific datasets, single horizon
    python3 scripts/evaluate.py \
        --model_path outputs/windfm-finetuned/best_model.pt \
        --datasets sdwpf kelmarsh \
        --pred_horizons 24
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve project root and make WindFM importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDFM_DIR = PROJECT_ROOT / "WindFM"
if str(WINDFM_DIR) not in sys.path:
    sys.path.insert(0, str(WINDFM_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model.windfm import WindFM, WindFMTokenizer, WindFMPredictor, calc_time_stamps  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "sdwpf": {
        "path": PROJECT_ROOT / "data" / "sdwpf" / "processed" / "test.parquet",
        "name": "SDWPF",
    },
    "kelmarsh": {
        "path": PROJECT_ROOT / "data" / "kelmarsh" / "processed" / "eval.parquet",
        "name": "Kelmarsh",
    },
    "penmanshiel": {
        "path": PROJECT_ROOT / "data" / "penmanshiel" / "processed" / "eval.parquet",
        "name": "Penmanshiel",
    },
    "hill_of_towie": {
        "path": PROJECT_ROOT / "data" / "hill_of_towie" / "processed" / "eval.parquet",
        "name": "Hill of Towie",
    },
}

# Aliases for convenience (e.g. "sdwpf_test" -> "sdwpf")
DATASET_ALIASES = {
    "sdwpf_test": "sdwpf",
}

WINDOWS_PER_TURBINE = 10  # evenly spaced evaluation windows per turbine


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    device: str,
) -> tuple:
    """Load WindFM model and tokenizer, supporting both HF-format dirs and
    single-file .pt/.pth checkpoints (for fine-tuned models)."""

    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)

    # --- Tokenizer: always loaded from a HF-format directory ---
    print(f"  Loading tokenizer from {tokenizer_path}")
    tokenizer = WindFMTokenizer.from_pretrained(str(tokenizer_path))

    # --- Model ---
    if model_path.is_dir():
        # HF-format directory (zero-shot weights)
        print(f"  Loading model from directory {model_path}")
        model = WindFM.from_pretrained(str(model_path))
    elif model_path.is_file() and model_path.suffix in (".pt", ".pth"):
        # Single-file checkpoint (fine-tuned)
        print(f"  Loading fine-tuned checkpoint from {model_path}")
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

        # The checkpoint may contain the full state dict directly, or wrap it
        # inside a dict with a "model_state_dict" (or similar) key.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # We need the architecture config.  First try loading the base model
        # from the default HF directory so we get the correct architecture,
        # then swap in the fine-tuned weights.
        base_model_dir = PROJECT_ROOT / "models" / "windfm"
        print(f"  Initializing architecture from {base_model_dir}")
        model = WindFM.from_pretrained(str(base_model_dir))
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"Model path {model_path} is neither a directory nor a .pt/.pth file."
        )

    tokenizer.to(device).eval()
    model.to(device).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Contiguous segment detection
# ---------------------------------------------------------------------------

def find_contiguous_segments(
    turbine_df: pd.DataFrame,
    max_gap_hours: int = 2,
) -> list[pd.DataFrame]:
    """Split a single-turbine DataFrame into contiguous segments where
    consecutive time steps are separated by at most *max_gap_hours* hours."""

    turbine_df = turbine_df.sort_values("time").reset_index(drop=True)
    time_diff = turbine_df["time"].diff()
    gap_mask = time_diff > pd.Timedelta(hours=max_gap_hours)

    # Identify segment boundaries
    seg_ids = gap_mask.cumsum()
    segments = []
    for _, seg_df in turbine_df.groupby(seg_ids):
        segments.append(seg_df.reset_index(drop=True))
    return segments


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def select_eval_windows(
    segments: list[pd.DataFrame],
    lookback: int,
    pred_horizon: int,
    n_windows: int = WINDOWS_PER_TURBINE,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """From the list of contiguous segments, select up to *n_windows*
    evenly spaced (context, ground_truth) window pairs.

    Each window requires at least *lookback + pred_horizon* contiguous rows.

    Returns a list of (context_df, gt_df) tuples.
    """

    min_len = lookback + pred_horizon
    # Collect all valid start positions across segments
    candidates = []  # (segment_index, start_row)
    for seg_idx, seg in enumerate(segments):
        if len(seg) < min_len:
            continue
        n_possible = len(seg) - min_len + 1
        for start in range(n_possible):
            candidates.append((seg_idx, start))

    if not candidates:
        return []

    # Evenly space the selections
    if len(candidates) <= n_windows:
        chosen = candidates
    else:
        indices = np.linspace(0, len(candidates) - 1, n_windows, dtype=int)
        chosen = [candidates[i] for i in indices]

    windows = []
    for seg_idx, start in chosen:
        seg = segments[seg_idx]
        ctx = seg.iloc[start : start + lookback]
        gt = seg.iloc[start + lookback : start + lookback + pred_horizon]
        windows.append((ctx, gt))

    return windows


# ---------------------------------------------------------------------------
# Single-window prediction
# ---------------------------------------------------------------------------

def predict_window(
    predictor: WindFMPredictor,
    ctx_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pred_horizon: int,
    n_samples: int,
    temperature: float,
    top_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run WindFM on a single (context, ground_truth) window.

    Returns
    -------
    y_true : ndarray of shape (pred_horizon,)
    y_samples : ndarray of shape (pred_horizon, n_samples)
    """

    x_timestamp = pd.to_datetime(ctx_df["time"]).reset_index(drop=True)
    y_timestamp = pd.to_datetime(gt_df["time"]).reset_index(drop=True)

    # Ensure tz-naive (WindFM's calc_time_stamps uses .dt.minute etc.)
    if x_timestamp.dt.tz is not None:
        x_timestamp = x_timestamp.dt.tz_localize(None)
    if y_timestamp.dt.tz is not None:
        y_timestamp = y_timestamp.dt.tz_localize(None)

    feature_cols = predictor.feature_cols
    input_df = ctx_df[feature_cols].reset_index(drop=True)

    pred_df = predictor.predict(
        df=input_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_horizon,
        T=temperature,
        top_k=0,
        top_p=top_p,
        sample_count=n_samples,
        verbose=False,
    )

    y_true = gt_df["power"].values.astype(np.float64)
    y_samples = pred_df.values.astype(np.float64)  # (pred_horizon, n_samples)

    return y_true, y_samples


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset_key: str,
    predictor: WindFMPredictor,
    pred_horizons: list[int],
    n_samples: int,
    n_turbines: int,
    lookback: int,
    temperature: float,
    top_p: float,
) -> dict:
    """Evaluate a single dataset across all prediction horizons.

    Returns a dict: { horizon_hours: { metric_name: value, ... }, ... }
    """

    info = DATASET_REGISTRY[dataset_key]
    parquet_path = info["path"]
    dataset_name = info["name"]

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  ({parquet_path.name})")
    print(f"{'='*60}")

    df = pd.read_parquet(parquet_path)
    df["time"] = pd.to_datetime(df["time"])

    turbine_ids = sorted(df["TurbID"].unique())
    if n_turbines > 0:
        # Evenly subsample turbines
        indices = np.linspace(0, len(turbine_ids) - 1, min(n_turbines, len(turbine_ids)), dtype=int)
        turbine_ids = [turbine_ids[i] for i in indices]

    print(f"  Evaluating {len(turbine_ids)} turbine(s)")

    results_by_horizon = {}

    for horizon_hours in pred_horizons:
        pred_horizon = horizon_hours  # data is hourly, so hours == rows
        lookback_rows = lookback

        print(f"\n  --- Horizon: {horizon_hours}h ---")

        all_y_true = []
        all_y_samples = []
        n_windows_total = 0

        for turb_idx, turb_id in enumerate(turbine_ids):
            turb_df = df[df["TurbID"] == turb_id].copy()
            segments = find_contiguous_segments(turb_df, max_gap_hours=2)
            windows = select_eval_windows(segments, lookback_rows, pred_horizon)

            if not windows:
                print(f"    Turbine {turb_id}: no valid windows (skipped)")
                continue

            print(
                f"    Turbine {turb_id} [{turb_idx+1}/{len(turbine_ids)}]: "
                f"{len(windows)} window(s) ...",
                end="",
                flush=True,
            )

            for ctx_df, gt_df in windows:
                try:
                    y_true, y_samples = predict_window(
                        predictor, ctx_df, gt_df,
                        pred_horizon, n_samples, temperature, top_p,
                    )
                    all_y_true.append(y_true)
                    all_y_samples.append(y_samples)
                    n_windows_total += 1
                except Exception as e:
                    print(f" [error: {e}]", end="")
                    continue

            print(" done")

        if n_windows_total == 0:
            print(f"  No valid windows for horizon {horizon_hours}h -- skipping")
            results_by_horizon[str(horizon_hours)] = {"error": "no valid windows"}
            continue

        # Stack across all turbines and windows
        combined_y_true = np.concatenate(all_y_true, axis=0)
        combined_y_samples = np.concatenate(all_y_samples, axis=0)

        metrics = compute_all_metrics(combined_y_true, combined_y_samples)
        metrics["n_windows"] = n_windows_total
        metrics["n_timesteps"] = int(combined_y_true.shape[0])

        results_by_horizon[str(horizon_hours)] = metrics

        print(f"  Results ({horizon_hours}h): MAE={metrics['mae']:.4f}  "
              f"RMSE={metrics['rmse']:.4f}  R²={metrics['r_squared']:.4f}  "
              f"CRPS={metrics['crps']:.4f}  Cov90={metrics['coverage_90']:.2%}")

    return results_by_horizon


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict) -> None:
    """Print a formatted summary table of results across datasets and horizons."""

    print(f"\n{'='*100}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*100}")

    header = f"{'Dataset':<18} {'Horizon':>8} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'CRPS':>10} {'AQL':>10} {'Cov90':>8} {'Cov50':>8} {'Windows':>8}"
    print(header)
    print("-" * 100)

    for dataset_key, horizons in all_results.items():
        dataset_name = DATASET_REGISTRY.get(dataset_key, {}).get("name", dataset_key)
        for horizon, metrics in horizons.items():
            if "error" in metrics:
                print(f"{dataset_name:<18} {horizon:>7}h {'ERROR':>10}")
                continue
            print(
                f"{dataset_name:<18} {horizon:>7}h "
                f"{metrics['mae']:>10.4f} "
                f"{metrics['rmse']:>10.4f} "
                f"{metrics['r_squared']:>8.4f} "
                f"{metrics['crps']:>10.4f} "
                f"{metrics['aql']:>10.4f} "
                f"{metrics['coverage_90']:>7.2%} "
                f"{metrics['coverage_50']:>7.2%} "
                f"{metrics['n_windows']:>8d}"
            )

    print(f"{'='*100}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WindFM models on wind-power forecasting datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/windfm",
        help="Path to model weights (directory for zero-shot, .pt file for fine-tuned).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="models/windfm-tokenizer",
        help="Path to tokenizer weights directory.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_REGISTRY.keys()),
        help="Comma-separated list of datasets to evaluate. "
             "Available: " + ", ".join(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--pred_horizons",
        type=str,
        default="24,48,72,168",
        help="Comma-separated prediction horizons in hours.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of probabilistic forecast samples.",
    )
    parser.add_argument(
        "--n_turbines",
        type=int,
        default=0,
        help="Number of turbines to evaluate per dataset (0 = all).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=240,
        help="Lookback / context window in hours.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Compute device.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for saving result files.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for autoregressive generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling (top-p) threshold.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve relative paths w.r.t. project root
    model_path = (
        Path(args.model_path)
        if Path(args.model_path).is_absolute()
        else PROJECT_ROOT / args.model_path
    )
    tokenizer_path = (
        Path(args.tokenizer_path)
        if Path(args.tokenizer_path).is_absolute()
        else PROJECT_ROOT / args.tokenizer_path
    )
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_horizons = [int(h.strip()) for h in args.pred_horizons.split(",")]

    # Parse comma-separated datasets and resolve aliases
    raw_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    datasets = []
    for d in raw_datasets:
        resolved = DATASET_ALIASES.get(d, d)
        if resolved not in DATASET_REGISTRY:
            print(f"WARNING: unknown dataset '{d}' -- skipping")
            continue
        datasets.append(resolved)

    print("WindSight Evaluation Runner")
    print(f"  Model       : {model_path}")
    print(f"  Tokenizer   : {tokenizer_path}")
    print(f"  Datasets    : {datasets}")
    print(f"  Horizons    : {pred_horizons}h")
    print(f"  Samples     : {args.n_samples}")
    print(f"  Lookback    : {args.lookback}h")
    print(f"  Device      : {args.device}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Top-p       : {args.top_p}")
    print(f"  Output dir  : {output_dir}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        device=args.device,
    )

    predictor = WindFMPredictor(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_context=512,
        clip=5,
    )
    print("  Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    all_results = {}
    wall_start = time.time()

    for ds_key in datasets:
        ds_results = evaluate_dataset(
            dataset_key=ds_key,
            predictor=predictor,
            pred_horizons=pred_horizons,
            n_samples=args.n_samples,
            n_turbines=args.n_turbines,
            lookback=args.lookback,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_results[ds_key] = ds_results

    wall_elapsed = time.time() - wall_start

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Derive a short model name for the output filename
    model_name = model_path.stem if model_path.is_file() else model_path.name
    results_file = output_dir / f"results_{model_name}.json"

    output_payload = {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "pred_horizons": pred_horizons,
        "n_samples": args.n_samples,
        "lookback": args.lookback,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": args.device,
        "wall_time_seconds": round(wall_elapsed, 2),
        "results": all_results,
    }

    with open(results_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary_table(all_results)
    print(f"Total wall time: {wall_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
