"""
LoRA fine-tuning script for WindFM.

Only trains low-rank adapters on attention Q/V projections.
All original weights stay frozen → preserves sampling distribution.

Usage:
    python3 -m src.training.finetune_lora \
        --train_data data/sdwpf/processed/train.parquet,data/kelmarsh/processed/train.parquet,data/penmanshiel/processed/train.parquet \
        --val_data data/sdwpf/processed/val.parquet,data/kelmarsh/processed/val.parquet,data/penmanshiel/processed/val.parquet \
        --output_dir outputs/lora \
        --rank 8 --alpha 16 --lr 1e-4 --epochs 50
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WINDFM_DIR = _PROJECT_ROOT / "WindFM"
if str(_WINDFM_DIR) not in sys.path:
    sys.path.insert(0, str(_WINDFM_DIR))

from model.windfm import WindFM, WindFMTokenizer, calc_time_stamps  # noqa: E402

# Reuse dataset and scheduler from main finetune script
from src.training.finetune import WindFMFinetuneDataset, CosineWarmupScheduler, run_epoch  # noqa: E402
from src.training.lora import apply_lora, get_lora_state_dict, merge_lora  # noqa: E402

# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "windfm"
_DEFAULT_TOKENIZER_PATH = _PROJECT_ROOT / "models" / "windfm-tokenizer"
_DEFAULT_TRAIN_DATA = str(_PROJECT_ROOT / "data" / "sdwpf" / "processed" / "train.parquet")
_DEFAULT_VAL_DATA = str(_PROJECT_ROOT / "data" / "sdwpf" / "processed" / "val.parquet")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "lora"


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune WindFM")

    parser.add_argument("--train_data", type=str, default=_DEFAULT_TRAIN_DATA)
    parser.add_argument("--val_data", type=str, default=_DEFAULT_VAL_DATA)
    parser.add_argument("--model_path", type=str, default=str(_DEFAULT_MODEL_PATH))
    parser.add_argument("--tokenizer_path", type=str, default=str(_DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output_dir", type=str, default=str(_DEFAULT_OUTPUT_DIR))

    # LoRA hyperparameters
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Higher LR than full fine-tune since fewer params")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", "--max_epochs", type=int, default=50, dest="max_epochs")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_turbines", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    train_paths = [p.strip() for p in args.train_data.split(",")]
    val_paths = [p.strip() for p in args.val_data.split(",")]

    print("=" * 60)
    print("WindFM LoRA Fine-Tuning")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  train_paths: {train_paths}")
    print(f"  val_paths:   {val_paths}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\n[1/4] Loading tokenizer and model ...")
    tokenizer = WindFMTokenizer.from_pretrained(args.tokenizer_path)
    model = WindFM.from_pretrained(args.model_path)

    # Apply LoRA BEFORE moving to device
    model = apply_lora(model, rank=args.rank, alpha=args.alpha)

    tokenizer.to(device)
    model.to(device)

    # Freeze tokenizer
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------------
    print("\n[2/4] Building datasets ...")
    train_ds = WindFMFinetuneDataset(
        parquet_paths=train_paths,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=args.stride,
        num_turbines=args.num_turbines,
        device=device,
    )
    val_ds = WindFMFinetuneDataset(
        parquet_paths=val_paths,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=args.stride,
        num_turbines=args.num_turbines,
        device=device,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Optimiser (only LoRA params)
    # ------------------------------------------------------------------
    print("\n[3/4] Setting up optimiser ...")
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    print(f"  Total steps: {total_steps}  |  Warmup steps: {warmup_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n[4/4] Training ...\n")
    best_val_loss = float("inf")
    patience_counter = 0
    log: list[dict] = []
    best_ckpt_path = os.path.join(args.output_dir, "best_lora.pt")
    merged_ckpt_path = os.path.join(args.output_dir, "best_model_merged.pt")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model, train_loader, device,
            optimizer=optimizer, scheduler=scheduler,
            max_grad_norm=args.max_grad_norm,
        )

        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "train_ce_s1": round(train_metrics["ce_s1"], 6),
            "train_ce_s2": round(train_metrics["ce_s2"], 6),
            "val_ce_s1": round(val_metrics["ce_s1"], 6),
            "val_ce_s2": round(val_metrics["ce_s2"], 6),
            "lr": current_lr,
            "time_s": round(elapsed, 1),
        }
        log.append(entry)

        improved = val_metrics["loss"] < best_val_loss
        marker = ""
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            # Save only LoRA weights (tiny file)
            torch.save(get_lora_state_dict(model), best_ckpt_path)
            # Also save merged model for easy evaluation
            import copy
            merged = copy.deepcopy(model)
            merged = merge_lora(merged)
            torch.save(merged.state_dict(), merged_ckpt_path)
            del merged
            marker = " *"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{args.max_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"ce_s1={val_metrics['ce_s1']:.4f} | "
            f"ce_s2={val_metrics['ce_s2']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s{marker}"
        )

        log_path = os.path.join(args.output_dir, "training_log.json")
        log_payload = {
            "args": vars(args),
            "train_paths": train_paths,
            "val_paths": val_paths,
            "method": "lora",
            "epochs": log,
        }
        with open(log_path, "w") as f:
            json.dump(log_payload, f, indent=2)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
            break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"LoRA checkpoint: {best_ckpt_path}")
    print(f"Merged checkpoint: {merged_ckpt_path}")


if __name__ == "__main__":
    main()
