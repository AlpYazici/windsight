"""
Progressive fine-tuning script for the WindFM autoregressive Transformer.

Supports 2-stage training:
  Stage 1: Train on SDWPF data
    python3 -m src.training.finetune --epochs 50 --batch_size 64

  Stage 2: Resume from Stage 1 checkpoint, train on multi-site data
    python3 -m src.training.finetune \
        --train_data data/sdwpf/processed/train.parquet,data/kelmarsh/processed/train.parquet,data/penmanshiel/processed/train.parquet \
        --val_data data/sdwpf/processed/val.parquet,data/kelmarsh/processed/val.parquet \
        --resume_from outputs/stage1/best_model.pt \
        --output_dir outputs/stage2 \
        --epochs 30 --lr 5e-6

The tokenizer is frozen; only the AR model parameters are updated.
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Ensure the WindFM package is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WINDFM_DIR = _PROJECT_ROOT / "WindFM"
if str(_WINDFM_DIR) not in sys.path:
    sys.path.insert(0, str(_WINDFM_DIR))

from model.windfm import WindFM, WindFMTokenizer, calc_time_stamps  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "windfm"
_DEFAULT_TOKENIZER_PATH = _PROJECT_ROOT / "models" / "windfm-tokenizer"
_DEFAULT_TRAIN_DATA = str(_PROJECT_ROOT / "data" / "sdwpf" / "processed" / "train.parquet")
_DEFAULT_VAL_DATA = str(_PROJECT_ROOT / "data" / "sdwpf" / "processed" / "val.parquet")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "stage1"

FEATURE_COLS = ["wind_speed", "wind_direction", "power", "density", "temperature", "pressure"]


# ============================================================================
# Dataset
# ============================================================================

class WindFMFinetuneDataset(Dataset):
    """
    Sliding-window dataset over per-turbine time series.

    Accepts one or more parquet files (for multi-dataset training).  All
    parquets are concatenated and windows are extracted per turbine.

    Each sample is a window of ``seq_len`` consecutive hourly steps from one
    turbine.  The raw features are z-score-normalised **per window** (matching
    inference behaviour) and then encoded through the frozen tokenizer into
    (s1_ids, s2_ids) pairs.

    For autoregressive training the tokens are shifted:
        input  = tokens[:-1]
        target = tokens[1:]
    Timestamps are computed once and stored alongside.
    """

    def __init__(
        self,
        parquet_paths: str | Path | list[str | Path],
        tokenizer: WindFMTokenizer,
        seq_len: int = 512,
        stride: int = 128,
        clip: float = 5.0,
        num_turbines: int = 0,
        device: str = "mps",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.clip = clip
        self.device = device

        # Support single path or list of paths
        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]
        dfs = [pd.read_parquet(p) for p in parquet_paths]
        df = pd.concat(dfs, ignore_index=True)
        df["TurbID"] = df["TurbID"].astype(str)
        turbine_ids = sorted(df["TurbID"].unique())
        if num_turbines > 0:
            turbine_ids = turbine_ids[:num_turbines]

        # Pre-compute all windows ------------------------------------------
        self.s1_ids_list: list[torch.Tensor] = []
        self.s2_ids_list: list[torch.Tensor] = []
        self.stamps_list: list[torch.Tensor] = []

        tokenizer.eval()
        with torch.no_grad():
            for tid in turbine_ids:
                tdf = df[df["TurbID"] == tid].sort_values("time").reset_index(drop=True)
                features = tdf[FEATURE_COLS].values.astype(np.float32)  # (T, 6)
                timestamps = pd.to_datetime(tdf["time"])

                n_steps = len(features)
                if n_steps < seq_len:
                    continue

                for start in range(0, n_steps - seq_len + 1, stride):
                    end = start + seq_len
                    window = features[start:end]  # (seq_len, 6)
                    ts_window = timestamps.iloc[start:end]

                    # --- per-window z-score normalisation ---
                    mean = window.mean(axis=0)
                    std = window.std(axis=0)
                    normed = (window - mean) / (std + 1e-6)
                    normed = np.clip(normed, -clip, clip)

                    # --- timestamp features ---
                    stamp_df = calc_time_stamps(ts_window)
                    stamp = torch.tensor(
                        stamp_df.values.astype(np.float32), dtype=torch.float32
                    )  # (seq_len, 5)

                    # --- tokenise ---
                    x_tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(device)
                    try:
                        indices = tokenizer.encode(x_tensor, half=True)
                    except RuntimeError:
                        # MPS scatter_reduce fallback: encode on CPU
                        x_cpu = x_tensor.to("cpu")
                        tok_cpu = tokenizer.to("cpu")
                        indices = tok_cpu.encode(x_cpu, half=True)
                        tokenizer.to(device)

                    s1 = indices[0].squeeze(0).cpu()  # (seq_len,)
                    s2 = indices[1].squeeze(0).cpu()  # (seq_len,)

                    self.s1_ids_list.append(s1)
                    self.s2_ids_list.append(s2)
                    self.stamps_list.append(stamp)

        print(f"[Dataset] {len(self.s1_ids_list)} windows from {len(turbine_ids)} turbines "
              f"(seq_len={seq_len}, stride={stride})")

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.s1_ids_list)

    def __getitem__(self, idx: int):
        s1 = self.s1_ids_list[idx]   # (seq_len,)
        s2 = self.s2_ids_list[idx]   # (seq_len,)
        stamp = self.stamps_list[idx] # (seq_len, 5)

        # Autoregressive shift: input = [:-1], target = [1:]
        return {
            "s1_input": s1[:-1],
            "s2_input": s2[:-1],
            "s1_target": s1[1:],
            "s2_target": s2[1:],
            "stamp": stamp[:-1],
        }


# ============================================================================
# LR scheduler: cosine with linear warmup
# ============================================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ============================================================================
# Training loop
# ============================================================================

def run_epoch(model, dataloader, device, optimizer=None, scheduler=None, max_grad_norm=1.0):
    """Run one epoch (train or eval). Returns dict of averaged metrics."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_ce_s1 = 0.0
    total_ce_s2 = 0.0
    n_batches = 0

    for batch in dataloader:
        s1_input = batch["s1_input"].to(device)
        s2_input = batch["s2_input"].to(device)
        s1_target = batch["s1_target"].to(device)
        s2_target = batch["s2_target"].to(device)
        stamp = batch["stamp"].to(device)

        s1_logits, s2_logits = model(
            s1_input, s2_input, stamp,
            use_teacher_forcing=True,
            s1_targets=s1_target,
        )

        loss, ce_s1, ce_s2 = model.head.compute_loss(
            s1_logits, s2_logits, s1_target, s2_target
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_ce_s1 += ce_s1.item()
        total_ce_s2 += ce_s2.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "ce_s1": total_ce_s1 / max(n_batches, 1),
        "ce_s2": total_ce_s2 / max(n_batches, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune WindFM (supports progressive 2-stage training)")

    # data
    parser.add_argument("--train_data", type=str, default=_DEFAULT_TRAIN_DATA,
                        help="Comma-separated parquet paths for training")
    parser.add_argument("--val_data", type=str, default=_DEFAULT_VAL_DATA,
                        help="Comma-separated parquet paths for validation")
    parser.add_argument("--model_path", type=str, default=str(_DEFAULT_MODEL_PATH))
    parser.add_argument("--tokenizer_path", type=str, default=str(_DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output_dir", type=str, default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to .pt checkpoint to resume from (e.g., outputs/stage1/best_model.pt)")

    # hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", "--max_epochs", type=int, default=50, dest="max_epochs")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_turbines", type=int, default=0,
                        help="0 = all turbines, else subsample for faster experiments")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse comma-separated data paths
    train_paths = [p.strip() for p in args.train_data.split(",")]
    val_paths = [p.strip() for p in args.val_data.split(",")]

    print("=" * 60)
    print("WindFM Fine-Tuning (Progressive)")
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

    # Resume from checkpoint if specified (load before moving to device)
    if args.resume_from:
        print(f"  Resuming from checkpoint: {args.resume_from}")
        ckpt_state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt_state)
        print("  Checkpoint loaded successfully.")

    tokenizer.to(device)
    model.to(device)

    # Freeze tokenizer
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # Ensure model is trainable
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in tokenizer.parameters())
    print(f"  Trainable params (model):  {trainable:,}")
    print(f"  Frozen params (tokenizer): {frozen:,}")

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
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    print("\n[3/4] Setting up optimiser and scheduler ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

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
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")

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
            torch.save(model.state_dict(), best_ckpt_path)
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

        # Save log after every epoch (include args for reproducibility)
        log_path = os.path.join(args.output_dir, "training_log.json")
        log_payload = {
            "args": vars(args),
            "train_paths": train_paths,
            "val_paths": val_paths,
            "epochs": log,
        }
        with open(log_path, "w") as f:
            json.dump(log_payload, f, indent=2)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
            break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
