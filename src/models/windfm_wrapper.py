"""
Wrapper around the WindFM model and tokenizer for inference and fine-tuning access.

Loads pre-trained weights from local safetensors files and provides:
  - inference via the WindFMPredictor API (configurable device)
  - direct access to model / tokenizer nn.Modules for fine-tuning
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Make the WindFM package importable regardless of where we run from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WINDFM_DIR = _PROJECT_ROOT / "WindFM"
if str(_WINDFM_DIR) not in sys.path:
    sys.path.insert(0, str(_WINDFM_DIR))

from model.windfm import WindFM, WindFMTokenizer, WindFMPredictor, calc_time_stamps  # noqa: E402


# Default local weight paths (relative to project root)
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "windfm"
_DEFAULT_TOKENIZER_PATH = _PROJECT_ROOT / "models" / "windfm-tokenizer"


class WindFMWrapper:
    """High-level wrapper that owns a WindFM AR model + tokenizer pair."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str = "mps",
        max_context: int = 512,
        clip: float = 5.0,
    ):
        self.device = device
        self.max_context = max_context
        self.clip = clip

        model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        tokenizer_path = Path(tokenizer_path) if tokenizer_path else _DEFAULT_TOKENIZER_PATH

        # Load tokenizer (always from HF-format directory)
        self.tokenizer: WindFMTokenizer = WindFMTokenizer.from_pretrained(
            str(tokenizer_path)
        )

        # Load model: supports both HF directories and .pt checkpoints
        if model_path.is_file() and model_path.suffix in (".pt", ".pth"):
            import torch
            self.model: WindFM = WindFM.from_pretrained(str(_DEFAULT_MODEL_PATH))
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model: WindFM = WindFM.from_pretrained(str(model_path))

        self.tokenizer.to(self.device)
        self.model.to(self.device)
        self.tokenizer.eval()
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference (batched sampling to avoid OOM)
    # ------------------------------------------------------------------
    _SAMPLES_PER_BATCH = 10  # max samples per forward pass

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run probabilistic AR inference in small batches to control memory.

        WindFM's auto_regressive_inference repeats all tensors sample_count
        times in GPU memory.  With sample_count=100 on MPS this easily
        exceeds 100 GB.  We split into batches of _SAMPLES_PER_BATCH and
        concatenate the results.
        """
        import gc

        predictor = WindFMPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_context=self.max_context,
            clip=self.clip,
        )

        batch_size = self._SAMPLES_PER_BATCH
        all_dfs: list[pd.DataFrame] = []
        samples_done = 0

        while samples_done < sample_count:
            chunk = min(batch_size, sample_count - samples_done)
            chunk_df = predictor.predict(
                df=df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_k=top_k,
                top_p=top_p,
                sample_count=chunk,
                verbose=(verbose and samples_done == 0),
            )
            # Rename columns to avoid clashes when concatenating
            chunk_df.columns = [
                f"pred-{samples_done + i}" for i in range(chunk)
            ]
            all_dfs.append(chunk_df)
            samples_done += chunk

            # Free MPS memory between batches
            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()

        return pd.concat(all_dfs, axis=1)

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------
    def get_model(self) -> WindFM:
        """Return the AR Transformer model (for fine-tuning)."""
        return self.model

    def get_tokenizer(self) -> WindFMTokenizer:
        """Return the tokenizer (should be frozen during fine-tuning)."""
        return self.tokenizer

    def freeze_tokenizer(self) -> None:
        """Freeze all tokenizer parameters so they are not updated."""
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def unfreeze_model(self) -> None:
        """Ensure all AR model parameters are trainable."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
