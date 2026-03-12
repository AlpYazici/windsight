"""
LoRA (Low-Rank Adaptation) for WindFM.

Applies low-rank adapters to attention Q and V projections,
keeping all original weights frozen. This preserves the pre-trained
sampling distribution while allowing domain adaptation.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter.

    output = frozen_linear(x) + scale * B(A(x))
    where A: (in, r), B: (r, out), scale = alpha/r
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scale = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A) @ self.lora_B * self.scale
        return base + lora


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Apply LoRA adapters to all Q and V projections in the model.

    Freezes ALL original parameters and only leaves LoRA params trainable.

    Returns the modified model.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0

    # Apply to transformer self-attention layers
    if hasattr(model, 'transformer'):
        for i, layer in enumerate(model.transformer):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                attn.q_proj = LoRALinear(attn.q_proj, rank=rank, alpha=alpha)
                attn.v_proj = LoRALinear(attn.v_proj, rank=rank, alpha=alpha)
                lora_count += 2

    # Apply to dep_layer cross-attention
    if hasattr(model, 'dep_layer') and hasattr(model.dep_layer, 'cross_attn'):
        cross_attn = model.dep_layer.cross_attn
        cross_attn.q_proj = LoRALinear(cross_attn.q_proj, rank=rank, alpha=alpha)
        cross_attn.v_proj = LoRALinear(cross_attn.v_proj, rank=rank, alpha=alpha)
        lora_count += 2

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[LoRA] Applied to {lora_count} projections (rank={rank}, alpha={alpha})")
    print(f"[LoRA] Trainable: {trainable:,} | Frozen: {frozen:,} | Ratio: {trainable/frozen:.4%}")

    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters for saving (tiny checkpoint)."""
    return {
        name: param.cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the original linear layers (for inference).

    After merging, the model behaves identically but without the LoRA overhead.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_A @ module.lora_B) * module.scale
                module.original.weight.add_(delta.T)
    return model
