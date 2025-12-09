"""Optional capacity bridge applied before the LoRA drafter head."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Minimal RMSNorm variant shared by the drafter bridge."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        dims = (-1,)  # last dim always hidden size
        rms = x.to(torch.float32).pow(2).mean(dim=dims, keepdim=True)
        x_normed = x * torch.rsqrt(rms + self.eps)
        x_normed = x_normed.to(input_dtype)
        weight = self.weight
        if x_normed.dim() == 3:
            weight = weight.view(1, 1, -1)
        else:
            weight = weight.view(1, -1)
        return x_normed * weight


class DrafterBridge(nn.Module):
    """Small trainable bridge (RMSNorm → core → RMSNorm) applied to hk with identity init."""

    def __init__(
        self,
        d_model: int,
        *,
        kind: Literal["mha", "ffn"] = "ffn",
        n_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kind not in {"mha", "ffn"}:
            raise ValueError("bridge kind must be 'mha' or 'ffn'")
        self.kind = kind
        self.norm_in = RMSNorm(d_model)
        if kind == "mha":
            self.core: nn.Module = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
                bias=False,
            )
        else:  # kind == "ffn"
            self.core = nn.Sequential(
                nn.Linear(d_model, 4 * d_model, bias=False),
                nn.SiLU(),
                nn.Linear(4 * d_model, d_model, bias=False),
            )
        self.norm_out = RMSNorm(d_model)
        self._init_identity()

    def _init_identity(self) -> None:
        """Zero out the core so the bridge starts as an identity residual."""

        if self.kind == "mha":
            mha: nn.MultiheadAttention = self.core  # type: ignore[assignment]
            nn.init.zeros_(mha.in_proj_weight)
            if mha.in_proj_bias is not None:
                nn.init.zeros_(mha.in_proj_bias)
            nn.init.zeros_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                nn.init.zeros_(mha.out_proj.bias)
        else:
            for layer in self.core:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, hk: torch.Tensor) -> torch.Tensor:
        squeeze = False
        original_dtype = hk.dtype
        if hk.dim() == 2:
            hk_reshaped = hk.unsqueeze(1)
            squeeze = True
        else:
            hk_reshaped = hk

        residual = hk_reshaped.to(torch.float32)
        x = self.norm_in(hk_reshaped.to(torch.float32))
        if self.kind == "mha":
            delta, _ = self.core(x, x, x, need_weights=False)
        else:
            delta = self.core(x)
        delta = self.norm_out(delta)
        out = residual + delta
        out = out.to(original_dtype)

        if squeeze:
            out = out.squeeze(1)
        return out


__all__ = ["DrafterBridge"]
