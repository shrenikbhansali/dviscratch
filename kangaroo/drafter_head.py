"""LoRA-augmented drafter head used for draft logits."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and a LoRA update.

    The effective weight is ``W = W_0 + (alpha / r) * A @ B`` with ``A`` and ``B``
    trainable and the base projection ``W_0`` kept frozen. When ``r`` or ``alpha``
    are zero, the module reduces to the frozen base projection exactly.
    """

    def __init__(self, in_f: int, out_f: int, r: int, alpha: float, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.rank = int(r)
        self.alpha = float(alpha)
        self.base = nn.Linear(in_f, out_f, bias=bias)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        if self.rank > 0:
            self.A = nn.Parameter(torch.zeros(out_f, self.rank))
            self.B = nn.Parameter(torch.zeros(self.rank, in_f))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus the scaled low-rank update."""

        result = self.base(x)
        if self.rank <= 0 or self.alpha == 0.0:
            return result

        lora_update = F.linear(x, self.B)
        lora_update = F.linear(lora_update, self.A)
        return result + self.scaling * lora_update

    def trainable_params(self) -> List[nn.Parameter]:
        """Return the LoRA parameters ``[A, B]`` if trainable, otherwise ``[]``."""

        if self.rank <= 0 or self.alpha == 0.0:
            return []
        return [self.A, self.B]


class DrafterHead(nn.Module):
    """Projection head producing draft logits from shallow hidden states."""

    def __init__(self, hidden_size: int, vocab_size: int, r: int, alpha: float, *, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.proj = LoRALinear(hidden_size, vocab_size, r=r, alpha=alpha, bias=bias)

    def forward(self, hk: torch.Tensor) -> torch.Tensor:
        """Project shallow hidden states ``hk`` to vocabulary logits."""

        return self.proj(hk)

    def lora_params(self) -> List[nn.Parameter]:
        """Return the LoRA parameters belonging to the projection."""

        return self.proj.trainable_params()
