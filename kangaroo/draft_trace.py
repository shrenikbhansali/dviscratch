from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class DraftBlockTrace:
    """Immutable per-block trace of drafted tokens and their shallow states."""

    tokens: List[int]
    hk_fp16: torch.Tensor

    def __len__(self) -> int:
        return len(self.tokens)

    def hk_state_of(self, i: int) -> torch.Tensor:
        if i < 1 or i > len(self.tokens):
            raise IndexError(f"draft index {i} out of range for length {len(self)}")
        return self.hk_fp16[i - 1]

    @staticmethod
    def empty(d_model: int, max_len: int, device: torch.device) -> "DraftBlockTraceScratch":
        return DraftBlockTraceScratch(d_model=d_model, max_len=max_len, device=device)


class DraftBlockTraceScratch:
    """Mutable scratch buffer for capturing drafted states within a block."""

    def __init__(self, d_model: int, max_len: int, device: torch.device):
        self._d_model = int(d_model)
        self._max_len = int(max_len)
        if self._max_len < 0:
            raise ValueError("max_len must be non-negative")
        self._hk_fp16_buf = torch.empty((self._max_len, self._d_model), dtype=torch.float16, device=device)
        self._tokens: List[int] = []
        self._next_index = 0
        self._finalized = False

    def append(self, hk: torch.Tensor, token_id: int) -> None:
        if self._finalized:
            raise RuntimeError("cannot append after finalize()")
        if self._next_index >= self._max_len:
            raise RuntimeError("draft trace capacity exceeded")

        if not isinstance(token_id, int):
            token_id = int(token_id)

        hk_detached = hk.detach()
        hk_flat = hk_detached.reshape(-1)
        if hk_flat.numel() != self._d_model:
            raise ValueError(
                f"expected hidden size {self._d_model}, got tensor with {hk_flat.numel()} elements"
            )
        if hk_flat.dtype != torch.float16:
            hk_flat = hk_flat.to(dtype=torch.float16)
        self._hk_fp16_buf[self._next_index].copy_(hk_flat)

        self._tokens.append(token_id)
        self._next_index += 1

    def finalize(self) -> DraftBlockTrace:
        if self._finalized:
            raise RuntimeError("finalize() has already been called")
        self._finalized = True
        truncated = self._hk_fp16_buf[: self._next_index]
        return DraftBlockTrace(tokens=list(self._tokens), hk_fp16=truncated)

