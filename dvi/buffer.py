from __future__ import annotations

from typing import Dict, Optional

import torch


class DVIRingBuffer:
    """GPU-resident ring buffer for online DVI training.

    The buffer is single-producer/single-consumer and **not** thread-safe.
    Storage is preallocated during construction and lives entirely on the
    provided device. Push operations expect tensors on the same device and
    perform dtype conversions only when necessary.
    """

    def __init__(
        self,
        *,
        capacity: int,
        d_model: int,
        vocab_size: int,
        store: str = "topk",
        topk: int = 1024,
        logits_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        store_mode = store.lower()
        if store_mode not in {"full", "topk"}:
            raise ValueError("store must be either 'full' or 'topk'")
        if store_mode == "topk":
            if topk <= 0:
                raise ValueError("topk must be positive when store='topk'")
            if topk > vocab_size:
                raise ValueError("topk cannot exceed vocab_size")

        self._cap = int(capacity)
        self._d_model = int(d_model)
        self._vocab_size = int(vocab_size)
        self._store = store_mode
        self._topk = int(topk)
        self._logits_dtype = logits_dtype
        self._device = torch.device(device)

        # Preallocate storage tensors
        self._hk = torch.empty((self._cap, self._d_model), dtype=torch.float16, device=self._device)
        self._token = torch.empty((self._cap,), dtype=torch.int64, device=self._device)
        self._reward = torch.empty((self._cap,), dtype=torch.int8, device=self._device)
        self._pos = torch.empty((self._cap,), dtype=torch.int16, device=self._device)
        self._is_first_reject = torch.empty((self._cap,), dtype=torch.bool, device=self._device)

        if self._store == "full":
            self._z_phi = torch.empty((self._cap, self._vocab_size), dtype=self._logits_dtype, device=self._device)
            self._z_idx = None
            self._z_val = None
        else:
            self._z_phi = None
            self._z_idx = torch.empty((self._cap, self._topk), dtype=torch.int32, device=self._device)
            self._z_val = torch.empty((self._cap, self._topk), dtype=self._logits_dtype, device=self._device)

        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def device(self) -> torch.device:
        return self._device

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    def stats(self) -> Dict[str, float]:
        fill_frac = float(self._size) / float(self._cap)
        hk_bytes = self._hk.element_size() * self._hk.numel()
        token_bytes = self._token.element_size() * self._token.numel()
        reward_bytes = self._reward.element_size() * self._reward.numel()
        pos_bytes = self._pos.element_size() * self._pos.numel()
        if self._store == "full":
            logits_bytes = self._z_phi.element_size() * self._z_phi.numel()  # type: ignore[union-attr]
        else:
            logits_bytes = (
                self._z_idx.element_size() * self._z_idx.numel()  # type: ignore[union-attr]
                + self._z_val.element_size() * self._z_val.numel()  # type: ignore[union-attr]
            )
        total_bytes = hk_bytes + token_bytes + reward_bytes + pos_bytes + logits_bytes
        return {
            "size": self._size,
            "capacity": self._cap,
            "fill_frac": fill_frac,
            "hk_bytes": float(hk_bytes),
            "meta_bytes": float(token_bytes + reward_bytes + pos_bytes),
            "logits_bytes": float(logits_bytes),
            "total_bytes": float(total_bytes),
        }

    def _write_common(
        self,
        idx: int,
        *,
        hk: torch.Tensor,
        token: int,
        reward: int,
        pos: int,
        is_first_reject: bool,
    ) -> None:
        if hk.device != self._device:
            raise ValueError("hk tensor must be on buffer device")
        if hk.dim() != 1 or hk.shape[0] != self._d_model:
            raise ValueError(f"hk must have shape [{self._d_model}]")
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        if pos <= 0 or pos > 32767:
            raise ValueError("pos must be in [1, 32767]")

        if hk.dtype != torch.float16:
            hk = hk.to(dtype=torch.float16)
        self._hk[idx].copy_(hk)

        self._token[idx] = int(token)
        self._reward[idx] = int(reward)
        self._pos[idx] = int(pos)
        self._is_first_reject[idx] = bool(is_first_reject)

    def _advance(self) -> int:
        idx = self._ptr % self._cap
        self._ptr += 1
        if self._size < self._cap:
            self._size += 1
        return idx

    def push_full(
        self,
        *,
        hk: torch.Tensor,
        token: int,
        z_phi: torch.Tensor,
        reward: int,
        pos: int,
        is_first_reject: bool,
    ) -> None:
        if self._store != "full":
            raise RuntimeError("push_full called on buffer configured for topk storage")
        if z_phi.device != self._device:
            raise ValueError("z_phi tensor must be on buffer device")
        if z_phi.dim() != 1 or z_phi.shape[0] != self._vocab_size:
            raise ValueError(f"z_phi must have shape [{self._vocab_size}]")

        idx = self._advance()
        self._write_common(idx, hk=hk, token=token, reward=reward, pos=pos, is_first_reject=is_first_reject)

        if z_phi.dtype != self._logits_dtype:
            z_phi = z_phi.to(dtype=self._logits_dtype)
        self._z_phi[idx].copy_(z_phi)  # type: ignore[index]

    def push_topk(
        self,
        *,
        hk: torch.Tensor,
        token: int,
        z_phi: torch.Tensor,
        reward: int,
        pos: int,
        is_first_reject: bool,
    ) -> None:
        if self._store != "topk":
            raise RuntimeError("push_topk called on buffer configured for full storage")
        if z_phi.device != self._device:
            raise ValueError("z_phi tensor must be on buffer device")
        if z_phi.dim() != 1 or z_phi.shape[0] != self._vocab_size:
            raise ValueError(f"z_phi must have shape [{self._vocab_size}]")

        values, indices = torch.topk(z_phi, k=self._topk, dim=0)
        if values.dtype != self._logits_dtype:
            values = values.to(dtype=self._logits_dtype)
        indices = indices.to(dtype=torch.int32)

        idx = self._advance()
        self._write_common(idx, hk=hk, token=token, reward=reward, pos=pos, is_first_reject=is_first_reject)

        self._z_idx[idx].copy_(indices)  # type: ignore[index]
        self._z_val[idx].copy_(values)  # type: ignore[index]

    def sample(self, n: int) -> Dict[str, Optional[torch.Tensor]]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if n < 0:
            raise ValueError("Sample size must be non-negative")
        if n == 0:
            base = {
                "hk": self._hk[:0],
                "token": self._token[:0],
                "reward": self._reward[:0].to(torch.float32),
                "pos": self._pos[:0].to(torch.int32),
                "is_first_reject": self._is_first_reject[:0],
                "z_phi": None,
                "z_idx": None,
                "z_val": None,
            }
            if self._store == "full":
                base["z_phi"] = self._z_phi[:0]  # type: ignore[index]
            else:
                base["z_idx"] = self._z_idx[:0]  # type: ignore[index]
                base["z_val"] = self._z_val[:0]  # type: ignore[index]
            return base

        indices = torch.randint(0, self._size, (n,), device=self._device)

        batch: Dict[str, Optional[torch.Tensor]] = {
            "hk": self._hk.index_select(0, indices),
            "token": self._token.index_select(0, indices),
            "reward": self._reward.index_select(0, indices).to(torch.float32),
            "pos": self._pos.index_select(0, indices).to(torch.int32),
            "is_first_reject": self._is_first_reject.index_select(0, indices),
            "z_phi": None,
            "z_idx": None,
            "z_val": None,
        }

        if self._store == "full":
            batch["z_phi"] = self._z_phi.index_select(0, indices)  # type: ignore[index]
        else:
            batch["z_idx"] = self._z_idx.index_select(0, indices)  # type: ignore[index]
            batch["z_val"] = self._z_val.index_select(0, indices)  # type: ignore[index]

        return batch
