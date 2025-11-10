from __future__ import annotations

import time
from typing import Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_

from .schedule import PiecewiseSchedule


def mean_over_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    total = (x * mask_f).sum()
    return total / denom


class OnlineTrainer:
    def __init__(
        self,
        model,
        *,
        lr: float,
        tau: float,
        schedule: PiecewiseSchedule,
        max_ms: int,
        store_mode: str,
        topk: int,
        vocab_size: int,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        ema_m: float = 0.01,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tau = float(tau)
        self.schedule = schedule
        self.max_ms = int(max_ms)
        self.store_mode = store_mode
        self.topk = int(topk)
        self.vocab_size = int(vocab_size)
        self.grad_clip = float(grad_clip)
        self.ema_m = float(ema_m)
        self.ema_b = 0.0

        self._trainable = list(model.dvi_trainable_params())
        if not self._trainable:
            raise RuntimeError("No trainable LoRA parameters returned by model.dvi_trainable_params().")
        trainable_ids = {id(p) for p in self._trainable}
        offending = [name for name, p in model.named_parameters() if id(p) not in trainable_ids and p.requires_grad]
        if offending:
            raise RuntimeError(
                "Non-LoRA parameters have requires_grad=True: " + ", ".join(offending)
            )
        for param in self._trainable:
            if not param.requires_grad:
                param.requires_grad_(True)
        debug_stats = ", ".join(
            f"{tuple(p.shape)}@{p.device}:{p.dtype}=grad{p.requires_grad}"
            for p in self._trainable
        )
        print(f"[DVI][debug]init trainables -> {debug_stats}")
        self._debug_once = False

        if device is None:
            self._device = self._trainable[0].device
        else:
            self._device = device
        self.opt = torch.optim.AdamW(self._trainable, lr=lr, weight_decay=weight_decay)

        if self.store_mode not in {"full", "topk"}:
            raise ValueError(f"Unknown store_mode={self.store_mode}")
        if self.store_mode == "topk":
            if not (0 < self.topk <= self.vocab_size):
                raise ValueError("topk must satisfy 0 < topk <= vocab_size when store_mode='topk'")
        else:
            self.topk = self.vocab_size

    def step(self, batch: Dict[str, torch.Tensor | None], global_step: int) -> Dict[str, float]:
        del global_step
        t0 = time.time()
        eps = 1e-12

        hk = batch["hk"].to(self._device).detach()
        token = batch["token"].to(self._device)
        reward = batch["reward"].to(self._device)
        pos = batch["pos"].to(self._device)
        is_first_reject = batch["is_first_reject"].to(self._device)
        _ = pos

        z_phi = batch.get("z_phi")
        z_idx = batch.get("z_idx")
        z_val = batch.get("z_val")
        if self.store_mode == "full":
            assert z_phi is not None and z_idx is None and z_val is None, (
                "full mode requires z_phi and forbids z_idx/z_val"
            )
            z_phi = z_phi.to(self._device)
        elif self.store_mode == "topk":
            assert z_phi is None and z_idx is not None and z_val is not None, (
                "topk mode requires z_idx/z_val and forbids z_phi"
            )
            z_idx = z_idx.to(self._device).long()
            z_val = z_val.to(self._device)
        else:
            raise ValueError(f"Unknown store_mode={self.store_mode}")

        B = hk.size(0)

        with torch.enable_grad():
            draft_logits = self.model.drafter_logits_from_hk(hk)
            if not draft_logits.requires_grad:
                dbg = ", ".join(
                    f"{tuple(p.shape)}:requires_grad={p.requires_grad}"
                    for p in self._trainable
                )
                print(f"[DVI][debug] draft_logits lacks grad_fn; trainables -> {dbg}")
                raise RuntimeError(
                    "drafter logits are detached from autograd; ensure LoRA parameters remain trainable."
                )
            if not self._debug_once:
                stats = ", ".join(
                    f"{tuple(p.shape)}:grad={p.requires_grad}"
                    for p in self._trainable
                )
                print(f"[DVI][debug] trainables -> {stats}, logits require_grad={draft_logits.requires_grad}")
                self._debug_once = True
            logp = torch.log_softmax(draft_logits.float(), dim=-1)
            nll = -logp[torch.arange(B, device=logp.device), token]

            if self.store_mode == "full":
                log_p_tau = torch.log_softmax(z_phi.float() / self.tau, dim=-1)
                p_tau = log_p_tau.exp()
                kd = (p_tau * (log_p_tau - logp)).sum(dim=-1).mean()
                p = logp.exp()
                rkl = (p * (logp - log_p_tau)).sum(dim=-1).mean()
            else:
                gathered = draft_logits.gather(1, z_idx)
                logp_k = torch.log_softmax(gathered.float(), dim=-1)
                p_k = logp_k.exp()
                log_p_tau_k = torch.log_softmax(z_val.float() / self.tau, dim=-1)
                p_tau_k = log_p_tau_k.exp()
                kd = (p_tau_k * (log_p_tau_k - logp_k)).sum(dim=-1).mean()
                rkl = (p_k * (logp_k - log_p_tau_k)).sum(dim=-1).mean()

            mask_acc = reward == 1.0
            mask_pg = mask_acc | is_first_reject

            ce = mean_over_mask(nll, mask_acc)
            # REINFORCE: gradient should increase log-prob of rewarded actions.
            pg_term = nll * (reward - self.ema_b) * mask_pg.float()
            pg = mean_over_mask(pg_term, mask_pg)

            w = self.schedule.weights()
            loss = w["kd"] * kd + w["ce"] * ce + w["pg"] * pg + w["kl"] * rkl

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self._trainable, self.grad_clip)
            self.opt.step()

        self.ema_b = (1.0 - self.ema_m) * self.ema_b + self.ema_m * reward.mean().item()
        self.schedule.step += 1

        ms = (time.time() - t0) * 1000.0
        acc_ratio = mask_acc.float().mean().item()

        def _to_float(v: torch.Tensor | float) -> float:
            if isinstance(v, torch.Tensor):
                return float(v.detach().item())
            return float(v)

        return {
            "loss": float(loss.detach().item()),
            "kd": float(kd.detach().item()),
            "ce": _to_float(ce),
            "pg": _to_float(pg),
            "kl": float(rkl.detach().item()),
            "ms": float(ms),
            "acc_ratio": float(acc_ratio),
        }
