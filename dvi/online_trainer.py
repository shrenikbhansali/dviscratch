from __future__ import annotations

import time
import math
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
        desired_store_mode: str,
        store_warmup_steps: int,
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
        self.desired_store_mode = desired_store_mode
        self.store_warmup_steps = int(store_warmup_steps)
        self.topk = int(topk)
        self.vocab_size = int(vocab_size)
        self.grad_clip = float(grad_clip)
        self.ema_m = float(ema_m)
        self.ema_b = 0.0
        self._warned_zero_grads = False

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
        if self.desired_store_mode not in {"full", "topk"}:
            raise ValueError(f"Unknown desired_store_mode={self.desired_store_mode}")

    def step(self, batch: Dict[str, torch.Tensor | None], global_step: int) -> Dict[str, float]:
        t0 = time.time()

        hk = batch["hk"].to(self._device).detach()
        token = batch["token"].to(self._device)
        reward = batch["reward"].to(self._device)
        pos = batch["pos"].to(self._device)
        is_first_reject = batch["is_first_reject"].to(self._device)
        _ = pos

        z_phi = batch.get("z_phi")
        z_idx = batch.get("z_idx")
        z_val = batch.get("z_val")

        z_idx_eff = None
        z_val_eff = None

        if self.store_mode == "full":
            if z_phi is None or z_idx is not None or z_val is not None:
                raise ValueError("full mode requires z_phi and forbids z_idx/z_val")
            z_phi = z_phi.to(self._device)
        elif self.store_mode == "topk":
            if z_phi is not None or z_idx is None or z_val is None:
                raise ValueError("topk mode requires z_idx/z_val and forbids z_phi")
            z_idx = z_idx.to(self._device).long()
            z_val = z_val.to(self._device)
        else:
            raise ValueError(f"Unknown store_mode={self.store_mode}")

        effective_mode = self.desired_store_mode
        if (
            self.desired_store_mode == "topk"
            and self.store_mode == "full"
            and global_step < self.store_warmup_steps
        ):
            effective_mode = "full"
        if effective_mode == "full" and self.store_mode != "full":
            raise RuntimeError("Cannot compute full-mode KD from top-k buffer; disable store warmup or use full store.")

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

            teacher_topk_hit = 1.0
            if effective_mode == "full":
                log_p_tau = torch.log_softmax(z_phi.float() / self.tau, dim=-1)
                p_tau = log_p_tau.exp()
                kd = (p_tau * (log_p_tau - logp)).sum(dim=-1).mean()
                p = logp.exp()
                rkl = (p * (logp - log_p_tau)).sum(dim=-1).mean()
            else:
                if self.store_mode == "full":
                    k = min(self.topk, z_phi.shape[1])
                    z_val_eff, z_idx_eff = torch.topk(z_phi, k=k, dim=-1)
                else:
                    z_val_eff, z_idx_eff = z_val, z_idx
                gathered = draft_logits.gather(1, z_idx_eff)
                logp_k = torch.log_softmax(gathered.float(), dim=-1)
                p_k = logp_k.exp()
                log_p_tau_k = torch.log_softmax(z_val_eff.float() / self.tau, dim=-1)
                p_tau_k = log_p_tau_k.exp()
                kd = (p_tau_k * (log_p_tau_k - logp_k)).sum(dim=-1).mean()
                rkl = (p_k * (logp_k - log_p_tau_k)).sum(dim=-1).mean()
                teacher_in_topk = (z_idx_eff == token.unsqueeze(1)).any(dim=1).float()
                teacher_topk_hit = teacher_in_topk.mean().item()

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
            params_with_grad = 0
            grad_sq = 0.0
            for param in self._trainable:
                if param.grad is not None:
                    params_with_grad += 1
                    grad_sq += float(param.grad.detach().float().pow(2).sum().item())
            grad_norm_pre = math.sqrt(grad_sq) if grad_sq > 0 else 0.0
            if params_with_grad == 0 and not self._warned_zero_grads:
                print("[DVI][warn] no gradients flowed to drafter parameters.")
                self._warned_zero_grads = True
            grad_norm = clip_grad_norm_(self._trainable, self.grad_clip)
            self.opt.step()

        lr = float(self.opt.param_groups[0]["lr"])
        self.ema_b = (1.0 - self.ema_m) * self.ema_b + self.ema_m * reward.mean().item()
        self.schedule.step += 1

        ms = (time.time() - t0) * 1000.0
        acc_ratio = mask_acc.float().mean().item()
        over_budget_ms = max(0.0, ms - float(self.max_ms))

        if effective_mode == "full":
            teacher_top1 = z_phi.argmax(dim=-1)
        else:
            teacher_top1 = (z_idx_eff if self.store_mode == "full" else z_idx)[:, 0].long()
        student_top1 = draft_logits.argmax(dim=-1)
        argmax_agree = (teacher_top1 == student_top1).float().mean().item()

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
            "argmax_agree": float(argmax_agree),
            "grad_norm": float(grad_norm),
            "grad_norm_pre": float(grad_norm_pre),
            "grad_params": float(params_with_grad),
            "lr": lr,
            "over_budget_ms": float(over_budget_ms),
            "teacher_topk_hit": float(teacher_topk_hit),
            "w_kd": float(w["kd"]),
            "w_ce": float(w["ce"]),
            "w_pg": float(w["pg"]),
            "w_kl": float(w["kl"]),
        }
