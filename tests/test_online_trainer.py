from __future__ import annotations

import math
import os
import sys
from typing import Dict

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dvi.online_trainer import OnlineTrainer
from dvi.schedule import PiecewiseSchedule


class DummyModel(torch.nn.Module):
    def __init__(self, d_model: int = 4, vocab_size: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight.requires_grad_(False)
        self.lora = torch.nn.Parameter(torch.zeros(vocab_size, d_model))

    def drafter_logits_from_hk(self, hk: torch.Tensor) -> torch.Tensor:
        weight = self.proj.weight + self.lora
        return hk @ weight.t()

    def dvi_trainable_params(self):
        return [self.lora]


class BadModel(DummyModel):
    def __init__(self, d_model: int = 4, vocab_size: int = 5) -> None:
        super().__init__(d_model, vocab_size)
        self.extra = torch.nn.Parameter(torch.ones(1))

    def dvi_trainable_params(self):
        return [self.lora]


def make_schedule() -> PiecewiseSchedule:
    return PiecewiseSchedule(warmup=2, kl0=1.0, klmin=0.1, pgmax=0.2)


def make_batch(
    *,
    B: int,
    d_model: int,
    vocab: int,
    mode: str,
    topk: int | None = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor | None]:
    g = torch.Generator().manual_seed(seed)
    hk = torch.randn(B, d_model, generator=g)
    token = torch.randint(0, vocab, (B,), generator=g)
    reward = torch.zeros(B, dtype=torch.float32)
    pos = torch.arange(1, B + 1, dtype=torch.int32)
    is_first_reject = torch.zeros(B, dtype=torch.bool)
    z_phi = torch.randn(B, vocab, generator=g)

    batch: Dict[str, torch.Tensor | None] = {
        "hk": hk,
        "token": token,
        "reward": reward,
        "pos": pos,
        "is_first_reject": is_first_reject,
        "z_phi": None,
        "z_idx": None,
        "z_val": None,
    }
    if mode == "full":
        batch["z_phi"] = z_phi
    elif mode == "topk":
        assert topk is not None
        z_idx = torch.arange(vocab).unsqueeze(0).expand(B, -1)
        batch["z_idx"] = z_idx[:, :topk]
        batch["z_val"] = z_phi[:, :topk]
    else:
        raise ValueError(mode)
    return batch


def make_rewarded_batch(batch: Dict[str, torch.Tensor | None]) -> Dict[str, torch.Tensor | None]:
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    reward = torch.tensor([1.0, 0.0], dtype=torch.float32)
    pos = torch.tensor([1, 2], dtype=torch.int32)
    is_first_reject = torch.tensor([False, True])
    out["reward"] = reward
    out["pos"] = pos
    out["is_first_reject"] = is_first_reject
    return out


def test_store_mode_validation_full() -> None:
    model = DummyModel()
    trainer = OnlineTrainer(
        model,
        lr=1e-3,
        tau=1.5,
        schedule=make_schedule(),
        max_ms=10,
        store_mode="full",
        desired_store_mode="full",
        store_warmup_steps=0,
        topk=4,
        vocab_size=model.vocab_size,
        device=torch.device("cpu"),
    )
    batch = make_batch(B=2, d_model=4, vocab=model.vocab_size, mode="full")
    bad = dict(batch)
    bad["z_phi"] = None
    with pytest.raises(ValueError, match="full mode requires z_phi and forbids z_idx/z_val"):
        trainer.step(bad, global_step=0)


def test_store_mode_validation_topk() -> None:
    model = DummyModel()
    trainer = OnlineTrainer(
        model,
        lr=1e-3,
        tau=1.5,
        schedule=make_schedule(),
        max_ms=10,
        store_mode="topk",
        desired_store_mode="topk",
        store_warmup_steps=0,
        topk=2,
        vocab_size=model.vocab_size,
        device=torch.device("cpu"),
    )
    batch = make_batch(B=2, d_model=4, vocab=model.vocab_size, mode="topk", topk=2)
    bad = dict(batch)
    bad["z_phi"] = torch.randn(2, model.vocab_size)
    with pytest.raises(ValueError, match="topk mode requires z_idx/z_val and forbids z_phi"):
        trainer.step(bad, global_step=0)


def test_topk_matches_full_when_k_equals_vocab() -> None:
    model_full = DummyModel()
    model_topk = DummyModel()
    model_topk.load_state_dict(model_full.state_dict())

    schedule_full = make_schedule()
    schedule_topk = make_schedule()

    trainer_full = OnlineTrainer(
        model_full,
        lr=1e-3,
        tau=1.5,
        schedule=schedule_full,
        max_ms=10,
        store_mode="full",
        desired_store_mode="full",
        store_warmup_steps=0,
        topk=model_full.vocab_size,
        vocab_size=model_full.vocab_size,
        device=torch.device("cpu"),
    )
    trainer_topk = OnlineTrainer(
        model_topk,
        lr=1e-3,
        tau=1.5,
        schedule=schedule_topk,
        max_ms=10,
        store_mode="topk",
        desired_store_mode="topk",
        store_warmup_steps=0,
        topk=model_topk.vocab_size,
        vocab_size=model_topk.vocab_size,
        device=torch.device("cpu"),
    )

    base = make_batch(B=2, d_model=4, vocab=model_full.vocab_size, mode="full", seed=42)
    batch_full = make_rewarded_batch(base)
    batch_full["z_idx"] = None
    batch_full["z_val"] = None

    batch_topk = make_rewarded_batch(base)
    vocab = model_topk.vocab_size
    z_idx = torch.arange(vocab).unsqueeze(0).expand(2, -1)
    batch_topk["z_phi"] = None
    batch_topk["z_idx"] = z_idx
    batch_topk["z_val"] = base["z_phi"].clone()

    metrics_full = trainer_full.step(batch_full, global_step=0)
    metrics_topk = trainer_topk.step(batch_topk, global_step=0)

    assert math.isclose(metrics_full["kd"], metrics_topk["kd"], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(metrics_full["kl"], metrics_topk["kl"], rel_tol=0.0, abs_tol=1e-6)


def test_mask_edge_cases_produce_zero_losses() -> None:
    model = DummyModel()
    trainer = OnlineTrainer(
        model,
        lr=1e-3,
        tau=1.5,
        schedule=make_schedule(),
        max_ms=10,
        store_mode="full",
        desired_store_mode="full",
        store_warmup_steps=0,
        topk=model.vocab_size,
        vocab_size=model.vocab_size,
        device=torch.device("cpu"),
    )
    batch = make_batch(B=2, d_model=4, vocab=model.vocab_size, mode="full", seed=123)
    metrics = trainer.step(batch, global_step=0)
    assert metrics["ce"] == pytest.approx(0.0)
    assert metrics["pg"] == pytest.approx(0.0)


def test_schedule_step_advances_after_step() -> None:
    model = DummyModel()
    schedule = PiecewiseSchedule(warmup=1, kl0=1.0, klmin=0.1, pgmax=0.2)
    trainer = OnlineTrainer(
        model,
        lr=1e-3,
        tau=1.5,
        schedule=schedule,
        max_ms=10,
        store_mode="full",
        desired_store_mode="full",
        store_warmup_steps=0,
        topk=model.vocab_size,
        vocab_size=model.vocab_size,
        device=torch.device("cpu"),
    )
    batch = make_rewarded_batch(make_batch(B=2, d_model=4, vocab=model.vocab_size, mode="full", seed=99))
    trainer.step(batch, global_step=0)
    assert schedule.step == 1
    trainer.step(batch, global_step=1)
    assert schedule.step == 2
    weights = schedule.weights()
    assert math.isclose(weights["pg"], schedule.pgmax, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(weights["kd"], 0.5, rel_tol=0.0, abs_tol=1e-6)


def test_time_reporting_positive() -> None:
    model = DummyModel()
    trainer = OnlineTrainer(
        model,
        lr=1e-3,
        tau=1.5,
        schedule=make_schedule(),
        max_ms=10,
        store_mode="full",
        desired_store_mode="full",
        store_warmup_steps=0,
        topk=model.vocab_size,
        vocab_size=model.vocab_size,
        device=torch.device("cpu"),
    )
    batch = make_rewarded_batch(make_batch(B=2, d_model=4, vocab=model.vocab_size, mode="full", seed=17))
    metrics = trainer.step(batch, global_step=0)
    assert metrics["ms"] > 0.0
    assert metrics["over_budget_ms"] >= 0.0
    assert "argmax_agree" in metrics
    assert "grad_norm" in metrics
    assert "lr" in metrics


def test_topk_bounds_validation() -> None:
    model = DummyModel()
    with pytest.raises(ValueError):
        OnlineTrainer(
            model,
            lr=1e-3,
            tau=1.5,
            schedule=make_schedule(),
            max_ms=10,
            store_mode="topk",
            desired_store_mode="topk",
            store_warmup_steps=0,
            topk=model.vocab_size + 1,
            vocab_size=model.vocab_size,
            device=torch.device("cpu"),
        )


def test_frozen_parameter_audit_raises_for_requires_grad() -> None:
    model = BadModel()
    with pytest.raises(RuntimeError, match="Non-LoRA parameters have requires_grad=True"):
        OnlineTrainer(
            model,
            lr=1e-3,
            tau=1.5,
            schedule=make_schedule(),
            max_ms=10,
            store_mode="full",
            desired_store_mode="full",
            store_warmup_steps=0,
            topk=model.vocab_size,
            vocab_size=model.vocab_size,
            device=torch.device("cpu"),
        )


def test_schedule_weights_progression() -> None:
    sched = PiecewiseSchedule(warmup=4, kl0=1.0, klmin=0.2, pgmax=0.5)
    w0 = sched.weights()
    assert w0["kd"] == pytest.approx(1.0)
    assert w0["pg"] == pytest.approx(0.0)
    assert w0["kl"] == pytest.approx(1.0)

    sched.step = 4
    w1 = sched.weights()
    assert w1["kd"] == pytest.approx(0.5)
    assert w1["pg"] == pytest.approx(0.5)
    assert w1["kl"] == pytest.approx(0.2)
