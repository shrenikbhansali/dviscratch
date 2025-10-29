import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest
import torch

from dvi.buffer import DVIRingBuffer


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_topk_capacity_wraparound():
    device = _device()
    buffer = DVIRingBuffer(
        capacity=4,
        d_model=3,
        vocab_size=16,
        store="topk",
        topk=4,
        logits_dtype=torch.float16,
        device=device,
    )

    for i in range(6):
        hk = torch.full((3,), float(i + 1), dtype=torch.float16, device=device)
        z_phi = torch.linspace(0, 1, 16, dtype=torch.float32, device=device) + i
        buffer.push_topk(
            hk=hk,
            token=i + 1,
            z_phi=z_phi,
            reward=int((i + 1) % 2),
            pos=((i % 5) + 1),
            is_first_reject=(i % 3 == 0),
        )

    assert buffer.size == 4
    tokens = buffer._token.tolist()
    assert sorted(tokens) == [3, 4, 5, 6]


def test_sampling_shapes_and_dtypes():
    device = _device()

    full_buffer = DVIRingBuffer(
        capacity=6,
        d_model=4,
        vocab_size=10,
        store="full",
        logits_dtype=torch.float32,
        device=device,
    )
    for i in range(6):
        hk = torch.arange(4, dtype=torch.float16, device=device) + i
        z_phi = torch.arange(10, dtype=torch.float32, device=device) * 0.1 + i
        full_buffer.push_full(
            hk=hk,
            token=i,
            z_phi=z_phi,
            reward=1,
            pos=i % 10 + 1,
            is_first_reject=False,
        )

    batch_full = full_buffer.sample(3)
    assert batch_full["hk"].shape == (3, 4)
    assert batch_full["hk"].dtype == torch.float16
    assert batch_full["token"].shape == (3,)
    assert batch_full["token"].dtype == torch.int64
    assert batch_full["reward"].dtype == torch.float32
    assert batch_full["pos"].dtype == torch.int32
    assert batch_full["is_first_reject"].dtype == torch.bool
    assert batch_full["z_phi"].shape == (3, 10)
    assert batch_full["z_phi"].dtype == torch.float32
    assert batch_full["z_idx"] is None
    assert batch_full["z_val"] is None

    topk_buffer = DVIRingBuffer(
        capacity=5,
        d_model=4,
        vocab_size=12,
        store="topk",
        topk=3,
        logits_dtype=torch.float16,
        device=device,
    )
    for i in range(5):
        hk = torch.arange(4, dtype=torch.float16, device=device) + i
        z_phi = torch.linspace(-1, 1, 12, dtype=torch.float32, device=device) + i
        topk_buffer.push_topk(
            hk=hk,
            token=i,
            z_phi=z_phi,
            reward=i % 2,
            pos=i + 1,
            is_first_reject=(i == 2),
        )

    batch_topk = topk_buffer.sample(4)
    assert batch_topk["hk"].shape == (4, 4)
    assert batch_topk["hk"].dtype == torch.float16
    assert batch_topk["z_phi"] is None
    assert batch_topk["z_idx"].shape == (4, 3)
    assert batch_topk["z_idx"].dtype == torch.int32
    assert batch_topk["z_val"].shape == (4, 3)
    assert batch_topk["z_val"].dtype == torch.float16


def test_topk_sample_never_dense_logits():
    device = _device()
    buffer = DVIRingBuffer(
        capacity=4,
        d_model=2,
        vocab_size=8,
        store="topk",
        topk=2,
        logits_dtype=torch.float16,
        device=device,
    )

    for i in range(4):
        hk = torch.arange(2, dtype=torch.float16, device=device)
        z_phi = torch.arange(8, dtype=torch.float32, device=device) + i
        buffer.push_topk(
            hk=hk,
            token=i,
            z_phi=z_phi,
            reward=1,
            pos=i + 1,
            is_first_reject=False,
        )

    batch = buffer.sample(3)
    for key, tensor in batch.items():
        if tensor is None:
            continue
        assert tensor.device == torch.device(device)
        if tensor.ndim >= 2:
            assert tensor.shape[-1] != buffer._vocab_size


def test_validation_errors():
    device = _device()
    buffer_full = DVIRingBuffer(
        capacity=2,
        d_model=3,
        vocab_size=6,
        store="full",
        logits_dtype=torch.float16,
        device=device,
    )

    hk = torch.zeros(3, dtype=torch.float16, device=device)
    bad_z = torch.zeros(7, dtype=torch.float16, device=device)

    with pytest.raises(ValueError):
        buffer_full.push_full(
            hk=hk,
            token=0,
            z_phi=bad_z,
            reward=1,
            pos=1,
            is_first_reject=False,
        )

    good_z = torch.zeros(6, dtype=torch.float16, device=device)
    with pytest.raises(ValueError):
        buffer_full.push_full(
            hk=hk,
            token=0,
            z_phi=good_z,
            reward=2,
            pos=1,
            is_first_reject=False,
        )

    with pytest.raises(ValueError):
        buffer_full.push_full(
            hk=hk,
            token=0,
            z_phi=good_z,
            reward=1,
            pos=0,
            is_first_reject=False,
        )


def test_device_roundtrip():
    device = _device()
    buffer = DVIRingBuffer(
        capacity=3,
        d_model=2,
        vocab_size=5,
        store="topk",
        topk=2,
        logits_dtype=torch.float16,
        device=device,
    )

    hk = torch.ones(2, dtype=torch.float16, device=device)
    z_phi = torch.ones(5, dtype=torch.float32, device=device)
    buffer.push_topk(
        hk=hk,
        token=7,
        z_phi=z_phi,
        reward=1,
        pos=1,
        is_first_reject=False,
    )

    batch = buffer.sample(1)
    assert batch["hk"].device == torch.device(device)
    assert batch["token"].device == torch.device(device)
    assert batch["reward"].device == torch.device(device)
    assert batch["pos"].device == torch.device(device)
    assert batch["is_first_reject"].device == torch.device(device)
    assert batch["z_idx"].device == torch.device(device)
    assert batch["z_val"].device == torch.device(device)
