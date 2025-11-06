"""Generate answers with Kangaroo's speculative decoding pipeline."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer

from kangaroo.cli_utils import (
    add_dvi_args,
    add_model_and_adapter_args,
    normalize_adapter_path,
    normalize_model_flags,
    resolve_adapter_mode,
    str2bool,
)
from kangaroo.draft_trace import DraftBlockTrace
from kangaroo.kangaroo_model import KangarooModel


global_step = 0
blocks_processed = 0
tokens_accepted_total = 0
COUNTER_LOG_FREQUENCY = 50


def _make_default_dvi_runtime() -> Dict[str, Any]:
    return {
        "enabled": False,
        "buffer": None,
        "trainer": None,
        "store_mode": "topk",
        "batch_size": 0,
        "update_every": 1,
        "logits_dtype": torch.float16,
        "max_ms": 0.0,
    }


DVI_RUNTIME: Dict[str, Any] = _make_default_dvi_runtime()


def record_block_result(tokens_accepted: int) -> None:
    """Update driver counters after each verify/commit block."""
    global global_step, blocks_processed, tokens_accepted_total
    blocks_processed += 1
    tokens_accepted_total += max(0, tokens_accepted)
    global_step += 1
    if COUNTER_LOG_FREQUENCY and global_step % COUNTER_LOG_FREQUENCY == 0:
        print(
            f"[DVI] step={global_step} blocks={blocks_processed} "
            f"accepted={tokens_accepted_total}"
        )


def reset_counters() -> None:
    """Reset counters (used by tests)."""
    global global_step, blocks_processed, tokens_accepted_total
    global_step = 0
    blocks_processed = 0
    tokens_accepted_total = 0


def configure_dvi_runtime(args: argparse.Namespace, model: KangarooModel, device: torch.device) -> None:
    """Initialize (or disable) the DVI runtime structures based on *args*."""

    global DVI_RUNTIME

    if not getattr(args, "dvi_online", False):
        DVI_RUNTIME = _make_default_dvi_runtime()
        return

    from dvi.buffer import DVIRingBuffer
    from dvi.online_trainer import OnlineTrainer
    from dvi.schedule import PiecewiseSchedule

    if args.dvi_batch_size <= 0:
        raise ValueError("--dvi-batch-size must be positive when DVI is enabled")
    if args.dvi_update_every <= 0:
        raise ValueError("--dvi-update-every must be positive when DVI is enabled")

    logits_dtype = torch.float16 if args.dvi_logits_dtype == "float16" else torch.float32
    buffer = DVIRingBuffer(
        capacity=args.dvi_buffer_size,
        d_model=model.config.hidden_size,
        vocab_size=model.config.vocab_size,
        store=args.dvi_store,
        topk=args.dvi_topk,
        logits_dtype=logits_dtype,
        device=str(device),
    )

    schedule = PiecewiseSchedule(
        warmup=args.dvi_warmup_steps,
        kl0=args.dvi_kl_lambda0,
        klmin=args.dvi_kl_lambdamin,
        pgmax=args.dvi_pg_lambda_max,
    )

    trainer = OnlineTrainer(
        model,
        lr=5e-5,
        tau=args.dvi_tau,
        schedule=schedule,
        max_ms=args.max_online_train_ms,
        store_mode=args.dvi_store,
        topk=args.dvi_topk,
        vocab_size=model.config.vocab_size,
        weight_decay=0.01,
        grad_clip=1.0,
        ema_m=0.01,
        device=device,
    )

    DVI_RUNTIME = {
        "enabled": True,
        "buffer": buffer,
        "trainer": trainer,
        "store_mode": args.dvi_store,
        "batch_size": int(args.dvi_batch_size),
        "update_every": int(args.dvi_update_every),
        "logits_dtype": logits_dtype,
        "max_ms": float(args.max_online_train_ms),
    }


def _extract_block_logits(logits: torch.Tensor, k: int, vocab_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Return the verifier logits for the drafted block with shape [k, vocab]."""

    if k <= 0:
        raise ValueError("k must be positive when extracting verifier logits")

    if logits.dim() == 3:
        block = logits[:, -k:, :].reshape(-1, logits.shape[-1])
    elif logits.dim() == 2:
        block = logits[-k:, :]
    else:
        raise RuntimeError("Unexpected verifier logits rank")

    if block.shape[0] != k:
        raise RuntimeError("verifier logits length mismatch")
    if block.shape[1] != vocab_size:
        raise RuntimeError("verifier logits vocab size mismatch")

    block = block.detach()
    if block.dtype != dtype:
        block = block.to(dtype=dtype)
    return block


def _dvi_process_block(
    *,
    model: KangarooModel,
    draft_trace: DraftBlockTrace,
    logits: torch.Tensor,
    accepted_tokens: int,
    eos_token_id: Optional[int],
) -> None:
    """Push accepted/reject tuples for the current speculative block."""

    runtime = DVI_RUNTIME
    if not runtime.get("enabled", False):
        return

    buffer = runtime.get("buffer")
    if buffer is None:
        return

    k = len(draft_trace)
    if k == 0:
        return

    if accepted_tokens < 0:
        raise RuntimeError("accepted token count cannot be negative")
    if accepted_tokens > k:
        raise RuntimeError("accepted token count exceeds drafted length")

    logits_block = _extract_block_logits(
        logits=logits,
        k=k,
        vocab_size=model.config.vocab_size,
        dtype=runtime["logits_dtype"],
    )

    eos_accepted = (
        accepted_tokens > 0
        and eos_token_id is not None
        and draft_trace.tokens[accepted_tokens - 1] == eos_token_id
    )

    store_mode = runtime.get("store_mode", "topk")

    # Push accepted tuples.
    for pos in range(1, accepted_tokens + 1):
        hk_state = draft_trace.hk_state_of(pos)
        token_id = int(draft_trace.tokens[pos - 1])
        z_phi = logits_block[pos - 1]
        if store_mode == "full":
            buffer.push_full(
                hk=hk_state,
                token=token_id,
                z_phi=z_phi,
                reward=1,
                pos=pos,
                is_first_reject=False,
            )
        else:
            buffer.push_topk(
                hk=hk_state,
                token=token_id,
                z_phi=z_phi,
                reward=1,
                pos=pos,
                is_first_reject=False,
            )

    # Push first reject if applicable.
    if accepted_tokens < k and not eos_accepted:
        reject_pos = accepted_tokens + 1
        hk_state = draft_trace.hk_state_of(reject_pos)
        token_id = int(draft_trace.tokens[reject_pos - 1])
        z_phi = logits_block[reject_pos - 1]
        if store_mode == "full":
            buffer.push_full(
                hk=hk_state,
                token=token_id,
                z_phi=z_phi,
                reward=0,
                pos=reject_pos,
                is_first_reject=True,
            )
        else:
            buffer.push_topk(
                hk=hk_state,
                token=token_id,
                z_phi=z_phi,
                reward=0,
                pos=reject_pos,
                is_first_reject=True,
            )


def _dvi_maybe_step() -> None:
    """Run a trainer step on cadence when enough tuples are buffered."""

    runtime = DVI_RUNTIME
    if not runtime.get("enabled", False):
        return

    buffer = runtime.get("buffer")
    trainer = runtime.get("trainer")
    if buffer is None or trainer is None:
        return

    batch_size = int(runtime.get("batch_size", 0))
    if batch_size <= 0 or buffer.size < batch_size:
        return

    update_every = int(runtime.get("update_every", 1))
    if update_every <= 0:
        update_every = 1

    if global_step % update_every != 0:
        return

    batch = buffer.sample(batch_size)
    stats = trainer.step(batch, global_step)
    fill_ratio = buffer.size / buffer.capacity
    print(
        "[DVI] step="
        f"{global_step} loss={stats['loss']:.3f} kd={stats['kd']:.3f} "
        f"ce={stats['ce']:.3f} pg={stats['pg']:.3f} kl={stats['kl']:.3f} "
        f"ms={stats['ms']:.2f} acc_ratio={stats['acc_ratio']:.3f} "
        f"buf={buffer.size}/{buffer.capacity} fill={fill_ratio:.2f}"
    )
    if stats["ms"] > runtime.get("max_ms", 0.0):
        print(
            "[DVI][warn] trainer step "
            f"{stats['ms']:.2f}ms > budget {runtime.get('max_ms', 0.0):.2f}ms"
        )


def _draft_block(
    model: KangarooModel,
    global_tokens: torch.Tensor,
    global_position_ids: torch.Tensor,
    start_index: int,
    hidden_state: torch.Tensor,
    previous_exited_hidden_states: Optional[torch.Tensor],
    adapter_past_key_values,
    steps: int,
    threshold: float,
) -> Tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    DraftBlockTrace,
]:
    """Run the shallow drafter for up to ``steps`` tokens and capture hk states."""

    scratch = DraftBlockTrace.empty(
        d_model=hidden_state.shape[-1],
        max_len=max(0, steps),
        device=hidden_state.device,
    )

    carry_hidden_states = previous_exited_hidden_states
    exited_hidden_states: Optional[torch.Tensor] = None
    adapter_cache = adapter_past_key_values
    adapter_hidden_state = hidden_state
    end_index = start_index + 1
    predict_score = float("inf")

    for step in range(1 + steps):
        in_tokens_small = global_tokens[:, end_index - 1 : end_index]
        cache_length = adapter_cache[0][0].shape[2]
        assert cache_length <= end_index - 1, f"{adapter_cache[0][0].shape} - {end_index - 1}"
        if cache_length < end_index - 1:
            position_ids = global_position_ids[:, start_index - 1 : end_index]
            source_hidden_states = (
                carry_hidden_states
                if carry_hidden_states is not None
                else exited_hidden_states
            )
            hidden_state_early_last = (
                source_hidden_states[:, -1:, :]
                if source_hidden_states is not None
                else None
            )
        else:
            position_ids = global_position_ids[:, end_index - 1 : end_index]
            hidden_state_early_last = None

        hidden_state_early = model.base_model.forward_draft_or_large_model(
            in_tokens_small=in_tokens_small[:, -1:], position_ids=position_ids[:, -1:]
        )

        exited_hidden_states = (
            hidden_state_early
            if exited_hidden_states is None
            else torch.cat([exited_hidden_states, hidden_state_early], dim=1)
        )

        if hidden_state_early_last is not None:
            hidden_state_early = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

        if step == 0:
            carry_hidden_states = None

        if step == steps or (step > 0 and predict_score < threshold):
            break

        adapter_hidden_state, adapter_cache = model.adapter_model.forward_early_stop(
            inputs_embeds=hidden_state_early,
            position_ids=position_ids,
            past_key_values=adapter_cache,
            use_cache=True,
        )

        hk_slice = adapter_hidden_state[:, -1:, :]
        if hasattr(model, "drafter_head"):
            predict_logits = model.drafter_logits_from_hk(hk_slice).float()
        else:
            predict_logits = model.head_model(hk_slice).float()

        next_token_tensor = torch.argmax(predict_logits[:, -1, :], dim=-1)
        if next_token_tensor.numel() != 1:
            raise RuntimeError("Draft tracing currently supports batch_size=1.")
        next_token_id = int(next_token_tensor.item())

        scratch.append(hk_slice, next_token_id)
        global_tokens[:, end_index] = next_token_tensor

        end_index += 1
        predict_score = predict_logits.softmax(dim=-1).max().item()

    draft_trace = scratch.finalize()
    return exited_hidden_states, end_index, adapter_hidden_state, adapter_cache, draft_trace


def kangaroo_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    do_sample: bool = False,
    max_length: int = 2048,
    EARLY_STOP_LAYER: int = 2,
    SPECULATIVE_DECODING_STEPS: int = 6,
    threshold: float = 0.6,
    dvi_debug_hk: bool = False,
):
    context_tokens = inputs.input_ids
    device = context_tokens.device
    token_eos = tokenizer.eos_token_id
    batch_size, context_length = context_tokens.shape
    global_tokens = torch.ones((batch_size, max_length), dtype=torch.long, device=device) * token_eos
    global_position_ids = torch.LongTensor([[i for i in range(max_length)]]).to(device)
    accept_length_list = [1]

    start_index = context_length
    global_tokens[:, :start_index] = context_tokens

    # Init KV-cache and sample the first token
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index]
        output = model.base_model(
            context_tokens, position_ids=position_ids, output_hidden_states=True
        )
        model.base_model.past_key_values = list(output.past_key_values)
        hidden_state = output.hidden_states[-1]
        logits = output.logits  # batchsize, input_length, vocab_size
        global_tokens[:, start_index] = torch.argmax(logits[:, -1, :], dim=-1).item()
        hidden_state_early = output.hidden_states[EARLY_STOP_LAYER]

        # KV-cache for the adapter
        hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(
            inputs_embeds=hidden_state_early[:, :, :],
            position_ids=global_position_ids[:, :context_length],
            use_cache=True,
        )

    total_inference_steps = 0
    block_traces: List[DraftBlockTrace] = []
    previous_exited_hidden_states: Optional[torch.Tensor] = None

    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False

        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:
            start_index_copy = start_index

            (
                exited_hidden_states,
                end_index,
                _adapter_hidden_state,
                adapter_past_key_values,
                draft_trace,
            ) = _draft_block(
                model=model,
                global_tokens=global_tokens,
                global_position_ids=global_position_ids,
                start_index=start_index,
                hidden_state=hidden_state,
                previous_exited_hidden_states=previous_exited_hidden_states,
                adapter_past_key_values=adapter_past_key_values,
                steps=SPECULATIVE_DECODING_STEPS,
                threshold=threshold,
            )

            # STEP2: Big model inference
            position_ids = global_position_ids[:, start_index:end_index]
            assert (
                model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape[2] == start_index
            ), "{} - {}".format(model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape, start_index)
            assert exited_hidden_states.shape[1] == position_ids.shape[1]
            hidden_state_, hidden_state = model.base_model.forward_draft_or_large_model(
                in_features_large=exited_hidden_states, position_ids=position_ids
            )

            logits = model.head_model(hidden_state).float()  # batchsize, input_length, vocab_size
            output_tokens = torch.argmax(logits[:, :, :], dim=-1)

            # Verification for greedy decoding
            output_lenght = end_index - start_index
            for i in range(output_lenght):
                if (
                    i == output_lenght - 1
                    or output_tokens[0, i] == token_eos
                    or output_tokens[0, i] != global_tokens[0, start_index + 1 + i]
                ):
                    global_tokens[0, start_index + 1 + i] = output_tokens[0, i]
                    start_index = start_index + 1 + i
                    if output_tokens[0, i] == token_eos:
                        stop = True
                    break

            accepted_tokens = start_index - start_index_copy
            if DVI_RUNTIME.get("enabled", False):
                _dvi_process_block(
                    model=model,
                    draft_trace=draft_trace,
                    logits=logits,
                    accepted_tokens=accepted_tokens,
                    eos_token_id=token_eos,
                )
            record_block_result(accepted_tokens)
            accept_length_list.append(accepted_tokens)
            if dvi_debug_hk:
                print(
                    f"[DVI] hk captured: L={len(draft_trace)}, "
                    f"shape={draft_trace.hk_fp16.shape}, dtype={draft_trace.hk_fp16.dtype}"
                )
            block_traces.append(draft_trace)
            previous_exited_hidden_states = exited_hidden_states
            hidden_state = hidden_state[:, : output_lenght - (end_index - start_index), :]

            # STEP 4: Post process KV-cache
            if model.base_model.past_key_values[0][0].shape[2] > start_index:
                past_key_values_large_ = []
                for k, v in model.base_model.past_key_values:
                    past_key_values_large_.append((k[:, :, :start_index, :], v[:, :, :start_index, :]))
                model.base_model.past_key_values = past_key_values_large_

            if adapter_past_key_values[0][0].shape[2] > start_index:
                adapter_past_key_values_ = []
                for k, v in adapter_past_key_values:
                    adapter_past_key_values_.append((k[:, :, :start_index, :], v[:, :, :start_index, :]))
                adapter_past_key_values = tuple(adapter_past_key_values_)
                del adapter_past_key_values_

            total_inference_steps += 1

            if DVI_RUNTIME.get("enabled", False):
                _dvi_maybe_step()

            if stop:
                break

    output_ids = global_tokens[0, : start_index + 1].tolist()
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1
    return [output_ids], new_token, idx, accept_length_list, block_traces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kangaroo speculative decoding")
    add_model_and_adapter_args(parser)
    add_dvi_args(parser)

    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--exitlayer",
        type=int,
        default=2,
        help="The shallow exit layer used for drafting.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Number of speculative decoding steps per block.",
    )
    parser.add_argument(
        "--dvi-lora-rank",
        type=int,
        default=8,
        help="LoRA rank used when attaching the drafter head.",
    )
    parser.add_argument(
        "--dvi-lora-alpha",
        type=float,
        default=None,
        help="LoRA scaling alpha; defaults to the configured rank when omitted.",
    )
    parser.add_argument(
        "--dvi-attach-drafter",
        type=str2bool,
        default=None,
        help="Whether to attach the drafter LoRA head (defaults to true when DVI is active).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.model_id = normalize_model_flags(args)
    except argparse.ArgumentError as err:
        parser.error(str(err))

    adapter_path = normalize_adapter_path(args.adapter_path)
    adapter_mode = resolve_adapter_mode(args)
    if adapter_mode == "load" and adapter_path is None:
        parser.error("adapter-mode=load requires --adapter-path")

    args.adapter_mode = adapter_mode
    args.adapter_path = adapter_path
    args.model_path = args.model_id

    if args.dvi_lora_alpha is None:
        args.dvi_lora_alpha = float(args.dvi_lora_rank)

    attach_default = bool(args.dvi_online) or bool(getattr(args, "load_lora", ""))
    if args.dvi_attach_drafter is None:
        args.dvi_attach_drafter = attach_default

    return args


# ---------------------------
# Tests
# ---------------------------


def _make_dummy_config():
    from transformers.models.llama.configuration_llama import LlamaConfig

    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        max_position_embeddings=32,
        pad_token_id=0,
    )


def test_normalize_model_flags_accepts_matching():
    args = argparse.Namespace(model_id="foo", model_path="foo")
    assert normalize_model_flags(args) == "foo"


def test_normalize_model_flags_mismatch():
    import pytest

    args = argparse.Namespace(model_id="foo", model_path="bar")
    with pytest.raises(argparse.ArgumentError):
        normalize_model_flags(args)


def test_identity_adapter_no_io(monkeypatch):
    import copy
    import torch

    import kangaroo.kangaroo_model as km

    base_config = _make_dummy_config()
    exit_layer = 2

    monkeypatch.setattr(
        km.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: copy.deepcopy(base_config)),
    )

    def fake_from_pretrained(model_id, torch_dtype=None, device_map=None, EARLY_STOP_LAYER=None):
        cfg = copy.deepcopy(base_config)
        model = km.EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=EARLY_STOP_LAYER)
        return model.eval()

    monkeypatch.setattr(km.EarlyExitLlamaForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))

    load_calls = []

    def forbid_load(*args, **kwargs):
        load_calls.append(args)
        raise AssertionError("torch.load should not be invoked in identity mode")

    monkeypatch.setattr(km.torch, "load", forbid_load)

    model = km.KangarooModel(
        model_id="dummy",
        adapter_mode="none",
        adapter_path=None,
        exit_layer=exit_layer,
        dtype="float32",
    )

    assert not load_calls

    input_ids = torch.randint(0, base_config.vocab_size, (1, 3))
    outputs = model.base_model(input_ids, output_hidden_states=True)
    hidden_early = outputs.hidden_states[exit_layer]
    adapter_hidden, cache = model.adapter_model.forward_early_stop(
        inputs_embeds=hidden_early,
        position_ids=torch.arange(hidden_early.shape[1]).unsqueeze(0),
        use_cache=True,
    )

    assert adapter_hidden.shape == hidden_early.shape
    assert len(cache) == exit_layer


def test_identity_adapter_metadata(monkeypatch):
    import copy

    import kangaroo.kangaroo_model as km

    base_config = _make_dummy_config()
    exit_layer = 3

    monkeypatch.setattr(
        km.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: copy.deepcopy(base_config)),
    )

    def fake_from_pretrained(model_id, torch_dtype=None, device_map=None, EARLY_STOP_LAYER=None):
        cfg = copy.deepcopy(base_config)
        model = km.EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=EARLY_STOP_LAYER)
        return model.eval()

    monkeypatch.setattr(km.EarlyExitLlamaForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))

    model = km.KangarooModel(
        model_id="dummy",
        adapter_mode="none",
        adapter_path=None,
        exit_layer=exit_layer,
        dtype="float32",
    )

    assert model.adapter_cfg.num_hidden_layers == exit_layer
    assert len(model.adapter_model.layers) == exit_layer
    assert model.adapter_cfg.hidden_size == base_config.hidden_size
    assert model.adapter_cfg.vocab_size == base_config.vocab_size
    assert all(not param.requires_grad for param in model.adapter_model.parameters())


def test_counters_increment():
    reset_counters()
    record_block_result(3)
    record_block_result(0)

    assert global_step == 2
    assert blocks_processed == 2
    assert tokens_accepted_total == 3


def test_draft_block_trace_roundtrip():
    device = torch.device("cpu")
    scratch = DraftBlockTrace.empty(d_model=4, max_len=3, device=device)
    vec0 = torch.arange(4, dtype=torch.float32, device=device)
    vec1 = torch.arange(4, dtype=torch.float16, device=device) + 1

    scratch.append(vec0, 11)
    scratch.append(vec1, 13)
    trace = scratch.finalize()

    assert len(trace) == 2
    assert trace.tokens == [11, 13]
    assert trace.hk_fp16.dtype == torch.float16
    torch.testing.assert_close(trace.hk_state_of(1).float(), vec0.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(trace.hk_state_of(2).float(), vec1.float(), atol=1e-3, rtol=1e-3)

    try:
        trace.hk_state_of(3)
        raise AssertionError("hk_state_of should raise when index is out of range")
    except IndexError:
        pass


if __name__ == "__main__":
    from evaluation.eval import reorg_answer_file, run_eval

    args = parse_args()

    question_file = "data/question.jsonl"

    model = KangarooModel(
        model_id=args.model_id,
        adapter_mode=args.adapter_mode,
        adapter_path=args.adapter_path,
        exit_layer=args.exitlayer,
        dtype=args.dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    do_sample = False

    if args.dvi_attach_drafter or args.dvi_online or bool(args.load_lora):
        model.attach_drafter_head(r=args.dvi_lora_rank, alpha=args.dvi_lora_alpha)

    if args.dvi_online:
        try:
            ref_param = next(model.head_model.parameters())
        except StopIteration as exc:
            raise RuntimeError(
                "model.head_model exposes no parameters for device inference"
            ) from exc
        dvi_device = ref_param.device
    else:
        dvi_device = torch.device("cpu")
    configure_dvi_runtime(args, model, dvi_device)

    assert not args.answer_file
    os.makedirs(f"data/{args.bench_name}/{args.model_id}", exist_ok=True)

    for run in range(3):
        answer_file = f"data/{args.bench_name}/{args.model_id}/{run}.jsonl"
        print(f"Output to {answer_file}")

        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=kangaroo_forward,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            do_sample=do_sample,
            threshold=args.threshold,
            SPECULATIVE_DECODING_STEPS=args.steps,
            EARLY_STOP_LAYER=args.exitlayer,
            dvi_debug_hk=args.dvi_debug_hk,
        )

        reorg_answer_file(answer_file)
