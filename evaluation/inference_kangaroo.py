"""Generate answers with Kangaroo's speculative decoding pipeline."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

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
)
from kangaroo.kangaroo_model import KangarooModel


global_step = 0
blocks_processed = 0
tokens_accepted_total = 0
COUNTER_LOG_FREQUENCY = 50


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

    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False

        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:
            start_index_copy = start_index
            end_index = start_index + 1

            # STEP 1: Small model decoding
            for step in range(1 + SPECULATIVE_DECODING_STEPS):
                assert (
                    adapter_past_key_values[0][0].shape[2] <= end_index - 1
                ), "{} - {}".format(adapter_past_key_values[0][0].shape, end_index - 1)
                in_tokens_small = global_tokens[:, end_index - 1 : end_index]
                if adapter_past_key_values[0][0].shape[2] < end_index - 1:
                    # Once all drafted tokens are accepted, the KV-cache of the last draft token for
                    # the adapter is missing (see Kangaroo paper).
                    position_ids = global_position_ids[:, start_index - 1 : end_index]
                    hidden_state_early_last = exited_hidden_states[:, -1:, :]
                else:
                    position_ids = global_position_ids[:, end_index - 1 : end_index]
                    hidden_state_early_last = None

                hidden_state_early = model.base_model.forward_draft_or_large_model(
                    in_tokens_small=in_tokens_small[:, -1:], position_ids=position_ids[:, -1:]
                )

                if step == 0:
                    exited_hidden_states = None

                exited_hidden_states = (
                    hidden_state_early
                    if exited_hidden_states is None
                    else torch.cat([exited_hidden_states, hidden_state_early], dim=1)
                )

                if hidden_state_early_last is not None:
                    hidden_state_early = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

                # early exiting
                if step == SPECULATIVE_DECODING_STEPS or (step > 0 and predict_score < threshold):
                    break

                hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(
                    inputs_embeds=hidden_state_early,
                    position_ids=position_ids,
                    past_key_values=adapter_past_key_values,
                    use_cache=True,
                )

                predict_logits = model.head_model(hidden_state[:, -1:, :]).float()
                global_tokens[:, end_index] = torch.argmax(predict_logits[:, -1, :], dim=-1)

                end_index += 1
                predict_score = predict_logits.softmax(dim=-1).max().item()

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
            record_block_result(accepted_tokens)
            accept_length_list.append(accepted_tokens)
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

            if stop:
                break

    output_ids = global_tokens[0, : start_index + 1].tolist()
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1
    return [output_ids], new_token, idx, accept_length_list


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
        )

        reorg_answer_file(answer_file)
