"""Utilities for streaming ShareGPT samples through a forward function."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from data.sharegpt_stream import iter_sharegpt_jsonl


@dataclass
class ShareGPTConfig:
    """Runtime knobs for ShareGPT streaming."""

    path: str
    max_samples: int = 0
    skip_samples: int = 0
    max_src_len: int = 2048
    max_tgt_len: int = 512
    keep_system: bool = False
    use_last_turn: bool = True
    joiner: str = "\n"


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter (or a CUDA device when available)."""

    param_device: Optional[torch.device] = None
    for param in model.parameters():
        param_device = param.device
        break

    if param_device is not None:
        return param_device

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_sync(device: torch.device) -> None:
    """Synchronize the CUDA stream when needed for accurate timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_completion(tokenizer, completion_ids) -> str:
    """Convert completion token ids into a normalized string."""

    if hasattr(completion_ids, "tolist"):
        completion_ids = completion_ids.tolist()
    text = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )
    return text.strip()


def _decode_token_preview(tokenizer, token_ids: List[int]) -> str:
    """Decode a short token list for debugging."""

    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    ).strip()


def _iter_sharegpt_samples(cfg: ShareGPTConfig) -> Iterator[Tuple[int, str, str]]:
    """Yield (sample_idx, prompt, reference_answer) triples according to *cfg*."""

    stream = iter_sharegpt_jsonl(
        cfg.path,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
        keep_system=cfg.keep_system,
        use_last_turn=cfg.use_last_turn,
        joiner=cfg.joiner,
    )

    skipped = 0
    emitted = 0
    for dataset_idx, (prompt, reference) in enumerate(stream):
        if skipped < cfg.skip_samples:
            skipped += 1
            continue
        yield dataset_idx, prompt, reference
        emitted += 1
        if cfg.max_samples and emitted >= cfg.max_samples:
            break


def stream_sharegpt_answers(
    *,
    model,
    tokenizer,
    forward_func,
    output_path: str,
    max_new_tokens: int,
    sharegpt_cfg: ShareGPTConfig,
    forward_kwargs: Optional[Dict[str, Any]] = None,
    block_dump_path: Optional[str] = None,
    block_dump_max_tokens: int = 32,
    sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run *forward_func* over ShareGPT samples and persist per-sample metrics.

    Returns a dictionary containing aggregate throughput statistics.
    """

    forward_kwargs = dict(forward_kwargs or {})
    device = _infer_model_device(model)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    errored = 0
    total_wall = 0.0
    total_tokens = 0

    sample_iter = _iter_sharegpt_samples(sharegpt_cfg)
    block_dump_handle = None
    if block_dump_path:
        dump_dir = os.path.dirname(block_dump_path) or "."
        os.makedirs(dump_dir, exist_ok=True)
        block_dump_handle = open(block_dump_path, "w", encoding="utf-8")

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample_idx, prompt, reference in sample_iter:
            if sample_callback:
                sample_callback(
                    {
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "reference": reference,
                    }
                )
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(device)
            input_len = inputs.input_ids.shape[1]

            _maybe_sync(device)
            start_time = time.time()
            try:
                forward_result = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **forward_kwargs,
                )
                if len(forward_result) == 5:
                    output_ids, new_token, idx, accept_lengths, block_traces = forward_result
                else:
                    output_ids, new_token, idx, accept_lengths = forward_result
                    block_traces = []
                error_msg = None
            except Exception as exc:  # pylint: disable=broad-except
                _maybe_sync(device)
                total_time = time.time() - start_time
                errored += 1
                record = {
                    "sample_id": sample_idx,
                    "prompt": prompt,
                    "reference": reference,
                    "error": repr(exc),
                    "wall_time": total_time,
                    "new_tokens": 0,
                    "output": "",
                    "idx": -1,
                    "accept_lengths": [],
                }
                fout.write(json.dumps(record) + "\n")
                print(f"[ShareGPT] sample {sample_idx} failed: {exc}")
                continue

            _maybe_sync(device)
            total_time = time.time() - start_time

            # Slice the newly generated portion.
            completion_ids = output_ids[0][input_len:]
            completion_text = _decode_completion(tokenizer, completion_ids)

            record = {
                "sample_id": sample_idx,
                "prompt": prompt,
                "reference": reference,
                "output": completion_text,
                "wall_time": total_time,
                "new_tokens": int(new_token),
                "idx": int(idx),
                "accept_lengths": list(map(int, accept_lengths)) if accept_lengths else [],
                "error": error_msg,
            }
            fout.write(json.dumps(record) + "\n")

            if block_dump_handle and block_traces:
                accept_list = record["accept_lengths"]
                if accept_list and len(accept_list) != len(block_traces):
                    print(
                        f"[ShareGPT][warn] accept_lengths({len(accept_list)}) "
                        f"!= block_traces({len(block_traces)}) for sample {sample_idx}"
                    )
                for block_idx, trace in enumerate(block_traces):
                    accepted = (
                        accept_list[block_idx]
                        if block_idx < len(accept_list)
                        else getattr(trace, "accepted_tokens", None)
                    )
                    draft_tokens = trace.tokens
                    verifier_tokens = getattr(trace, "verifier_tokens", [])
                    dump_record = {
                        "sample_id": sample_idx,
                        "block_index": block_idx,
                        "draft_len": len(draft_tokens),
                        "accepted": accepted,
                        "reject_reason": getattr(trace, "reject_reason", None),
                        "mismatch_index": getattr(trace, "mismatch_index", None),
                        "draft_tokens": draft_tokens[:block_dump_max_tokens],
                        "verifier_tokens": verifier_tokens[:block_dump_max_tokens],
                        "draft_preview": _decode_token_preview(
                            tokenizer, draft_tokens[:block_dump_max_tokens]
                        ),
                        "verifier_preview": _decode_token_preview(
                            tokenizer, verifier_tokens[:block_dump_max_tokens]
                        ),
                    }
                    block_dump_handle.write(json.dumps(dump_record) + "\n")

            processed += 1
            total_wall += total_time
            total_tokens += int(new_token)

            if progress_callback and processed:
                progress_callback(
                    {
                        "processed": processed,
                        "avg_wall_time": total_wall / processed,
                        "tokens_per_second": (total_tokens / total_wall) if total_wall > 0 else 0.0,
                        "errored": errored,
                    }
                )

            if processed % 10 == 0:
                tps = total_tokens / total_wall if total_wall > 0 else 0.0
                print(
                    f"[ShareGPT] processed={processed} avg_wall={total_wall / processed:.2f}s "
                    f"tokens/sec={tps:.2f}"
                )

    if block_dump_handle:
        block_dump_handle.close()

    return {
        "processed": processed,
        "errored": errored,
        "avg_wall_time": (total_wall / processed) if processed else 0.0,
        "tokens_per_second": (total_tokens / total_wall) if total_wall > 0 else 0.0,
        "total_tokens": total_tokens,
    }
