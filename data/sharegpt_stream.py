"""Streaming utilities for ShareGPT-formatted conversations."""

from __future__ import annotations

import io
import json
from typing import Iterator, List, Optional, Sequence, Tuple

RolePair = Tuple[str, str]

__all__ = ["iter_sharegpt_jsonl"]


def _norm_text(value: Optional[str]) -> str:
    """Normalize text blocks by coercing to string and normalizing newlines."""
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n")


def _normalize_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    role = str(role).lower()
    if role in {"assistant", "model"}:
        return "gpt"
    if role in {"user"}:
        return "human"
    if role in {"system", "human", "gpt"}:
        return role
    return None


def _format_blocks(blocks: Sequence[str], joiner: str = "\n") -> str:
    text = joiner.join(blocks)
    return text.strip()


def _has_content(seq: Sequence[RolePair]) -> bool:
    return any(text.strip() for _, text in seq)


def iter_sharegpt_jsonl(
    path: str,
    *,
    max_src_len: int = 2048,
    max_tgt_len: int = 256,
    joiner: str = "\n",
    keep_system: bool = False,
    use_last_turn: bool = True,
) -> Iterator[Tuple[str, str]]:
    """Yield (prompt, target) pairs from a ShareGPT-style JSONL file.

    Parameters mirror the project specification for Step 2. The loader reads
    the source file line-by-line without ever materializing the entire dataset
    in memory.
    """

    with io.open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            convs = record.get("conversations") or []
            if not isinstance(convs, list):
                continue

            system_blocks: List[str] = []
            sequence: List[RolePair] = []
            for turn in convs:
                if not isinstance(turn, dict):
                    continue
                role = _normalize_role(turn.get("from"))
                value = turn.get("value")
                if value is None:
                    value = turn.get("text") or turn.get("markdown")
                text = _norm_text(value)
                if role == "system":
                    if keep_system and text:
                        system_blocks.append(text)
                    continue
                if role in {"human", "gpt"}:
                    sequence.append((role, text))

            if not sequence or not _has_content(sequence):
                continue

            if use_last_turn and sequence[-1][0] != "gpt":
                # Final human message without assistant reply cannot form a pair.
                continue

            last_assistant_index = -1
            if sequence:
                for idx in range(len(sequence) - 1, -1, -1):
                    if sequence[idx][0] == "gpt":
                        last_assistant_index = idx
                        break
            if last_assistant_index == -1:
                continue

            def _build_prompt(prefix: Sequence[RolePair]) -> str:
                blocks: List[str] = []
                if keep_system and system_blocks:
                    system_text = _format_blocks(system_blocks, joiner)
                    if system_text:
                        blocks.append("[SYSTEM]\n" + system_text)
                for role, text in prefix:
                    if role == "human":
                        blocks.append(f"USER: {text}")
                    else:
                        blocks.append(f"ASSISTANT: {text}")
                blocks.append("ASSISTANT:")
                prompt = _format_blocks(blocks, joiner)
                if max_src_len > 0:
                    prompt = prompt[-max_src_len:]
                return prompt

            def _truncate_target(text: str) -> str:
                truncated = text[:max_tgt_len] if max_tgt_len > 0 else text
                return truncated

            def _emit(prefix: Sequence[RolePair], target_text: str) -> Optional[Tuple[str, str]]:
                if not any(role == "human" and text.strip() for role, text in prefix):
                    return None
                target_norm = _norm_text(target_text)
                target_trimmed = _truncate_target(target_norm)
                if not target_trimmed.strip():
                    return None
                prompt_text = _build_prompt(prefix)
                if not prompt_text.strip():
                    return None
                return prompt_text, target_trimmed

            if use_last_turn:
                prefix_seq = sequence[:last_assistant_index]
                target = sequence[last_assistant_index][1]
                pair = _emit(prefix_seq, target)
                if pair is not None:
                    yield pair
            else:
                for idx in range(len(sequence) - 1):
                    role, _ = sequence[idx]
                    next_role, next_text = sequence[idx + 1]
                    if role == "human" and next_role == "gpt":
                        prefix_seq = sequence[: idx + 1]
                        pair = _emit(prefix_seq, next_text)
                        if pair is not None:
                            yield pair


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Peek into ShareGPT streaming output.")
    parser.add_argument("--jsonl", required=True, help="Path to ShareGPT JSONL file")
    parser.add_argument("--peek", type=int, default=3, help="Number of samples to display")
    parser.add_argument("--keep-system", action="store_true", help="Retain system prompts")
    parser.add_argument(
        "--use-last-turn",
        dest="use_last_turn",
        action="store_true",
        default=True,
        help="Only emit the final assistant reply per conversation",
    )
    parser.add_argument(
        "--multi-yield",
        dest="use_last_turn",
        action="store_false",
        help="Yield every human→assistant pair instead of only the final reply",
    )
    parser.add_argument("--max-src-len", type=int, default=2048)
    parser.add_argument("--max-tgt-len", type=int, default=256)
    args = parser.parse_args()

    iterator = iter_sharegpt_jsonl(
        args.jsonl,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        keep_system=args.keep_system,
        use_last_turn=args.use_last_turn,
    )

    for _ in range(args.peek):
        try:
            prompt, target = next(iterator)
        except StopIteration:
            break
        print("PROMPT:", prompt[:200].replace("\n", "⏎"))
        print("TARGET:", target[:200].replace("\n", "⏎"))
        print("---")
