"""Download and convert ShareGPT dataset to JSONL.

Dataset: https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
License: Apache-2.0 (see dataset card for details).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from huggingface_hub import dataset_info


def _normalize_role(role: Any) -> str:
    if role is None:
        return ""
    role_str = str(role).lower()
    if role_str in {"assistant", "model"}:
        return "gpt"
    if role_str == "user":
        return "human"
    if role_str in {"system", "human", "gpt"}:
        return role_str
    return role_str


def _select_value(turn: Dict[str, Any]) -> str:
    for key in ("value", "text", "markdown"):
        if key in turn and turn[key] not in (None, ""):
            return str(turn[key])
    return ""


def _clean_conversation(conversation: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(turn.get("from"))
        text = _select_value(turn)
        if role == "system":
            # System turns are retained as-is; runtime loader decides whether to use them.
            cleaned.append({"from": role, "value": text})
        elif role in {"human", "gpt"}:
            cleaned.append({"from": role, "value": text})
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ShareGPT dataset and convert to JSONL")
    parser.add_argument("--out-jsonl", default="data/raw/sharegpt_aeala.train.jsonl")
    parser.add_argument("--out-meta", default="data/raw/sharegpt_aeala.meta.json")
    parser.add_argument("--dataset", default="Aeala/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)

    with open(args.out_jsonl, "w", encoding="utf-8") as stream:
        for example in dataset:
            conversations = example.get("conversations") or []
            cleaned_conv = _clean_conversation(conversations)
            if not cleaned_conv:
                continue
            record = {
                "id": str(example.get("id", "")),
                "conversations": cleaned_conv,
            }
            json_record = json.dumps(record, ensure_ascii=False)
            stream.write(json_record + "\n")

    revision = None
    try:
        info = dataset_info(args.dataset)
        revision = getattr(info, "sha", None)
    except Exception:  # pragma: no cover - best-effort metadata fetch
        revision = None

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "rows": len(dataset),
        "created_at": int(time.time()),
        "revision": revision,
    }
    with open(args.out_meta, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
