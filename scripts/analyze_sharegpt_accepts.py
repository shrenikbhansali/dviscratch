"""Summarize accept-length stats from a ShareGPT run log."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean, median
from typing import List


def _flatten_accepts(path: str) -> List[int]:
    accepts: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            accepts.extend(int(x) for x in record.get("accept_lengths", []) if isinstance(x, (int, float)))
    return accepts


def summarize_accepts(path: str) -> None:
    values = _flatten_accepts(path)
    if not values:
        print(f"[analyze] no accept_lengths found in {path}")
        return

    counter = Counter(values)
    total = len(values)
    gt_one = sum(v for k, v in counter.items() if k > 1)
    zeros = counter.get(0, 0)
    avg = mean(values)

    med = median(values)
    print(f"[analyze] file={path}")
    print(f"[analyze] samples={total} mean={avg:.2f} median={med:.2f}")
    print(f"[analyze] >1 tokens accepted: {gt_one}/{total} ({(gt_one / total) * 100:.2f}%)")
    print(f"[analyze] zero-token blocks: {zeros}/{total} ({(zeros / total) * 100:.2f}%)")
    print("[analyze] top counts:")
    for length, count in counter.most_common(10):
        print(f"  accept={length}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect accept_lengths distributions.")
    parser.add_argument("jsonl", help="Path to data/sharegpt_runs/... jsonl file")
    args = parser.parse_args()
    summarize_accepts(args.jsonl)


if __name__ == "__main__":
    main()
