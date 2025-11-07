"""Inspect Kangaroo block-dump files to see drafted vs verifier tokens."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import List


def _load_records(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_blocks(path: str, limit: int, sample_filter: int | None) -> None:
    reason_counter = Counter()
    accepted_values: List[int] = []
    printed = 0

    for record in _load_records(path):
        reason = record.get("reject_reason", "unknown")
        reason_counter[reason] += 1
        accepted = record.get("accepted")
        if isinstance(accepted, (int, float)):
            accepted_values.append(int(accepted))

        if printed < limit and (sample_filter is None or record.get("sample_id") == sample_filter):
            print(
                f"[block] sample={record.get('sample_id')} idx={record.get('block_index')} "
                f"len={record.get('draft_len')} accepted={accepted} reason={reason} "
                f"mismatch={record.get('mismatch_index')}"
            )
            print(f"        draft   : {record.get('draft_preview')}")
            print(f"        verifier: {record.get('verifier_preview')}")
            printed += 1

    if not accepted_values:
        print(f"[block] no entries found in {path}")
        return

    mean_accepted = sum(accepted_values) / len(accepted_values)
    frac_gt1 = sum(1 for v in accepted_values if v and v > 1) / len(accepted_values)
    print(f"[block] entries={len(accepted_values)} mean_accept={mean_accepted:.2f} frac_gt1={frac_gt1*100:.2f}%")
    print("[block] reject reasons:")
    for reason, count in reason_counter.most_common():
        print(f"  {reason or 'none'}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print drafted vs verifier token previews from a block dump file.")
    parser.add_argument("path", help="Path passed to --sharegpt-block-dump")
    parser.add_argument("--limit", type=int, default=5, help="How many entries to print verbosely.")
    parser.add_argument("--sample-id", type=int, default=None, help="Filter to a specific sample_id.")
    args = parser.parse_args()
    summarize_blocks(args.path, args.limit, args.sample_id)


if __name__ == "__main__":
    main()
