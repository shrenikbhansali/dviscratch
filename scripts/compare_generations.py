"""Compare two ShareGPT generation logs for latency and losslessness."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

# Ensure the repo root is importable when running as a script.
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from kangaroo.cli_utils import str2bool


def _load_records(path: str) -> Dict[int, dict]:
    records: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            if sample_id is None:
                continue
            if sample_id in records:
                print(f"[compare] warning: duplicate sample_id={sample_id} in {path}, keeping last occurrence")
            records[int(sample_id)] = payload
    return records


def _norm_text(text: str, normalize_space: bool) -> str:
    text = (text or "").strip()
    if normalize_space:
        text = " ".join(text.split())
    return text


def _summarize(records: Dict[int, dict]) -> Tuple[float, float, float]:
    wall = 0.0
    tokens = 0.0
    count = 0
    for payload in records.values():
        wall += float(payload.get("wall_time", 0.0))
        tokens += float(payload.get("new_tokens", 0))
        count += 1
    avg_wall = wall / count if count else math.nan
    tok_per_sec = tokens / wall if wall > 0 else math.nan
    avg_tokens = tokens / count if count else math.nan
    return avg_wall, tok_per_sec, avg_tokens


def _format_ratio(a: float, b: float) -> str:
    if math.isnan(a) or math.isnan(b) or b == 0:
        return "n/a"
    return f"{a / b:.2f}x"


def compare_logs(
    *,
    candidate_path: str,
    reference_path: str,
    normalize_space: bool,
    max_examples: int,
) -> None:
    cand = _load_records(candidate_path)
    ref = _load_records(reference_path)

    shared_ids = sorted(set(cand) & set(ref))
    missing_in_cand = sorted(set(ref) - set(cand))
    missing_in_ref = sorted(set(cand) - set(ref))

    print(f"[compare] candidate={candidate_path}")
    print(f"[compare] reference={reference_path}")
    print(f"[compare] shared samples={len(shared_ids)} missing_candidate={len(missing_in_cand)} missing_reference={len(missing_in_ref)}")

    lossless = 0
    mismatches: List[Tuple[int, str, str]] = []
    for sample_id in shared_ids:
        cand_text = _norm_text(cand[sample_id].get("output", ""), normalize_space)
        ref_text = _norm_text(ref[sample_id].get("output", ""), normalize_space)
        if cand_text == ref_text:
            lossless += 1
        else:
            if len(mismatches) < max_examples:
                mismatches.append((sample_id, cand_text, ref_text))

    lossless_ratio = (lossless / len(shared_ids)) * 100 if shared_ids else 0.0
    print(f"[compare] lossless matches: {lossless}/{len(shared_ids)} ({lossless_ratio:.2f}%)")

    cand_avg_wall, cand_tps, cand_avg_tok = _summarize(cand)
    ref_avg_wall, ref_tps, ref_avg_tok = _summarize(ref)

    print(
        "[compare] avg wall time (cand/ref): "
        f"{cand_avg_wall:.2f}s vs {ref_avg_wall:.2f}s (ratio={_format_ratio(ref_avg_wall, cand_avg_wall)})"
    )
    print(
        "[compare] tokens/s (cand/ref): "
        f"{cand_tps:.2f} vs {ref_tps:.2f} (speedup={_format_ratio(cand_tps, ref_tps)})"
    )
    print(
        "[compare] avg new tokens (cand/ref): "
        f"{cand_avg_tok:.2f} vs {ref_avg_tok:.2f}"
    )

    if missing_in_cand:
        print(f"[compare] warning: {len(missing_in_cand)} reference samples missing in candidate log (e.g. {missing_in_cand[:5]})")
    if missing_in_ref:
        print(f"[compare] warning: {len(missing_in_ref)} candidate samples missing in reference log (e.g. {missing_in_ref[:5]})")

    if mismatches:
        print(f"[compare] showing {len(mismatches)} mismatched samples (compare --max-examples to adjust):")
        for sample_id, cand_text, ref_text in mismatches:
            print(f"  sample {sample_id}:")
            print(f"    candidate: {cand_text[:160]}{'...' if len(cand_text) > 160 else ''}")
            print(f"    reference: {ref_text[:160]}{'...' if len(ref_text) > 160 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DVI vs vanilla ShareGPT logs.")
    parser.add_argument("--candidate", required=True, help="JSONL log from the Kangaroo/DVI run.")
    parser.add_argument("--reference", required=True, help="JSONL log from the vanilla baseline.")
    parser.add_argument(
        "--normalize-space",
        type=str2bool,
        default=True,
        help="Collapse whitespace before comparing outputs (default: true).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Show at most this many mismatch examples.",
    )
    args = parser.parse_args()
    compare_logs(
        candidate_path=args.candidate,
        reference_path=args.reference,
        normalize_space=args.normalize_space,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
