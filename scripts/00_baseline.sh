#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

mkdir -p runs/baseline

python -m scripts.env_dump > runs/baseline/env.txt

SEED=${SEED:-123}
MODEL_ID=${MODEL_ID:-lmsys/vicuna-7b-v1.5}
ADAPTER_PATH=${ADAPTER_PATH:-none}
EXITLAYER=${EXITLAYER:-2}
THRESHOLD=${THRESHOLD:-0.6}
STEPS=${STEPS:-6}
TEMPERATURE=${TEMPERATURE:-0.0}
PROMPTS=${PROMPTS:-data/smoke_prompts.jsonl}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MODEL_PATH=${MODEL_PATH:-$MODEL_ID}

TIME_BIN="/usr/bin/time"
TIME_FORMAT="%E real, %U user, %S sys, %M KB maxrss"
TIME_AVAILABLE=0
if [ -x "$TIME_BIN" ]; then
  TIME_AVAILABLE=1
fi

run_with_timing() {
  local label=$1
  local out_file=$2
  local err_file=$3
  local time_file=$4
  shift 4
  if [ "$TIME_AVAILABLE" -eq 1 ]; then
    if ! "$TIME_BIN" -f "$TIME_FORMAT" -o "$time_file" "$@" >"$out_file" 2>"$err_file"; then
      return $?
    fi
  else
    if ! python - "$out_file" "$err_file" "$time_file" "${label}" "$@" <<'PY'
import os
import subprocess
import sys
import time

out_file, err_file, time_file, label, *cmd = sys.argv[1:]
start = time.time()
with open(out_file, "w", encoding="utf-8") as out_f, open(err_file, "w", encoding="utf-8") as err_f:
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
    proc.wait()
elapsed = time.time() - start
with open(time_file, "w", encoding="utf-8") as fh:
    fh.write(f"{elapsed:.3f}s real, N/A user, N/A sys, N/A KB maxrss")
if proc.returncode != 0:
    sys.exit(proc.returncode)
PY
    then
      return $?
    fi
  fi
  return 0
}

KANGAROO_OUT="runs/baseline/kangaroo.out"
KANGAROO_ERR="runs/baseline/kangaroo.err"
KANGAROO_TIME="runs/baseline/kangaroo.time"

PROMPTS_FLAG_SUPPORTED=$(python - <<'PY' 2>/dev/null
import evaluation.inference_kangaroo as mod
parser = mod.build_parser()
print(int(any('--prompts-jsonl' in opt for action in parser._actions for opt in action.option_strings)))
PY
)
PROMPTS_FLAG_SUPPORTED=${PROMPTS_FLAG_SUPPORTED:-0}
PROMPTS_FLAG_SUPPORTED=${PROMPTS_FLAG_SUPPORTED//[[:space:]]/}

if [ "$PROMPTS_FLAG_SUPPORTED" = "1" ]; then
  KANGAROO_DRIVER="evaluation.inference_kangaroo"
  KANGAROO_CMD=(python -m evaluation.inference_kangaroo \
    --model-id "$MODEL_ID" \
    --adapter-path "$ADAPTER_PATH" \
    --exitlayer "$EXITLAYER" --threshold "$THRESHOLD" --steps "$STEPS" \
    --temperature "$TEMPERATURE" --seed "$SEED" \
    --prompts-jsonl "$PROMPTS")
else
  KANGAROO_DRIVER="inline_kangaroo_runner"
  FALLBACK_SCRIPT="runs/baseline/_kangaroo_driver.py"
  cat <<'PY' > "$FALLBACK_SCRIPT"
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path
import random

import torch
from transformers import AutoTokenizer

# Ensure the project root is importable when this script lives under runs/
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import inference_kangaroo as kangaroo
from kangaroo.kangaroo_model import KangarooModel


def normalize_adapter(path: str | None) -> tuple[str, str | None]:
    if path is None:
        return "none", None
    lowered = path.strip().lower()
    if lowered in {"", "none"}:
        return "none", None
    return "load", path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt")
            if prompt is None:
                continue
            prompts.append(prompt)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inline Kangaroo runner for smoke prompts")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", default="none")
    parser.add_argument("--exitlayer", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    adapter_mode, adapter_path = normalize_adapter(args.adapter_path)
    seed_everything(args.seed)

    dtype = "float16" if torch.cuda.is_available() else "float32"
    model = KangarooModel(
        model_id=args.model_id,
        adapter_mode=adapter_mode,
        adapter_path=adapter_path,
        exit_layer=args.exitlayer,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    device = next(model.base_model.parameters()).device

    prompts = load_prompts(Path(args.prompts_jsonl))
    do_sample = float(args.temperature) > 0.0

    for index, prompt in enumerate(prompts, start=1):
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs, _, _, _ = kangaroo.kangaroo_forward(
                inputs,
                model,
                tokenizer,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                EARLY_STOP_LAYER=args.exitlayer,
                SPECULATIVE_DECODING_STEPS=args.steps,
                threshold=args.threshold,
            )
        generated = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"### Prompt {index}")
        print(prompt)
        print("---")
        print(text)
        print()


if __name__ == "__main__":
    main()
PY
  KANGAROO_CMD=(python "$FALLBACK_SCRIPT" \
    --model-id "$MODEL_ID" \
    --adapter-path "$ADAPTER_PATH" \
    --exitlayer "$EXITLAYER" --threshold "$THRESHOLD" --steps "$STEPS" \
    --temperature "$TEMPERATURE" --seed "$SEED" \
    --prompts-jsonl "$PROMPTS" \
    --max-new-tokens "$MAX_NEW_TOKENS")
fi

set +e
run_with_timing "kangaroo" "$KANGAROO_OUT" "$KANGAROO_ERR" "$KANGAROO_TIME" "${KANGAROO_CMD[@]}"
KANGAROO_EXIT=$?
set -e

GREEDY_OUT="runs/baseline/greedy.out"
GREEDY_ERR="runs/baseline/greedy.err"
GREEDY_TIME="runs/baseline/greedy.time"
GREEDY_EXIT=0
GREEDY_STATUS="skipped"
GREEDY_DRIVER=""

if [ -f evaluation/inference_baseline.py ]; then
  GREEDY_PROMPTS_SUPPORTED=$(python - <<'PY'
from pathlib import Path

path = Path("evaluation/inference_baseline.py")
if not path.exists():
    print(0)
else:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    print(int("--prompts-jsonl" in text))
PY
)
  GREEDY_PROMPTS_SUPPORTED=${GREEDY_PROMPTS_SUPPORTED:-0}
  GREEDY_PROMPTS_SUPPORTED=${GREEDY_PROMPTS_SUPPORTED//[[:space:]]/}

  if [ "$GREEDY_PROMPTS_SUPPORTED" = "1" ]; then
    GREEDY_DRIVER="evaluation.inference_baseline"
    GREEDY_SEED_SUPPORTED=$(python - <<'PY'
from pathlib import Path

text = ""
try:
    text = Path("evaluation/inference_baseline.py").read_text(encoding="utf-8", errors="ignore")
except Exception:
    pass
print(int("--seed" in text))
PY
)
    GREEDY_SEED_SUPPORTED=${GREEDY_SEED_SUPPORTED:-0}
    GREEDY_SEED_SUPPORTED=${GREEDY_SEED_SUPPORTED//[[:space:]]/}
    EXTRA_GREEDY_FLAGS=()
    if [ "$GREEDY_SEED_SUPPORTED" = "1" ]; then
      EXTRA_GREEDY_FLAGS+=(--seed "$SEED")
    fi
    GREEDY_CMD=(python -m evaluation.inference_baseline \
      --model-path "$MODEL_PATH" \
      --model-id "$MODEL_ID" \
      --temperature "$TEMPERATURE" \
      "${EXTRA_GREEDY_FLAGS[@]}" \
      --prompts-jsonl "$PROMPTS")
  else
    GREEDY_DRIVER="inline_greedy_runner"
    GREEDY_FALLBACK_SCRIPT="runs/baseline/_greedy_driver.py"
    cat <<'PY' > "$GREEDY_FALLBACK_SCRIPT"
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kangaroo.kangaroo_model import KangarooModel


def normalize_adapter(path: str | None) -> tuple[str, str | None]:
    if path is None:
        return "none", None
    lowered = path.strip().lower()
    if lowered in {"", "none"}:
        return "none", None
    return "load", path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt")
            if prompt is None:
                continue
            prompts.append(prompt)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inline greedy runner for smoke prompts")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path", default="none")
    parser.add_argument("--exitlayer", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    adapter_mode, adapter_path = normalize_adapter(args.adapter_path)
    seed_everything(args.seed)

    source_id = args.model_path or args.model_id
    dtype = "float16" if torch.cuda.is_available() else "float32"
    model = KangarooModel(
        model_id=source_id,
        adapter_mode=adapter_mode,
        adapter_path=adapter_path,
        exit_layer=args.exitlayer,
        dtype=dtype,
    )
    base_model = model.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    device = next(base_model.parameters()).device

    prompts = load_prompts(Path(args.prompts_jsonl))
    do_sample = float(args.temperature) > 0.0

    for index, prompt in enumerate(prompts, start=1):
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = base_model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][encoded["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"### Prompt {index}")
        print(prompt)
        print("---")
        print(text)
        print()


if __name__ == "__main__":
    main()
PY
    GREEDY_CMD=(python "$GREEDY_FALLBACK_SCRIPT" \
      --model-id "$MODEL_ID" \
      --model-path "$MODEL_PATH" \
      --adapter-path "$ADAPTER_PATH" \
      --exitlayer "$EXITLAYER" \
      --temperature "$TEMPERATURE" \
      --seed "$SEED" \
      --prompts-jsonl "$PROMPTS" \
      --max-new-tokens "$MAX_NEW_TOKENS")
  fi

  if [ -n "$GREEDY_CMD" ]; then
    set +e
    run_with_timing "greedy" "$GREEDY_OUT" "$GREEDY_ERR" "$GREEDY_TIME" "${GREEDY_CMD[@]}"
    GREEDY_EXIT=$?
    set -e
    GREEDY_STATUS="completed"
  else
    echo "evaluation/inference_baseline.py present but CLI unsupported; skipping greedy baseline" > runs/baseline/greedy.SKIPPED
  fi
else
  echo "evaluation/inference_baseline.py not found; skipping greedy baseline" > runs/baseline/greedy.SKIPPED
fi

KANGAROO_RERUN_OUT="runs/baseline/kangaroo_rerun.out"
KANGAROO_RERUN_ERR="runs/baseline/kangaroo_rerun.err"
if ! "${KANGAROO_CMD[@]}" > "$KANGAROO_RERUN_OUT" 2> "$KANGAROO_RERUN_ERR"; then
  echo "WARNING: Determinism re-run exited with non-zero status" >> "$KANGAROO_RERUN_ERR"
fi

export SEED MODEL_ID ADAPTER_PATH EXITLAYER THRESHOLD STEPS TEMPERATURE PROMPTS KANGAROO_DRIVER MAX_NEW_TOKENS
export KANGAROO_EXIT GREEDY_EXIT GREEDY_STATUS GREEDY_DRIVER
python - <<'PY'
from __future__ import annotations

import os
import pathlib
import re
import textwrap

BASE = pathlib.Path("runs/baseline")
BASE.mkdir(parents=True, exist_ok=True)

def read_text(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

def first_chars(path: pathlib.Path, limit: int = 200) -> str:
    data = read_text(path)
    if not data:
        return "(no output)"
    snippet = data[:limit]
    snippet = snippet.replace("```", "`\u200b``")
    return snippet

def parse_time(path: pathlib.Path) -> tuple[str, str]:
    text = read_text(path).strip()
    if not text:
        return "N/A", "N/A"
    real = "N/A"
    rss = "N/A"
    match_real = re.search(r"([^,]+)\s+real", text)
    if match_real:
        real = match_real.group(1).strip()
    match_rss = re.search(r"([0-9.]+)\s*KB maxrss", text)
    if match_rss:
        rss = match_rss.group(1).strip()
    return real, rss

def determinism_result(primary: pathlib.Path, rerun: pathlib.Path) -> tuple[str, str]:
    first = first_chars(primary)
    second = first_chars(rerun)
    if first == second:
        return "pass", "First 200 characters match across runs."
    diff_lines = [
        "Primary:",
        first,
        "\nRe-run:",
        second,
    ]
    return "fail", "\n".join(diff_lines)

def load_commit(env_path: pathlib.Path) -> str:
    for line in read_text(env_path).splitlines():
        if line.strip().startswith("commit:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "(unknown)"

def find_timestamp(env_path: pathlib.Path) -> str:
    for line in read_text(env_path).splitlines():
        if line.strip().startswith("timestamp_utc:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "(unknown)"

def format_flags() -> str:
    return (
        f"model_id={os.environ.get('MODEL_ID')} "
        f"adapter_path={os.environ.get('ADAPTER_PATH')} "
        f"exitlayer={os.environ.get('EXITLAYER')} "
        f"threshold={os.environ.get('THRESHOLD')} "
        f"steps={os.environ.get('STEPS')} "
        f"temperature={os.environ.get('TEMPERATURE')} "
        f"seed={os.environ.get('SEED')} "
        f"prompts={os.environ.get('PROMPTS')} "
        f"max_new_tokens={os.environ.get('MAX_NEW_TOKENS')} "
        f"driver={os.environ.get('KANGAROO_DRIVER')}"
    )

env_path = BASE / "env.txt"
commit = load_commit(env_path)
timestamp = find_timestamp(env_path)

kangaroo_real, kangaroo_rss = parse_time(BASE / "kangaroo.time")
if not kangaroo_real:
    kangaroo_real = "N/A"
if not kangaroo_rss:
    kangaroo_rss = "N/A"

greedy_real, greedy_rss = parse_time(BASE / "greedy.time")
if os.environ.get("GREEDY_STATUS") != "completed":
    driver = os.environ.get("GREEDY_DRIVER")
    if driver:
        greedy_note = f"driver={driver} skipped"
    else:
        greedy_note = "skipped"
else:
    driver = os.environ.get("GREEDY_DRIVER") or "evaluation.inference_baseline"
    greedy_note = f"driver={driver} (exit={os.environ.get('GREEDY_EXIT')})"

kangaroo_note = (
    f"driver={os.environ.get('KANGAROO_DRIVER')} "
    f"exitlayer={os.environ.get('EXITLAYER')} "
    f"threshold={os.environ.get('THRESHOLD')} "
    f"steps={os.environ.get('STEPS')} "
    f"(exit={os.environ.get('KANGAROO_EXIT')})"
)

det_status, det_details = determinism_result(BASE / "kangaroo.out", BASE / "kangaroo_rerun.out")

readme_lines = [
    "# Baseline smoke test",
    "",
    f"Generated: {timestamp}",
    "",
    f"Commit: `{commit}`",
    "",
    "## Configuration",
    "",
    f"``""\n{format_flags()}\n``""",
    "",
    "<details>",
    "<summary>Environment snapshot</summary>",
    "",
    "```",
    read_text(env_path).strip(),
    "```",
    "",
    "</details>",
    "",
    "## Timings",
    "",
    "| Run | Real time | Max RSS (KB) | Seed | Notes |",
    "| --- | --- | --- | --- | --- |",
    f"| Kangaroo (identity) | {kangaroo_real} | {kangaroo_rss} | {os.environ.get('SEED')} | {kangaroo_note} |",
]

if os.environ.get("GREEDY_STATUS") == "completed":
    readme_lines.append(
        f"| Greedy (baseline) | {greedy_real} | {greedy_rss} | {os.environ.get('SEED')} | {greedy_note} |"
    )
else:
    readme_lines.append(
        f"| Greedy (baseline) | N/A | N/A | {os.environ.get('SEED')} | {greedy_note} |"
    )

readme_lines.extend([
    "",
    "## Output snippets (first 200 characters)",
    "",
    "```", "Kangaroo", "```",
    "```",
    first_chars(BASE / "kangaroo.out"),
    "```",
])

if os.environ.get("GREEDY_STATUS") == "completed":
    readme_lines.extend([
        "",
        "```", "Greedy", "```",
        "```",
        first_chars(BASE / "greedy.out"),
        "```",
    ])
else:
    skip_msg = read_text(BASE / "greedy.SKIPPED").strip()
    readme_lines.extend([
        "",
        "Greedy baseline: " + (skip_msg or "skipped."),
    ])

readme_lines.extend([
    "",
    "## Determinism",
    "",
    f"Status: **{det_status.upper()}**",
    "",
])

if det_status != "pass":
    readme_lines.extend([
        "```",
        det_details.strip(),
        "```",
    ])
else:
    readme_lines.append(det_details)

readme_path = BASE / "README.md"
readme_path.write_text("\n".join(readme_lines).strip() + "\n", encoding="utf-8")

print(f"README written to {readme_path}")
print(f"Determinism status: {det_status.upper()}")
PY

printf 'Kangaroo exit code: %s\n' "$KANGAROO_EXIT"
if [ "$GREEDY_STATUS" = "completed" ]; then
  printf 'Greedy exit code: %s\n' "$GREEDY_EXIT"
else
  printf 'Greedy baseline status: %s\n' "$GREEDY_STATUS"
fi
printf 'Artifacts directory: %s\n' "runs/baseline"
