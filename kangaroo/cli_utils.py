"""Utility helpers for Kangaroo command line interfaces."""
from __future__ import annotations

import argparse
from typing import Optional


def str2bool(value: str) -> bool:
    """Convert string arguments into booleans for argparse."""
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def add_model_and_adapter_args(parser: argparse.ArgumentParser) -> None:
    """Register model and adapter related CLI flags on *parser*."""
    parser.add_argument("--model-id", type=str, required=True, help="HuggingFace model id or local path")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Legacy alias for --model-id; must match if both are provided.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="none",
        help="Directory containing adapter weights or 'none' for identity adapter.",
    )
    parser.add_argument(
        "--adapter-mode",
        type=str,
        choices=["auto", "none", "load"],
        default="auto",
        help="Adapter loading strategy. 'auto' infers from --adapter-path.",
    )


def add_dvi_args(parser: argparse.ArgumentParser) -> None:
    """Register DVI control flags."""
    parser.add_argument("--dvi-online", type=str2bool, default=False, help="Enable online DVI updates.")
    parser.add_argument("--dvi-batch-size", type=int, default=64)
    parser.add_argument("--dvi-update-every", type=int, default=8)
    parser.add_argument("--dvi-buffer-size", type=int, default=2048)
    parser.add_argument(
        "--dvi-store",
        type=str,
        choices=["full", "topk"],
        default="topk",
        help="Tuple storage mode for online DVI training.",
    )
    parser.add_argument("--dvi-topk", type=int, default=1024)
    parser.add_argument(
        "--dvi-logits-dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--dvi-tau", type=float, default=1.5)
    parser.add_argument("--dvi-warmup-steps", type=int, default=1000)
    parser.add_argument("--dvi-pg-lambda-max", type=float, default=0.2)
    parser.add_argument("--dvi-kl-lambda0", type=float, default=1.0)
    parser.add_argument("--dvi-kl-lambdamin", type=float, default=0.1)
    parser.add_argument("--dvi-entropy-weight", type=float, default=0.0)
    parser.add_argument("--max-online-train-ms", type=int, default=5)
    parser.add_argument("--load-lora", type=str, default="")
    parser.add_argument("--save-lora-every", type=int, default=0)
    parser.add_argument("--save-lora-path", type=str, default="runs/checkpoints")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dvi-debug-hk",
        type=str2bool,
        default=False,
        help="Print shallow hk capture statistics for each draft block.",
    )


def normalize_model_flags(args: argparse.Namespace) -> str:
    """Return the resolved model identifier from *args* or raise on mismatch."""
    model_id = args.model_id
    if args.model_path:
        if not model_id:
            model_id = args.model_path
        elif args.model_path != model_id:
            raise argparse.ArgumentError(None, "--model-path must equal --model-id or be omitted")
    if not model_id:
        raise argparse.ArgumentError(None, "--model-id is required")
    return model_id


def normalize_adapter_path(path: Optional[str]) -> Optional[str]:
    """Translate adapter path strings into canonical values."""
    if path is None:
        return None
    lowered = path.strip().lower()
    return None if lowered in {"", "none"} else path


def resolve_adapter_mode(args: argparse.Namespace) -> str:
    """Return the effective adapter mode from parsed *args*."""
    normalized_path = normalize_adapter_path(getattr(args, "adapter_path", None))
    mode = getattr(args, "adapter_mode", "auto")

    if mode == "none" or normalized_path is None:
        return "none"
    if mode == "load":
        return "load"
    # auto mode
    return "load" if normalized_path else "none"
