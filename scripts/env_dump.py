#!/usr/bin/env python3
"""Emit a deterministic environment snapshot in a YAML-like structure."""
from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from typing import Any, Dict, List


def safe_import(name: str):
    try:
        module = __import__(name)
    except ImportError:
        return None
    else:
        return module


def get_git_info() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception as err:  # pragma: no cover - defensive
        commit = f"(error: {err.__class__.__name__})"
    return {"commit": commit}


def get_cuda_info(torch_module) -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    if torch_module is None:
        return info

    cuda_available = bool(torch_module.cuda.is_available())
    info["available"] = cuda_available
    info["version"] = getattr(torch_module.version, "cuda", None) or "(not available)"
    cudnn_version = getattr(torch_module.backends.cudnn, "version", lambda: None)()
    info["cudnn"] = cudnn_version if cudnn_version is not None else "(not available)"

    if cuda_available:
        devices: List[str] = []
        for idx in range(torch_module.cuda.device_count()):
            try:
                devices.append(torch_module.cuda.get_device_name(idx))
            except Exception:
                devices.append(f"cuda:{idx}")
        info["devices"] = devices
    return info


def format_value(value: Any, indent: int = 0) -> List[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key in sorted(value):
            lines.append(f"{prefix}{key}:" + ("" if isinstance(value[key], (dict, list)) else f" {value[key]}"))
            lines.extend(format_value(value[key], indent + 2))
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(format_value(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
        return lines
    return []


def main() -> None:
    torch_module = safe_import("torch")
    transformers_module = safe_import("transformers")
    accelerate_module = safe_import("accelerate")
    peft_module = safe_import("peft")
    datasets_module = safe_import("datasets")

    packages = {}
    for name, module in [
        ("torch", torch_module),
        ("transformers", transformers_module),
        ("accelerate", accelerate_module),
        ("peft", peft_module),
        ("datasets", datasets_module),
    ]:
        version = getattr(module, "__version__", "(not installed)") if module else "(not installed)"
        packages[name] = version

    timestamp = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    payload = {
        "timestamp_utc": timestamp,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": get_git_info(),
        "cuda": get_cuda_info(torch_module),
        "packages": packages,
    }

    print("environment:")
    for line in format_value(payload, indent=2):
        print(line)


if __name__ == "__main__":
    main()
