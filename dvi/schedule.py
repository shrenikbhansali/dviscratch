from __future__ import annotations

from typing import Dict


class PiecewiseSchedule:
    """KL→RL curriculum with explicit formulas and persistence."""

    def __init__(self, warmup: int, kl0: float, klmin: float, pgmax: float) -> None:
        self.warmup = max(0, int(warmup))
        self.kl0 = float(kl0)
        self.klmin = float(klmin)
        self.pgmax = float(pgmax)
        # IMPORTANT: callers should increment AFTER applying an optimizer step.
        self.step = 0

    def weights(self) -> Dict[str, float]:
        if self.warmup == 0:
            t = 1.0
        else:
            t = min(1.0, self.step / self.warmup)
        kd = 0.5 * (1.0 - t) + 0.5
        ce = 0.5
        pg = self.pgmax * t
        kl = self.kl0 * (1.0 - t) + self.klmin * t
        ent = 0.0
        return {"kd": kd, "ce": ce, "pg": pg, "kl": kl, "ent": ent}

    def state_dict(self) -> Dict[str, float | int]:
        return {
            "step": self.step,
            "warmup": self.warmup,
            "kl0": self.kl0,
            "klmin": self.klmin,
            "pgmax": self.pgmax,
        }

    def load_state_dict(self, s: Dict[str, float | int]) -> None:
        self.step = int(s["step"])
        self.warmup = int(s["warmup"])
        self.kl0 = float(s["kl0"])
        self.klmin = float(s["klmin"])
        self.pgmax = float(s["pgmax"])
