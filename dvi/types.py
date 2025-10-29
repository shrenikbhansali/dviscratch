"""Core DVI type definitions for training data structures."""

from typing import NamedTuple, Optional


class DVITrainSample(NamedTuple):
    """Container for DVI training samples.

    This placeholder matches the structure expected by later training
    steps. The loader introduced in Step 2 does not instantiate this
    directly but downstream components will rely on the stable import.
    """

    hk: "object" = None
    token: Optional[int] = None
    reward: Optional[int] = None
    pos: Optional[int] = None
    z_phi: "object" = None
    z_idx: "object" = None
    z_val: "object" = None
    is_first_reject: Optional[bool] = None


__all__ = ["DVITrainSample"]
