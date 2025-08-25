"""
RoPE utilities for long-context rotary position embeddings.
"""

from typing import Any


def build_rope_cache(max_positions: int, dim: int, theta: float):
    """
    Build sinusoidal cache for RoPE with long context; returns cos and sin caches. [G2]
    """
    raise NotImplementedError


def apply_rope(x: Any, theta: float) -> Any:
    """
    Apply RoPE to tensor x using precomputed caches.  This stub defers to build_rope_cache. [G2]
    """
    raise NotImplementedError
