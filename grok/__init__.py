"""
Grok-2 adaptation package.

This package reorganizes Grok1 code into modular components, allowing
upgrades to the attention, RoPE, and mixture-of-experts layers without
rewriting everything. Each module contains [G2] tags for lines that
diverge from the original Grok1 implementation.
"""

# Re-export the main model for convenience
from .model import Grok2Model
