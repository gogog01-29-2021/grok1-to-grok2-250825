"""
Residual Mixture-of-Experts feed-forward network with top-2 routing and dense residual path.
"""

from typing import Any

class ResidualMoE:
    def __init__(self, hidden_size: int, intermediate_size: int,
                 moe_intermediate_size: int, num_local_experts: int,
                 num_experts_per_tok: int, router_cap: float):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_cap = router_cap

    def __call__(self, x: Any) -> Any:
        """
        1. Compute dense FFN output.
        2. Compute router logits to select experts; clamp logits by self.router_cap. [G2]
        3. Route each token to num_experts_per_tok out of num_local_experts, using moe_intermediate_size width. [G2]
        4. Combine expert outputs and residual dense output.
        """
        raise NotImplementedError
