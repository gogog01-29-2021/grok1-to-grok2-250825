# Upgrade Notes for Grok‑1 JAX Example

To align the Grok‑1 JAX example code with the Grok‑2 configuration, consider the following modifications. These notes are intended for a developer familiar with the Grok‑1 code base; actual line numbers depend on your local checkout.

## 1. Configuration

- Replace the existing `config.json` in the repository with `config_grok2_like.json` provided here.
- Ensure your configuration parser supports fields such as `num_key_value_heads`, `scaling_factor`, `rope_theta`, and the soft‑capping parameters.

## 2. Attention Module

- **Grouped‑Query Attention**: Modify the self‑attention implementation so that query projections produce 64 heads while the key and value projections produce only eight groups. Share the key/value projections among multiple query heads.
- **Head dimension**: Keep the head dimension at 128 so that `hidden_size = num_attention_heads × head_dim` remains consistent.
- **Temperature schedule**: Introduce an optional temperature scaling for the attention logits that depends on sequence length (`attn_temperature_len`) if the Grok‑1 code does not already implement this.

## 3. Rotary Position Embeddings (RoPE)

- Increase the maximum sequence length to 131 072 tokens by applying position interpolation with `scaling_factor: 16.0`.
- Modify the RoPE implementation to accept a configurable `rope_theta` (~2.0853×10^8) to ensure stability over long contexts.

## 4. Mixture‑of‑Experts (MoE) Feed‑Forward Network

- Adjust the MoE layer to consist of eight local experts (`num_local_experts: 8`).
- Implement top‑2 routing (`num_experts_per_tok: 2`) so that each token activates at most two experts per layer.
- Retain a dense residual feed‑forward path alongside the MoE experts (`residual_moe: true`).
- Set the expert intermediate size to 16 384 and the shared dense intermediate size to 32 768.

## 5. Stability and Scaling

- Implement soft‑capping on logits by clipping or scaling attention logits, router logits, and final logits according to `attn_logit_softcapping`, `router_logit_softcapping`, and `final_logit_softcapping` values.
- Introduce `embedding_multiplier_scale` and `output_multiplier_scale` factors that scale the embedding outputs and the final logits respectively.

## 6. Vocabulary and Tokenizer

- Replace the old tokenizer with the new `tokenizer.tok.json` (vocab size 131 072).
- Resize the embedding and output projection matrices accordingly.

## 7. Additional Considerations

- These changes increase memory and computational requirements. You may need to adjust your parallelism strategy (tensor and data parallelism) to handle the larger context and MoE experts.
- The provided notes do not include modifications to optimizer hyperparameters, training data, or fine‑tuning procedures. Those aspects are not public in the Grok‑2 release.
