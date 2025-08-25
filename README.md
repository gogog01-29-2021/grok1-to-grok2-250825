# grok1-to-grok2-250825

This repository packages an annotated upgrade kit for moving the open-source Grok-1 demo towards the Grok-2 architecture released on Hugging Face. It contains:

- `config_grok2_like.json`: a Grok-2 style configuration extracted from the Hugging Face drop.
- `grok2_config_notes.md`: comments on each config parameter and what it implies.
- `upgrade_notes.md`: deeper explanations of the architectural differences.
- `grok/` package: skeleton code with [G2] comments marking where to implement Grouped-Query Attention, a residual Mixture-of-Experts layer, long-context Rotary Positional Embeddings, and FP8/TP=8 checkpoint loading.

## Migration checklist

Follow these steps to upgrade your Grok-1 codebase:

1. **Load the new configuration**

   Use `config_grok2_like.json` when loading model hyperparameters. Key deltas include:
   - `hidden_size = 8192`
   - `num_hidden_layers = 64`
   - `num_attention_heads = 64` (with `num_key_value_heads = 8` for GQA)
   - `intermediate_size = 32768`, `moe_intermediate_size = 16384`
   - `num_local_experts = 8`, `num_experts_per_tok = 2`
   - `original_max_position_embeddings = 8192`, `scaling_factor = 16.0` -> effective `max_position_embeddings = 131072`
   - `rope_theta = 208533496`
   - Soft caps: `attn_logit_softcapping = 30.0`, `router_logit_softcapping = 30.0`, `final_logit_softcapping = 50.0`
   - `vocab_size = 131072`

   See `grok2_config_notes.md` for a field-by-field explanation.

2. **Implement Grouped-Query Attention**

   Split query heads (`num_attention_heads`) from key/value heads (`num_key_value_heads`). Repeat K/V across groups to reduce KV-cache size. Add a per-token temperature schedule (optional) and clamp attention logits to `attn_logit_softcapping`. See `grok/attention.py` for a stub.

3. **Add long-context RoPE**

   Build a RoPE cache sized for `max_position_embeddings` and use the large `rope_theta` to preserve stability over 131k tokens. You can keep short-range behaviour consistent by matching the original 8k positions. See `grok/rope.py`.

4. **Introduce a residual Mixture-of-Experts layer**

   After attention + projection, route each token to the top 2 of 8 experts and combine their outputs with a dense residual feedforward path. Clip router logits to `router_logit_softcapping`. See `grok/moe.py`.

5. **Clamp outputs**

   Clip final logits to `final_logit_softcapping` and optionally scale embeddings/output projections via `embedding_multiplier_scale` and `output_multiplier_scale` if your model saturates.

6. **Adapt checkpoint loading**

   Grok-2 weights are stored as TP=8, FP8 quantised shards. Update your checkpoint loader to:

   - Look for `config_grok2_like.json` if `config.json` is missing.
   - Stitch 8 shards into full tensors.
   - Dequantise FP8 weights when serving with SGLang.

   The `grok/checkpoint.py` stub contains comments where to implement this.

7. **Swap tokeniser**

   Use the new 131072-token `tokenizer.tok.json` provided by xAI. Ensure EOS and special tokens align.

8. **Verify with unit tests**

   Test attention shapes with synthetic Q/K/V to ensure GQA works, test router determinism for MoE, and verify RoPE still matches Grok-1 at 8k length.

These steps should bring a Grok-1 style JAX implementation up to parity with the publicly released Grok-2 architecture while maintaining clarity and modularity.
