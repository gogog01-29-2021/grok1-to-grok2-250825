# Grok‑2 Config Notes

This document summarises the key fields in the `config.json` file from the Grok‑2 Hugging Face release. It lists each parameter and explains its role.

## Annotated Configuration

- **Family & block type** (`architectures`: ["Grok1ForCausalLM"]) – Identifies the model as a decoder‑only causal language model. No encoder component is included.
- **Depth & width** (`num_hidden_layers`: 64, `hidden_size`: 8192) – The model has 64 transformer layers and a very wide hidden dimension (8192). The original Grok‑1 also used 64 layers, but Grok‑2 increases the hidden width.
- **Attention topology** (`num_attention_heads`: 64, `head_dim`: 128, `num_key_value_heads`: 8) – There are 64 query heads and 8 grouped key/value heads (GQA). Each attention head has dimension 128 (64 × 128 = 8192).
- **Mixture‑of‑Experts** (`num_local_experts`: 8, `num_experts_per_tok`: 2, `residual_moe`: true) – Each transformer block contains eight feed‑forward experts, but only two are active per token. A dense residual FFN path runs alongside the experts.
- **Feed‑forward widths** (`intermediate_size`: 32768, `moe_intermediate_size`: 16384) – The dense FFN uses a width of 32 768 while each expert uses a width of 16 384.
- **Norms & dtypes** (`rms_norm_eps`: 1e‑5, `torch_dtype`: "bfloat16") – RMSNorm with epsilon 1e‑5 and bfloat16 weights for reduced memory footprint.
- **Context & RoPE** (`original_max_position_embeddings`: 8192, `scaling_factor`: 16.0, `max_position_embeddings`: 131072, `rope_type`: "original", `rope_theta`: 208533496) – Positional encoding uses RoPE. The original context length of 8192 tokens is extended to 131072 via a scaling factor of 16. The large `rope_theta` ensures stability at long sequence lengths.
- **Temperature scheduling** (`attn_temperature_len`: 1024) – Introduces a length‑dependent temperature schedule for attention, possibly to modulate focus over long contexts.
- **Stability knobs** (`attn_logit_softcapping`: 30.0, `router_logit_softcapping`: 30.0, `final_logit_softcapping`: 50, `embedding_multiplier_scale`, `output_multiplier_scale`) – Soft‑caps logits and scales embeddings/outputs to avoid numerical overflow during training and inference.
- **Tokenizer & vocabulary** (`vocab_size`: 131072, `tokenizer.tok.json`) – Uses a large BPE‑style tokenizer with 131 072 tokens.
- **Serving hint** – The README indicates that the release is about 500 GB of weights and is saved as a TP=8 checkpoint. Users are encouraged to serve the model via SGLang with FP8 quantization and Triton attention, implying multi‑GPU and multi‑host deployment requirements.

## Operational Implications

- The mixture‑of‑experts design means the total parameter count is extremely large (estimated around 270 B parameters) but only a subset of weights is active per token (attention, dense FFN and two experts).
- Grouped‑Query Attention reduces KV cache memory by sharing keys and values across 8 groups instead of 64 heads; this is crucial for the long (131k token) context window.
- Combining RoPE scaling and a large `rope_theta` keeps attention stable over the 131k token context length.
