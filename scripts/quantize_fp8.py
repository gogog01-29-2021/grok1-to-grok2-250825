"""
Helper script to quantize full-precision weights to FP8 format for SGLang inference.

[G2] notes:
- Use this script to quantize Grok-2 weights to FP8 as recommended by SGLang. The actual quantization routine depends on your framework.
- Read the TP=8 shards, convert each tensor to FP8, and save them in the format expected by your inference engine.
- This stub does not implement real quantization; fill in the conversion logic as needed.
"""

if __name__ == "__main__":
    print("TODO: implement FP8 quantization.")
