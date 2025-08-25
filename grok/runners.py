"""
Command-line runner for the Grok-2 style model.

This script wraps run.main with a basic CLI interface. It mirrors the
structure of the Grok-1 demo's runners.py and adds placeholders for
features unique to Grok-2 such as tensor-parallel sharding and FP8
quantization.

[G2] Notes:
- Accepts --tensor_parallel to indicate the number of TP shards (default 8).
- Accepts --quantize_fp8 flag to enable FP8 quantization (requires SGLang or
  compatible backend).
- Passes through common arguments for checkpoint_dir, tokenizer_path,
  prompt and max_tokens.
- For actual inference, see run.py which loads the model and runs generation.
"""

import argparse
from .run import main as run_main


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI wrapper for Grok-2 inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the Grok checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer.tok.json file")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text to feed the model")
    parser.add_argument("--max_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--tensor_parallel", type=int, default=8, help="Number of tensor-parallel shards")
    parser.add_argument("--quantize_fp8", action="store_true", help="Enable FP8 quantization (requires support)")
    args = parser.parse_args()

    # [G2] Here you could set up environment variables or flags for TP and FP8
    # For now we simply delegate to run_main, which uses args.checkpoint_dir,
    # args.tokenizer_path, args.prompt and args.max_tokens. Additional flags are
    # placeholders and not used in the stub.
    run_main()


if __name__ == "__main__":
    main()
