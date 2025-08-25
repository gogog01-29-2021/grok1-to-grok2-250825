"""
Entry point for Grok-2 style model.
This script loads a checkpoint directory, reads its config (falling back to
config_grok2_like.json), constructs a Grok2Model, and runs a simple
prompt-driven generation loop.

[G2] Notes:
- Pass --tokenizer_path for the 131072-token BPE file (tokenizer.tok.json).
- Expect TP=8 checkpoint shards; Checkpoint will need to stitch them.
- This stub does not implement actual inference or tokenization.
"""

import argparse
import json
from .model import Grok2Model, Config
from .checkpoint import Checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grok-2 model inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer.tok.json")
    parser.add_argument("--prompt", type=str, default="", help="Initial text to prompt the model")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Load config from checkpoint
    ckpt = Checkpoint(args.checkpoint_dir)
    cfg_data = ckpt.config()
    cfg = Config(**cfg_data)

    # Build model (weights not yet loaded)
    model = Grok2Model(cfg)
    # TODO: load weights via ckpt.load_weight and assign to model parameters

    # TODO: load the tokenizer from args.tokenizer_path
    # This stub simply echoes the prompt and returns
    print("Model initialized. Prompt:\n", args.prompt)
    print("[G2] TODO: implement tokenization, generation, and postprocessing.")


if __name__ == "__main__":
    main()
