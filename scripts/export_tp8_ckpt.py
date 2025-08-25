"""
Helper script to shard a full precision checkpoint into tensor-parallel (TP) shards.

[G2] notes:
- Grok-2 checkpoints are stored as TP=8 shards. Use this script as a starting point to split a full checkpoint into 8 parts.
- Implement logic to read dense and expert weights, partition them evenly across shards, and save them in your desired format (e.g. safetensors or npy).
- This script currently contains only comments; fill in the I/O and partitioning details according to your checkpoint format.
"""

if __name__ == "__main__":
    print("TODO: implement TP=8 checkpoint sharding.")
