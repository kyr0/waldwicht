#!/usr/bin/env python3
"""Export optimal Gemma 4 E4B model: e4-ne4-g64 mixed precision.

Winner from Phase III-B near-edge depth sweep (7-model ablation):
  - PPL = 3,490 (best across all configs)
  - Size = ~4.24 GB (before multimodal stripping, may shrink)
  - Config: 3-tier mixed precision, no transforms
    - Edge layers 0-3, 38-41: 4-bit affine, g64  (8 layers)
    - Near-edge layers 4-7, 34-37: 4-bit affine, g64  (8 layers)
    - Middle layers 8-33: 3-bit affine, g128  (26 layers)

Usage:
    .venv/bin/python optimal/export_e4b.py [--output PATH]
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rotq.pipeline import RotqConfig, rotq_export
from rotq import module_policy as mp

# ── E4B Constants ─────────────────────────────────────────────────────────────

E4B_N_LAYERS = 42
EDGE_LAYERS = 4
NEAR_EDGE_LAYERS = 4
BF16_PATH = "mlx/gemma-4-E4B-it-bf16"
DEFAULT_OUTPUT = "mlx/gemma-4-E4B-e4ne4g64-optimal"


def main():
    parser = argparse.ArgumentParser(description="Export optimal Gemma 4 E4B (e4-ne4-g64)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--bf16-path", default=BF16_PATH, help="BF16 source model path")
    args = parser.parse_args()

    # Set module policy for E4B's 42-layer architecture
    mp.N_LAYERS = E4B_N_LAYERS
    mp.EDGE_LAYERS = EDGE_LAYERS
    mp.NEAR_EDGE_LAYERS = NEAR_EDGE_LAYERS

    # Promote near-edge layers from sign_q1 to affine4 (4-bit)
    _orig_classify = mp.classify_weight

    def _mixed_classify(key, w):
        cls = _orig_classify(key, w)
        if cls == "sign_q1":
            tier = mp.layer_tier(key)
            if tier == "near_edge":
                return "affine4"
        return cls

    mp.classify_weight = _mixed_classify

    config = RotqConfig(
        equalization=False,
        permutation=False,
        rotation=False,
        group_size=128,       # middle layers: 3-bit g128
        quantize_bits=3,
        affine_bits=4,        # edge + near-edge: 4-bit
        affine_group_size=64, # edge + near-edge: g64
        seed=42,
        verbose=True,
    )

    print(f"[E4B Optimal] Exporting e4-ne4-g64 → {args.output}")
    print(f"  Source: {args.bf16_path}")
    print(f"  Edge layers: 0-{EDGE_LAYERS-1}, {E4B_N_LAYERS-EDGE_LAYERS}-{E4B_N_LAYERS-1} (4-bit g64)")
    print(f"  Near-edge: {EDGE_LAYERS}-{EDGE_LAYERS+NEAR_EDGE_LAYERS-1}, "
          f"{E4B_N_LAYERS-EDGE_LAYERS-NEAR_EDGE_LAYERS}-{E4B_N_LAYERS-EDGE_LAYERS-1} (4-bit g64)")
    print(f"  Middle: {EDGE_LAYERS+NEAR_EDGE_LAYERS}-{E4B_N_LAYERS-EDGE_LAYERS-NEAR_EDGE_LAYERS-1} (3-bit g128)")
    rotq_export(args.bf16_path, args.output, config)

    # Report size
    total = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith(".safetensors")
    )
    print(f"\n[E4B Optimal] Done. Model size: {total / 1e9:.2f} GB")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
