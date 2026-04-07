#!/usr/bin/env python3
"""Export optimal Gemma 4 E2B model: uniform 3-bit, group_size=64.

Winner from Phase 1 baselines (22-config ablation):
  - PPL = 5,452 (best across all 22 configs)
  - Size = ~2.2 GB (after multimodal stripping)
  - Config: uniform q3-g64, no transforms, no mixed precision
  - All linear layers quantized identically at 3-bit g64

Usage:
    .venv/bin/python optimal/export_e2b.py [--output PATH]
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rotq.pipeline import RotqConfig, rotq_export
from rotq import module_policy as mp

# ── E2B Constants ─────────────────────────────────────────────────────────────

E2B_N_LAYERS = 35
BF16_PATH = "hf/gemma-4-E2B-it"
DEFAULT_OUTPUT = "mlx/gemma-4-E2B-q3-g64-optimal"

def main():
    parser = argparse.ArgumentParser(description="Export optimal Gemma 4 E2B (q3-g64)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--bf16-path", default=BF16_PATH, help="BF16 source model path")
    args = parser.parse_args()

    # Patch module policy for E2B's 35-layer architecture
    mp.N_LAYERS = E2B_N_LAYERS

    # For uniform quantization: set edge/near-edge to 0 so everything is "sign_q1",
    # then reclassify all sign_q1 as affine4 to get uniform bits
    mp.EDGE_LAYERS = 0
    mp.NEAR_EDGE_LAYERS = 0

    _orig_classify = mp.classify_weight

    def _uniform_classify(key, w):
        cls = _orig_classify(key, w)
        if cls == "sign_q1":
            return "affine4"
        return cls

    mp.classify_weight = _uniform_classify

    config = RotqConfig(
        equalization=False,
        permutation=False,
        rotation=False,
        group_size=64,
        quantize_bits=3,
        affine_bits=3,
        affine_group_size=64,
        seed=42,
        verbose=True,
    )

    print(f"[E2B Optimal] Exporting uniform q3-g64 → {args.output}")
    print(f"  Source: {args.bf16_path}")
    rotq_export(args.bf16_path, args.output, config)

    # Report size
    total = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith(".safetensors")
    )
    print(f"\n[E2B Optimal] Done. Model size: {total / 1e9:.2f} GB")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
