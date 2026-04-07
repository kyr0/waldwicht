#!/usr/bin/env python3
"""Phase 1: Baselines + rotation sanity-check for Gemma 4 E2B.

Exports and evaluates:
  1. bf16           — Reference (BF16, ~10 GB)
  2. q4-g64         — Uniform 4-bit, g=64 (upper bound quality)
  3. q3-g128        — Uniform 3-bit, g=128 (target precision)
  4. q3-g64         — Uniform 3-bit, g=64 (group size comparison)
  5. q3-rotq-sanity — 3-bit g128 + edge 4-bit + block-WHT rotation
                      (confirm rotation is harmful for E2B too)

Same prompts and PPL text as E4B for cross-model comparison.

Usage:
    uv run python scripts_E2B/ablation_baselines.py
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts_E2B.e2b_config import (
    BF16_PATH, TMP_DIR, E2B_N_LAYERS, patch_module_policy,
)
from scripts_E2B.eval_utils import (
    PROMPTS, PPL_TEXT, model_size_gb, compute_ppl,
    test_generation, print_generation_results, avg_tok_per_sec,
)

import mlx.core as mx
from mlx_lm import load
from rotq.pipeline import RotqConfig, rotq_export
from rotq import module_policy as mp

# ── Patch for E2B ─────────────────────────────────────────────────────────────
patch_module_policy()

# ── Export Functions ──────────────────────────────────────────────────────────


def export_uniform(name: str, bits: int, group_size: int) -> str:
    """Export a uniform quantization (all layers same bits/group)."""
    out_path = os.path.join(TMP_DIR, f"e2b-{name}")

    # For uniform quant: set edge_layers=0 so everything is "sign_q1"
    # but actually we want uniform, so set all layers to the same bits.
    # Use affine_bits=bits and classify everything as affine4.
    orig_edge = mp.EDGE_LAYERS
    orig_ne = mp.NEAR_EDGE_LAYERS
    _orig_classify = mp.classify_weight

    # Everything at same bit-width: classify all linear as affine4
    def _uniform_classify(key, w):
        cls = _orig_classify(key, w)
        if cls == "sign_q1":
            return "affine4"
        return cls
    mp.classify_weight = _uniform_classify
    mp.EDGE_LAYERS = 0
    mp.NEAR_EDGE_LAYERS = 0

    config = RotqConfig(
        equalization=False,
        permutation=False,
        rotation=False,
        group_size=group_size,
        quantize_bits=bits,
        affine_bits=bits,
        affine_group_size=group_size,
        seed=42,
        verbose=False,
    )
    rotq_export(BF16_PATH, out_path, config)

    mp.EDGE_LAYERS = orig_edge
    mp.NEAR_EDGE_LAYERS = orig_ne
    mp.classify_weight = _orig_classify
    return out_path


def export_mixed_plain(name: str, edge_layers: int, group_size: int) -> str:
    """Export mixed 3-bit middle + 4-bit edge, no transforms."""
    out_path = os.path.join(TMP_DIR, f"e2b-{name}")

    orig_edge = mp.EDGE_LAYERS
    mp.EDGE_LAYERS = edge_layers

    config = RotqConfig(
        equalization=False,
        permutation=False,
        rotation=False,
        group_size=group_size,
        quantize_bits=3,
        affine_bits=4,
        affine_group_size=64,
        seed=42,
        verbose=False,
    )
    rotq_export(BF16_PATH, out_path, config)

    mp.EDGE_LAYERS = orig_edge
    return out_path


def export_mixed_rotq(name: str, edge_layers: int) -> str:
    """Export mixed 3-bit + 4-bit edge WITH rotation (sanity-check)."""
    out_path = os.path.join(TMP_DIR, f"e2b-{name}")

    orig_edge = mp.EDGE_LAYERS
    mp.EDGE_LAYERS = edge_layers

    config = RotqConfig(
        equalization=True,
        permutation=True,
        rotation=True,
        equalization_method="abs_max",
        rotation_families=["block_hadamard"],
        rotation_candidates=4,
        rotation_block_size=128,
        scale_method="mean_abs",
        group_size=128,
        quantize_bits=3,
        affine_bits=4,
        affine_group_size=64,
        seed=42,
        verbose=False,
    )
    rotq_export(BF16_PATH, out_path, config)

    mp.EDGE_LAYERS = orig_edge
    return out_path


# ── Model Configurations ─────────────────────────────────────────────────────

# (name, export_fn, kwargs)
CONFIGS = [
    ("q4-g64",         export_uniform,    {"bits": 4, "group_size": 64}),
    ("q3-g128",        export_uniform,    {"bits": 3, "group_size": 128}),
    ("q3-g64",         export_uniform,    {"bits": 3, "group_size": 64}),
    ("q3-plain-e4",    export_mixed_plain, {"edge_layers": 4, "group_size": 128}),
    ("q3-rotq-sanity", export_mixed_rotq, {"edge_layers": 4}),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("GEMMA 4 E2B — PHASE 1: BASELINES + ROTATION SANITY CHECK")
    print("=" * 90)
    print(f"  Model: Gemma 4 E2B ({E2B_N_LAYERS} layers, hidden=1536)")
    print(f"  BF16:  {BF16_PATH}")
    print(f"  Tmp:   {TMP_DIR}")
    print()

    os.makedirs(TMP_DIR, exist_ok=True)
    all_results = []

    # ── BF16 Reference ────────────────────────────────────────────────────
    print(f"\n{'─' * 90}")
    print("CONFIG: bf16 (reference)")
    print(f"{'─' * 90}")

    model, tok = load(BF16_PATH)
    size = model_size_gb(BF16_PATH)
    ppl = compute_ppl(model, tok)
    print(f"  Size = {size:.2f} GB, PPL = {ppl:.2f}")
    gen_results = test_generation(model, tok, max_tokens=60)
    print_generation_results(gen_results)
    all_results.append({
        "name": "bf16",
        "bits": 16,
        "group_size": None,
        "transforms": "none",
        "size_gb": round(size, 3),
        "ppl": round(ppl, 2),
        "generation": gen_results,
    })
    del model, tok
    gc.collect()

    # ── Quantized Variants ────────────────────────────────────────────────
    for name, export_fn, kwargs in CONFIGS:
        print(f"\n{'─' * 90}")
        print(f"CONFIG: {name} ({kwargs})")
        print(f"{'─' * 90}")

        t0 = time.time()
        path = export_fn(name, **kwargs)
        export_time = time.time() - t0
        size = model_size_gb(path)
        print(f"  Exported in {export_time:.1f}s — {size:.2f} GB")

        model, tok = load(path)
        ppl = compute_ppl(model, tok)
        print(f"  PPL = {ppl:.2f}")

        gen_results = test_generation(model, tok, max_tokens=60)
        print_generation_results(gen_results)

        # Determine bits and transforms from name
        transforms = "rotation+eq+perm" if "rotq" in name else "none"
        bits = 4 if name.startswith("q4") else 3

        all_results.append({
            "name": name,
            "bits": bits,
            "group_size": kwargs.get("group_size", 128),
            "transforms": transforms,
            "size_gb": round(size, 3),
            "ppl": round(ppl, 2),
            "generation": gen_results,
        })

        del model, tok
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("SUMMARY — E2B BASELINES")
    print(f"{'=' * 90}")
    print(f"  {'Config':<20} {'Bits':>4} {'Size':>7} {'PPL':>10} {'Speed':>8} {'Transforms'}")
    print(f"  {'-'*20} {'-'*4} {'-'*7} {'-'*10} {'-'*8} {'-'*15}")
    for r in sorted(all_results, key=lambda x: x["ppl"]):
        speed = avg_tok_per_sec(r["generation"])
        print(f"  {r['name']:<20} {r['bits']:>4} {r['size_gb']:>6.2f}G "
              f"{r['ppl']:>10.2f} {speed:>6.1f}t/s {r['transforms']}")

    # Save
    out_json = os.path.join(os.path.dirname(__file__), "..", "ablation_e2b_baselines.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
