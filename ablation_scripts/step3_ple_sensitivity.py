#!/usr/bin/env python3
"""Step 3: Per-PLE-layer sensitivity sweep.

For each of the 35 PLE embedding tables, demote it one rung below the PLE
floor (from Step 2) while keeping all other PLE layers at the floor.
Rank layers by PPL sensitivity, then greedily demote the least-sensitive
layers until the quality gate fails.

Usage:
    .venv/bin/python calibration-aware-ple/step3_ple_sensitivity.py \
        --ple-floor-bits 3 --other-bits 4
"""
import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

from infrastructure import (
    OPTIMAL_PATH, TMP_DIR, RESULTS_DIR, N_LAYERS, VALID_BITS,
    ensure_dirs, model_size_gb, compute_ppl,
    load_bf16_weights, save_sharded,
    log_result, cleanup_variant,
)
from mlx_lm import load


def bits_below(current: int) -> list[int]:
    return sorted([b for b in VALID_BITS if b < current], reverse=True)


def create_single_ple_demoted(
    bf16_weights: dict,
    base_model_path: str,
    out_path: str,
    layer_idx: int,
    ple_bits: int,
    ple_gs: int,
    demoted_bits: int,
    demoted_gs: int,
    other_bits: int,
    other_gs: int,
) -> float:
    """Create a variant where one PLE layer is demoted, rest stay at ple_bits.

    All non-PLE weights are quantized at other_bits/other_gs.
    """
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(base_model_path, out_path)

    # Remove old weights
    for f in os.listdir(out_path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_path, f))

    # Load existing optimal weights and replace PLE tensors
    opt_shards = sorted(glob.glob(os.path.join(base_model_path, "*.safetensors")))
    opt_weights = {}
    for s in opt_shards:
        opt_weights.update(mx.load(s))

    # Re-quantize all PLE layers from BF16
    for k in bf16_weights:
        if "embed_tokens_per_layer" not in k or not k.endswith(".weight"):
            continue

        # Determine if this is the demoted layer
        # Key pattern: "...embed_tokens_per_layer.{idx}.weight"
        is_demoted = f".{layer_idx}." in k

        bits = demoted_bits if is_demoted else ple_bits
        gs = demoted_gs if is_demoted else ple_gs

        w = bf16_weights[k].astype(mx.float32)
        if w.ndim != 2 or w.shape[-1] < gs or w.shape[-1] % gs != 0:
            continue

        weight, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
        mx.eval(weight, scales, biases)

        base = k[: -len(".weight")]
        opt_weights[base + ".weight"] = weight
        opt_weights[base + ".scales"] = scales
        opt_weights[base + ".biases"] = biases

    mx.eval(*list(opt_weights.values()))
    save_sharded(opt_weights, out_path)

    # Patch config with per-layer overrides
    config_path = os.path.join(out_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    quant = config.get("quantization", {})
    quant["bits"] = other_bits
    quant["group_size"] = other_gs

    # Set all PLE layers at ple_bits, except the demoted one
    for i in range(N_LAYERS):
        key = f"embed_tokens_per_layer.{i}"
        if i == layer_idx:
            quant[key] = {"bits": demoted_bits, "group_size": demoted_gs}
        else:
            quant[key] = {"bits": ple_bits, "group_size": ple_gs}

    config["quantization"] = quant
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    del opt_weights
    gc.collect()
    mx.clear_cache()

    return model_size_gb(out_path)


def main():
    parser = argparse.ArgumentParser(description="Step 3: Per-PLE-layer sensitivity")
    parser.add_argument("--ple-floor-bits", type=int, required=True,
                        help="PLE floor from Step 2 (the bit-width PLE passed at)")
    parser.add_argument("--other-bits", type=int, required=True,
                        help="Bit-width for non-PLE weights (uniform floor from Step 1)")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Only test first N PLE layers (for debugging)")
    args = parser.parse_args()

    ensure_dirs()
    results_path = os.path.join(RESULTS_DIR, "ple_sensitivity_e2b.json")

    demote_candidates = bits_below(args.ple_floor_bits)
    if not demote_candidates:
        print("No valid bit-widths below the PLE floor. Nothing to test.")
        return

    demoted_bits = demote_candidates[0]  # One rung below
    n_layers = args.max_layers or N_LAYERS

    print(f"{'#' * 70}")
    print(f"  Step 3: Per-PLE-Layer Sensitivity (E2B)")
    print(f"  PLE floor: {args.ple_floor_bits}-bit → demoting to {demoted_bits}-bit")
    print(f"  Non-PLE: {args.other_bits}-bit g{args.group_size}")
    print(f"  Testing {n_layers} PLE layers")
    print(f"{'#' * 70}")

    # Load BF16 weights once
    print("  Loading BF16 weights...")
    bf16_weights = load_bf16_weights()

    # Phase 1: Single-layer demotion sweep
    layer_results = []

    for i in range(n_layers):
        name = f"ple-layer-{i}-demoted-{demoted_bits}bit"
        out_path = os.path.join(TMP_DIR, name)

        print(f"\n  Layer {i}/{n_layers - 1}...")
        t0 = time.time()

        size = create_single_ple_demoted(
            bf16_weights, OPTIMAL_PATH, out_path,
            layer_idx=i,
            ple_bits=args.ple_floor_bits,
            ple_gs=args.group_size,
            demoted_bits=demoted_bits,
            demoted_gs=args.group_size,
            other_bits=args.other_bits,
            other_gs=args.group_size,
        )

        # Quick PPL-only eval (fast — no generation)
        model, tok = load(out_path)
        ppl = compute_ppl(model, tok)
        del model, tok
        gc.collect()
        mx.clear_cache()

        elapsed = time.time() - t0
        print(f"    PPL: {ppl:.1f} | {size:.2f} GB | {elapsed:.1f}s")

        layer_results.append({
            "layer": i,
            "ppl": round(ppl, 1),
            "size_gb": round(size, 3),
        })

        cleanup_variant(out_path)

    del bf16_weights
    gc.collect()
    mx.clear_cache()

    # Phase 2: Rank by sensitivity
    # Sort by PPL ascending — lowest PPL = least sensitive to demotion
    layer_results.sort(key=lambda r: r["ppl"])

    print(f"\n{'═' * 60}")
    print(f"  Sensitivity ranking (least → most sensitive):")
    print(f"{'═' * 60}")
    for rank, lr in enumerate(layer_results):
        print(f"  {rank + 1:3d}. Layer {lr['layer']:2d}: PPL {lr['ppl']:.1f}")

    # Save
    entry = {
        "ple_floor_bits": args.ple_floor_bits,
        "demoted_bits": demoted_bits,
        "other_bits": args.other_bits,
        "group_size": args.group_size,
        "layer_ranking": layer_results,
    }
    log_result(results_path, entry)

    print(f"\n  Results saved to {results_path}")
    print(f"\n  Next: Use the ranking to greedily demote least-sensitive layers.")
    print(f"  The top-N least-sensitive layers can be set to {demoted_bits}-bit.")


if __name__ == "__main__":
    main()
