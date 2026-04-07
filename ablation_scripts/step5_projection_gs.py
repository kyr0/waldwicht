#!/usr/bin/env python3
"""Step 5: Per-projection group size tuning.

Test g32, g64, g128 for each projection type at the bit-widths established
by Steps 1-4. Larger group_size = smaller model (fewer scales/biases),
but potentially lower quality.

Projection types tested:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
  per_layer_input_gate, per_layer_projection

Usage:
    .venv/bin/python calibration-aware-ple/step5_projection_gs.py \
        --default-bits 4 --default-gs 64
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

from infrastructure import (
    OPTIMAL_PATH, TMP_DIR, RESULTS_DIR,
    ensure_dirs, model_size_gb, compute_ppl,
    load_bf16_weights, save_sharded, patch_config_quantization,
    log_result, cleanup_variant,
)
from mlx_lm import load

PROJECTION_TYPES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "per_layer_input_gate", "per_layer_projection",
]

GROUP_SIZES = [32, 64, 128]


def export_with_proj_gs(
    bf16_weights: dict,
    out_path: str,
    default_bits: int,
    default_gs: int,
    proj_type: str,
    proj_gs: int,
) -> float:
    """Quantize with one projection type at a different group_size."""
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(OPTIMAL_PATH, out_path)

    for f in os.listdir(out_path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_path, f))

    quantized = {}
    for k in bf16_weights:
        if not k.endswith(".weight"):
            continue
        w = bf16_weights[k].astype(mx.float32)

        # Use proj_gs for matching projection type, default for rest
        gs = proj_gs if proj_type in k else default_gs

        if w.ndim != 2 or w.shape[-1] < gs or w.shape[-1] % gs != 0:
            quantized[k] = bf16_weights[k]
            continue

        weight, scales, biases = mx.quantize(w, group_size=gs, bits=default_bits)
        mx.eval(weight, scales, biases)
        base = k[: -len(".weight")]
        quantized[f"{base}.weight"] = weight
        quantized[f"{base}.scales"] = scales
        quantized[f"{base}.biases"] = biases

    # Copy non-weight tensors
    for k, v in bf16_weights.items():
        if k not in quantized and not any(k.endswith(s) for s in (".scales", ".biases")):
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            if f"{base}.scales" not in quantized:
                quantized[k] = v

    mx.eval(*list(quantized.values()))
    save_sharded(quantized, out_path)
    del quantized
    gc.collect()
    mx.clear_cache()

    overrides = {proj_type: {"bits": default_bits, "group_size": proj_gs}}
    patch_config_quantization(
        os.path.join(out_path, "config.json"),
        global_bits=default_bits, global_gs=default_gs,
        overrides=overrides,
    )

    return model_size_gb(out_path)


def main():
    parser = argparse.ArgumentParser(description="Step 5: Per-projection group size tuning")
    parser.add_argument("--default-bits", type=int, required=True,
                        help="Default bit-width (uniform floor)")
    parser.add_argument("--default-gs", type=int, default=64,
                        help="Default group size")
    parser.add_argument("--projections", nargs="+", default=None,
                        help="Only test these projection types")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=None,
                        help="Group sizes to test (default: 32 64 128)")
    args = parser.parse_args()

    ensure_dirs()
    results_path = os.path.join(RESULTS_DIR, "projection_gs_e2b.json")

    projections = args.projections or PROJECTION_TYPES
    group_sizes = args.group_sizes or GROUP_SIZES

    print(f"{'#' * 70}")
    print(f"  Step 5: Per-Projection Group Size Tuning (E2B)")
    print(f"  Default: {args.default_bits}-bit g{args.default_gs}")
    print(f"  Projections: {', '.join(projections)}")
    print(f"  Group sizes: {group_sizes}")
    print(f"  Total experiments: {len(projections) * len(group_sizes)}")
    print(f"{'#' * 70}")

    print("\n  Loading BF16 weights...")
    bf16_weights = load_bf16_weights()

    for proj in projections:
        print(f"\n{'═' * 60}")
        print(f"  Projection: {proj}")
        print(f"{'═' * 60}")

        for gs in group_sizes:
            if gs == args.default_gs:
                print(f"\n  g{gs} = default, skipping (baseline)")
                continue

            name = f"{proj}-g{gs}"
            out_path = os.path.join(TMP_DIR, name)

            print(f"\n  {proj} @ g{gs}...")
            t0 = time.time()

            size = export_with_proj_gs(
                bf16_weights, out_path,
                default_bits=args.default_bits,
                default_gs=args.default_gs,
                proj_type=proj,
                proj_gs=gs,
            )

            # Quick PPL eval
            model, tok = load(out_path)
            ppl = compute_ppl(model, tok)
            del model, tok
            gc.collect()
            mx.clear_cache()

            elapsed = time.time() - t0
            print(f"    PPL: {ppl:.1f} | {size:.2f} GB | {elapsed:.1f}s")

            entry = {
                "name": name,
                "projection": proj,
                "group_size": gs,
                "default_bits": args.default_bits,
                "default_gs": args.default_gs,
                "ppl": round(ppl, 1),
                "size_gb": round(size, 3),
            }
            log_result(results_path, entry)
            cleanup_variant(out_path)

    del bf16_weights
    gc.collect()
    mx.clear_cache()

    # Print summary
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"\n{'═' * 60}")
        print(f"  Summary (sorted by PPL):")
        print(f"{'═' * 60}")
        all_results.sort(key=lambda r: r.get("ppl", float("inf")))
        for r in all_results:
            print(f"  {r['name']:30s} PPL: {r['ppl']:.1f}  Size: {r['size_gb']:.2f} GB")

    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
