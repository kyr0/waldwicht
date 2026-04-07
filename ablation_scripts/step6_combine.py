#!/usr/bin/env python3
"""Step 6: Combine optimal configs from all studies.

Merges the best findings from Steps 2-5:
  - Per-component bit-widths (Step 2)
  - Per-PLE-layer allocation (Step 3)
  - AWQ scaling (Step 4)
  - Per-projection group sizes (Step 5)

Exports the combined model and runs full validation.

Usage:
    .venv/bin/python calibration-aware-ple/step6_combine.py \
        --config combined_config.json

    Or specify parameters directly:
    .venv/bin/python calibration-aware-ple/step6_combine.py \
        --default-bits 4 --default-gs 64 \
        --ple-bits 3 --ple-gs 64 \
        --awq-alpha 0.5 \
        --proj-overrides '{"k_proj": 32, "v_proj": 32}'
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
    BF16_PATH, OPTIMAL_PATH, TMP_DIR, RESULTS_DIR, N_LAYERS,
    ensure_dirs, model_size_gb, evaluate_variant,
    load_bf16_weights, save_sharded, patch_config_quantization,
    run_answer_quality,
    log_result, print_result, cleanup_variant,
)

# Import AWQ scaling from step4
from step4_awq_ple import collect_ple_activations, apply_awq_scaling


def main():
    parser = argparse.ArgumentParser(description="Step 6: Combined optimal export")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config file with all parameters")

    # Direct parameter specification
    parser.add_argument("--default-bits", type=int, default=4)
    parser.add_argument("--default-gs", type=int, default=64)
    parser.add_argument("--ple-bits", type=int, default=3)
    parser.add_argument("--ple-gs", type=int, default=64)
    parser.add_argument("--awq-alpha", type=float, default=0.0,
                        help="AWQ scaling alpha (0 = no scaling)")
    parser.add_argument("--proj-overrides", type=str, default="{}",
                        help='JSON dict of {proj_type: group_size} overrides')
    parser.add_argument("--ple-layer-bits", type=str, default="{}",
                        help='JSON dict of {layer_idx: bits} for per-PLE-layer allocation')
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: TMP_DIR/combined-optimal)")
    parser.add_argument("--keep", action="store_true",
                        help="Don't clean up the output directory")
    args = parser.parse_args()

    ensure_dirs()
    results_path = os.path.join(RESULTS_DIR, "combined_optimal_e2b.json")

    # Load config from file or args
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {
            "default_bits": args.default_bits,
            "default_gs": args.default_gs,
            "ple_bits": args.ple_bits,
            "ple_gs": args.ple_gs,
            "awq_alpha": args.awq_alpha,
            "proj_overrides": json.loads(args.proj_overrides),
            "ple_layer_bits": {int(k): v for k, v in json.loads(args.ple_layer_bits).items()},
        }

    out_path = args.output or os.path.join(TMP_DIR, "combined-optimal")

    print(f"{'#' * 70}")
    print(f"  Step 6: Combined Optimal Export (E2B)")
    print(f"  Config: {json.dumps(cfg, indent=2)}")
    print(f"  Output: {out_path}")
    print(f"{'#' * 70}")

    # ── Load BF16 weights ────────────────────────────────────────────
    print("\n  Loading BF16 weights...")
    bf16_weights = load_bf16_weights()

    # ── Apply AWQ scaling if alpha > 0 ───────────────────────────────
    if cfg["awq_alpha"] > 0:
        print(f"\n  Collecting PLE activations for AWQ (α={cfg['awq_alpha']})...")
        act_stats = collect_ple_activations(BF16_PATH)
        print(f"  Applying AWQ scaling...")
        bf16_weights = apply_awq_scaling(bf16_weights, act_stats, cfg["awq_alpha"])
        del act_stats

    # ── Quantize with combined config ────────────────────────────────
    print("\n  Quantizing with combined config...")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(OPTIMAL_PATH, out_path)

    for f in os.listdir(out_path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_path, f))

    proj_gs_map = cfg.get("proj_overrides", {})
    ple_layer_bits = cfg.get("ple_layer_bits", {})

    quantized = {}
    for k in bf16_weights:
        if not k.endswith(".weight"):
            continue
        w = bf16_weights[k].astype(mx.float32)

        # Determine bits and group_size for this weight
        bits = cfg["default_bits"]
        gs = cfg["default_gs"]

        # PLE embeddings
        if "embed_tokens_per_layer" in k:
            bits = cfg["ple_bits"]
            gs = cfg["ple_gs"]
            # Per-PLE-layer override
            for layer_idx, layer_bits in ple_layer_bits.items():
                if f".{layer_idx}." in k:
                    bits = layer_bits
                    break

        # Per-projection group size override
        for proj_type, proj_gs in proj_gs_map.items():
            if proj_type in k:
                gs = proj_gs
                break

        if w.ndim != 2 or w.shape[-1] < gs or w.shape[-1] % gs != 0:
            quantized[k] = bf16_weights[k]
            continue

        weight, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
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
    del bf16_weights
    gc.collect()

    save_sharded(quantized, out_path)
    del quantized
    gc.collect()
    mx.clear_cache()

    # ── Patch config.json ────────────────────────────────────────────
    config_overrides = {}
    # PLE layers
    for i in range(N_LAYERS):
        bits = ple_layer_bits.get(i, cfg["ple_bits"])
        config_overrides[f"embed_tokens_per_layer.{i}"] = {
            "bits": bits, "group_size": cfg["ple_gs"],
        }
    # Per-projection GS
    for proj_type, proj_gs in proj_gs_map.items():
        config_overrides[proj_type] = {
            "bits": cfg["default_bits"], "group_size": proj_gs,
        }

    patch_config_quantization(
        os.path.join(out_path, "config.json"),
        global_bits=cfg["default_bits"],
        global_gs=cfg["default_gs"],
        overrides=config_overrides,
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    size = model_size_gb(out_path)
    print(f"\n  Model size: {size:.2f} GB")

    print("\n  Running 64-prompt calibration eval...")
    t0 = time.time()
    metrics = evaluate_variant(out_path)
    metrics["size_gb"] = size
    print(f"  Evaluated in {time.time() - t0:.1f}s")
    print_result("combined-optimal", size, metrics)

    # ── Answer quality ───────────────────────────────────────────────
    aq_out = os.path.join(RESULTS_DIR, "answer_quality_combined_optimal.json")
    print(f"\n  Generating answer quality responses...")
    run_answer_quality(out_path, aq_out)

    # ── Compare to current winner ────────────────────────────────────
    current_winner_size = 2.32
    current_winner_ppl = 5452

    print(f"\n{'═' * 60}")
    print(f"  Comparison to current E2B winner:")
    print(f"    Current: {current_winner_size:.2f} GB / PPL {current_winner_ppl}")
    print(f"    Combined: {size:.2f} GB / PPL {metrics['ppl']}")
    delta_size = size - current_winner_size
    delta_ppl = metrics["ppl"] - current_winner_ppl
    print(f"    Δ size: {delta_size:+.2f} GB ({delta_size/current_winner_size*100:+.1f}%)")
    print(f"    Δ PPL:  {delta_ppl:+.0f} ({delta_ppl/current_winner_ppl*100:+.1f}%)")

    if size <= current_winner_size and metrics["ppl"] <= current_winner_ppl:
        print(f"  ✓ BEATS current winner on both size and PPL!")
    elif size < current_winner_size:
        print(f"  ◐ Smaller but higher PPL — check quality gate")
    elif metrics["ppl"] < current_winner_ppl:
        print(f"  ◐ Better PPL but larger — check if trade-off is acceptable")
    else:
        print(f"  ✗ Does not beat current winner")

    # Log
    entry = {"name": "combined-optimal", "config": cfg, **metrics}
    log_result(results_path, entry)

    if not args.keep:
        cleanup_variant(out_path)

    print(f"\n{'=' * 70}")
    print(f"  Results: {results_path}")
    print(f"  AQ file: {aq_out}")
    print(f"  Score AQ file, then run check_quality_gate() to validate.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
