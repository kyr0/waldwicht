#!/usr/bin/env python3
"""Step 4: AWQ-style activation-aware PLE channel scaling.

Collects per-channel activation statistics from the PLE embedding outputs
during calibration, then scales PLE weights by s and compensates
per_layer_projection columns by 1/s. This is mathematically exact
(diag(1/s) * diag(s) = I) and requires NO inference changes.

PLE path:
  output = per_layer_projection @ (gelu(per_layer_input_gate @ h) ⊙ ple(x))

Scaling ple by s and per_layer_projection by 1/s preserves the output exactly.

Usage:
    .venv/bin/python calibration-aware-ple/step4_awq_ple.py \
        --target-bits 3 --other-bits 4
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
    CALIBRATION_PATH,
    ensure_dirs, model_size_gb, evaluate_variant,
    load_bf16_weights, save_sharded, patch_config_quantization,
    run_answer_quality, log_result, print_result, cleanup_variant,
)
from mlx_lm import load


def collect_ple_activations(
    model_path: str,
    calibration_path: str = CALIBRATION_PATH,
    max_samples: int = 16,
) -> dict[int, mx.array]:
    """Run calibration prompts and capture per-layer PLE embedding outputs.

    Returns dict mapping layer_index → per-channel mean absolute activation
    of shape (256,).
    """
    model, tok = load(model_path)

    with open(calibration_path) as f:
        cal = json.load(f)

    samples = cal["samples"][:max_samples]

    # We need to hook into the PLE lookup result.
    # In Gemma4TextModel, get_per_layer_inputs() does:
    #   chunks = [emb(input_ids) for emb in self.embed_tokens_per_layer]
    #   result = mx.stack(chunks, axis=-2) * embed_tokens_per_layer_scale
    # We hook the output of each embed_tokens_per_layer[i]

    lang_model = model.language_model if hasattr(model, "language_model") else model
    text_model = lang_model.model if hasattr(lang_model, "model") else lang_model

    ple_layers = text_model.embed_tokens_per_layer
    n_ple = len(ple_layers)

    # Accumulate per-channel abs activation magnitudes
    act_accum = {i: None for i in range(n_ple)}
    n_tokens_total = 0
    hooks = []

    def make_hook(idx):
        def hook(module, inputs, output):
            nonlocal act_accum, n_tokens_total
            # output shape: (batch, seq_len, 256)
            abs_out = mx.abs(output).mean(axis=(0, 1))  # (256,)
            mx.eval(abs_out)
            if act_accum[idx] is None:
                act_accum[idx] = abs_out
            else:
                act_accum[idx] = act_accum[idx] + abs_out
        return hook

    for i, emb in enumerate(ple_layers):
        h = emb.register_output_hook(make_hook(i))
        hooks.append(h)

    try:
        for s in samples:
            msgs = [{"role": "user", "content": s["prompt"]}]
            chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            tokens = tok.encode(chat, return_tensors=None)
            token_ids = tokens if isinstance(tokens, list) else list(tokens)
            x = mx.array(token_ids)[None, :]
            _ = model(x)
            mx.eval(list(v for v in act_accum.values() if v is not None))
            n_tokens_total += 1
    finally:
        for h in hooks:
            h.remove()

    # Average
    for i in range(n_ple):
        if act_accum[i] is not None and n_tokens_total > 0:
            act_accum[i] = act_accum[i] / n_tokens_total
            mx.eval(act_accum[i])

    del model, tok
    gc.collect()
    mx.clear_cache()

    return act_accum


def apply_awq_scaling(
    bf16_weights: dict,
    act_stats: dict[int, mx.array],
    alpha: float,
) -> dict[str, mx.array]:
    """Apply AWQ-style scaling to PLE and per_layer_projection weights.

    For each PLE layer i:
      - Compute s = act_stats[i]^alpha, normalized
      - Scale PLE embedding: weight *= s (broadcast over vocab rows)
      - Compensate per_layer_projection: W[:, j] /= s[j]

    Returns a new dict with the scaled weights (still float32, not quantized).
    """
    scaled = dict(bf16_weights)  # Shallow copy — we'll replace specific tensors

    for layer_idx, act in act_stats.items():
        if act is None:
            continue

        # Compute scales
        s = mx.power(mx.maximum(act, mx.array(1e-6)), alpha)
        # Normalize: s = s / sqrt(s.max() * s.min())
        s_norm = mx.sqrt(mx.max(s) * mx.min(s))
        if s_norm.item() > 1e-8:
            s = s / s_norm
        mx.eval(s)

        # Find PLE weight key for this layer
        ple_key = None
        proj_key = None
        for k in bf16_weights:
            if f"embed_tokens_per_layer.{layer_idx}.weight" in k:
                ple_key = k
            if f"layers.{layer_idx}.per_layer_projection.weight" in k:
                proj_key = k

        if ple_key is not None:
            # PLE: (vocab, 256) — scale channels (axis=1)
            w_ple = bf16_weights[ple_key].astype(mx.float32)
            scaled[ple_key] = w_ple * s[None, :]  # broadcast s over vocab dim
            mx.eval(scaled[ple_key])

        if proj_key is not None:
            # per_layer_projection: (1536, 256) — divide input columns by s
            w_proj = bf16_weights[proj_key].astype(mx.float32)
            scaled[proj_key] = w_proj / s[None, :]  # column j divided by s[j]
            mx.eval(scaled[proj_key])

    return scaled


def export_awq_variant(
    scaled_weights: dict,
    out_path: str,
    ple_bits: int,
    ple_gs: int,
    other_bits: int,
    other_gs: int,
) -> float:
    """Quantize scaled weights and export model."""
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(OPTIMAL_PATH, out_path)

    for f in os.listdir(out_path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_path, f))

    quantized = {}
    for k in scaled_weights:
        if not k.endswith(".weight"):
            continue
        w = scaled_weights[k].astype(mx.float32)

        # Choose bits based on weight type
        if "embed_tokens_per_layer" in k:
            bits, gs = ple_bits, ple_gs
        else:
            bits, gs = other_bits, other_gs

        if w.ndim != 2 or w.shape[-1] < gs or w.shape[-1] % gs != 0:
            quantized[k] = scaled_weights[k]
            continue

        weight, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
        mx.eval(weight, scales, biases)
        base = k[: -len(".weight")]
        quantized[f"{base}.weight"] = weight
        quantized[f"{base}.scales"] = scales
        quantized[f"{base}.biases"] = biases

    # Copy non-weight tensors
    for k, v in scaled_weights.items():
        if k not in quantized and not any(k.endswith(s) for s in (".scales", ".biases")):
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            if f"{base}.scales" not in quantized:
                quantized[k] = v

    mx.eval(*list(quantized.values()))
    save_sharded(quantized, out_path)
    del quantized
    gc.collect()
    mx.clear_cache()

    # Patch config
    overrides = {"embed_tokens_per_layer": {"bits": ple_bits, "group_size": ple_gs}}
    patch_config_quantization(
        os.path.join(out_path, "config.json"),
        global_bits=other_bits, global_gs=other_gs,
        overrides=overrides,
    )

    return model_size_gb(out_path)


def main():
    parser = argparse.ArgumentParser(description="Step 4: AWQ-style PLE scaling")
    parser.add_argument("--target-bits", type=int, required=True,
                        help="PLE target bits (from Step 2 PLE floor)")
    parser.add_argument("--other-bits", type=int, required=True,
                        help="Non-PLE bits (uniform floor from Step 1)")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Alpha values to search")
    parser.add_argument("--cal-samples", type=int, default=16,
                        help="Number of calibration samples for activation capture")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    results_path = os.path.join(RESULTS_DIR, "awq_ple_results_e2b.json")

    print(f"{'#' * 70}")
    print(f"  Step 4: AWQ-for-PLE Channel Scaling (E2B)")
    print(f"  PLE target: {args.target_bits}-bit g{args.group_size}")
    print(f"  Non-PLE: {args.other_bits}-bit g{args.group_size}")
    print(f"  Alpha search: {args.alphas}")
    print(f"{'#' * 70}")

    # Collect activations from BF16 model
    print("\n  Collecting PLE activation statistics...")
    t0 = time.time()
    act_stats = collect_ple_activations(BF16_PATH, max_samples=args.cal_samples)
    print(f"  Collected in {time.time() - t0:.1f}s")

    # Print activation magnitude summary
    for i in sorted(act_stats.keys()):
        if act_stats[i] is not None:
            mean_act = act_stats[i].mean().item()
            max_act = mx.max(act_stats[i]).item()
            min_act = mx.min(act_stats[i]).item()
            if i < 5 or i >= N_LAYERS - 2:
                print(f"    Layer {i:2d}: mean={mean_act:.4f} min={min_act:.4f} max={max_act:.4f}")

    # Load BF16 weights once
    print("\n  Loading BF16 weights...")
    bf16_weights = load_bf16_weights()

    # Test each alpha
    for alpha in args.alphas:
        name = f"awq-alpha{alpha:.2f}-ple{args.target_bits}bit"
        out_path = os.path.join(TMP_DIR, name)
        aq_out = os.path.join(RESULTS_DIR, f"answer_quality_{name}.json")

        print(f"\n{'─' * 60}")
        print(f"  α = {alpha}")
        print(f"{'─' * 60}")

        # Apply scaling
        t0 = time.time()
        if alpha == 0.0:
            # No scaling — just quantize from BF16 directly
            scaled = bf16_weights
        else:
            scaled = apply_awq_scaling(bf16_weights, act_stats, alpha)
        print(f"  Scaling applied in {time.time() - t0:.1f}s")

        # Export
        t0 = time.time()
        size = export_awq_variant(
            scaled, out_path,
            ple_bits=args.target_bits, ple_gs=args.group_size,
            other_bits=args.other_bits, other_gs=args.group_size,
        )
        print(f"  Exported in {time.time() - t0:.1f}s — {size:.2f} GB")

        if alpha != 0.0:
            del scaled
            gc.collect()

        # Evaluate
        if not args.skip_eval:
            t0 = time.time()
            metrics = evaluate_variant(out_path)
            metrics["size_gb"] = size
            print(f"  Evaluated in {time.time() - t0:.1f}s")
            print_result(name, size, metrics)
        else:
            metrics = {"size_gb": size, "ppl": None, "exact_match_pct": None,
                       "avg_token_overlap": None, "n_samples": None, "exact_match": None}

        # AQ generation
        print(f"  Generating answer quality responses...")
        run_answer_quality(out_path, aq_out)

        entry = {
            "name": name, "alpha": alpha,
            "ple_bits": args.target_bits, "other_bits": args.other_bits,
            "group_size": args.group_size, **metrics,
        }
        log_result(results_path, entry)

        cleanup_variant(out_path)

    del bf16_weights
    gc.collect()
    mx.clear_cache()

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {results_path}")
    print(f"  Next: Score AQ files, pick best alpha, and optionally retry")
    print(f"  failed demotions from Step 2 with AWQ scaling applied.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
