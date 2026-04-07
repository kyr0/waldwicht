"""ROTQ pipeline — foldable geometry optimization for N-bit affine export.

Takes a BF16 model and produces a quantized model with NO runtime
transforms.  All geometry optimization (equalization, permutation,
rotation) is folded back into the weight matrices before export.

Supports any bit-width from 1 to 8.  At bits=1 uses sign quantization;
at bits>=2 uses standard MLX affine quantization (mx.quantize).

The output loads with standard mlx_lm.load() using QuantizedLinear.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import module_policy as mp
from . import quantizer
from .transforms import equalization, permutation, rotation


# ── Quantizer helpers ────────────────────────────────────────────────────────


def _make_quantize_fn(bits: int, group_size: int, scale_method: str = "mean_abs"):
    """Create a quantize function matching the sign_quantize signature.

    Returns a callable (w, group_size) → (packed, scales, biases).
    At bits=1 this is sign_quantize; at bits>=2 this is mx.quantize.
    """
    if bits == 1:
        def fn(w, gs, *args, **kwargs):
            return quantizer.sign_quantize(w, gs, scale_method)
        return fn, bits
    else:
        def fn(w, gs, *args, **kwargs):
            return mx.quantize(w, group_size=gs, bits=bits)
        return fn, bits


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class RotqConfig:
    """Configuration for the ROTQ pipeline."""

    # Transform toggles
    equalization: bool = True
    permutation: bool = True
    rotation: bool = True

    # Transform parameters
    equalization_method: str = "abs_max"
    permutation_methods: list[str] = field(
        default_factory=lambda: ["identity", "magnitude", "abs_mean", "variance"]
    )
    rotation_families: list[str] = field(
        default_factory=lambda: ["block_hadamard", "signed_hadamard"]
    )
    rotation_candidates: int = 8
    rotation_block_size: int = 128

    # Quantizer
    scale_method: str = "mean_abs"
    group_size: int = 128
    quantize_bits: int = 1  # Target bits for main weights (1=sign, 2-8=affine)

    # Affine fallback (for embeddings / edge layers)
    affine_bits: int = 4
    affine_group_size: int = 64

    # Module policies (if None, uses defaults)
    policies: Optional[dict] = None

    # Processing
    seed: int = 42
    verbose: bool = True


# ── Core pipeline ────────────────────────────────────────────────────────────


def rotq_process_weights(
    weights: dict[str, mx.array],
    config: RotqConfig,
) -> dict[str, mx.array]:
    """Apply ROTQ geometry optimization and sign quantization to weights.

    This is the main processing function.  It operates on a flat weight
    dictionary and returns a quantized weight dictionary.

    Steps:
      1. Classify each weight (sign_q1 / affine4 / skip)
      2. Apply equalization to identified layer pairs
      3. For each sign_q1 weight:
         a. Search for best permutation
         b. Search for best rotation (within foldable subspace)
         c. Apply transforms and sign-quantize
      4. For affine4 weights: standard MLX quantize
      5. Return quantized weight dict

    Args:
        weights: Name→tensor weight dictionary (BF16/FP32).
        config: ROTQ configuration.

    Returns:
        Quantized weight dictionary in MLX format.
    """
    policies = config.policies or mp.default_gemma4_policies()
    out_weights = {}
    affine4_bases = set()  # Track which weights got affine4 treatment
    rotated_bases = {}     # base → {"family", "seed", "block_size"}
    stats = {"n_quantized": 0, "n_affine4": 0, "n_skip": 0}
    t0 = time.time()

    # Build the quantize function for the target bit-width
    quantize_fn, qbits = _make_quantize_fn(
        config.quantize_bits, config.group_size, config.scale_method
    )

    # ── Step 1: Equalization ──────────────────────────────────────────────
    if config.equalization:
        eq_pairs = mp.identify_equalization_pairs()
        if config.verbose:
            print(f"[ROTQ] Equalizing {len(eq_pairs)} layer pairs...")

        # For MLP intermediate equalization, gate/up rows ↔ down columns
        # gate_proj shape: (intermediate, hidden), down_proj shape: (hidden, intermediate)
        # Shared dimension is intermediate.
        # Equalize: gate_proj rows (output) ↔ down_proj columns (input)
        # This means: scale gate_proj rows by D, scale down_proj columns by 1/D
        for prod_key, cons_key in eq_pairs:
            if prod_key not in weights or cons_key not in weights:
                continue
            w_p = weights[prod_key]
            w_c = weights[cons_key]
            # Producer output dim = w_p rows, consumer input dim = w_c columns
            # For equalization, we need the shared dimension to be columns of both
            # gate_proj: (intermediate, hidden) — rows are intermediate
            # down_proj: (hidden, intermediate) — columns are intermediate
            # We need to treat gate rows and down columns as the shared space
            # Transpose producer so shared dim is columns for both
            if w_p.shape[0] == w_c.shape[1]:
                # Producer rows == consumer columns → shared is producer output = consumer input
                # Equalize in this shared space
                D = equalization.compute_channel_scales(
                    w_p.T, w_c, method=config.equalization_method
                )
                mx.eval(D)
                # Scale producer rows (output) by D: multiply each row i by D[i]
                weights[prod_key] = w_p.astype(mx.float32) * D[:, None]
                # Scale consumer columns (input) by 1/D
                weights[cons_key] = w_c.astype(mx.float32) / D[None, :]
                mx.eval(weights[prod_key], weights[cons_key])

    # ── Step 2: Permutation search for shared subspaces ───────────────────
    perm_groups = mp.identify_permutation_groups() if config.permutation else []
    applied_perms = {}

    if config.permutation and perm_groups:
        if config.verbose:
            print(f"[ROTQ] Searching permutations for {len(perm_groups)} subspaces...")

        for pg in perm_groups:
            # Use the first column_key weight to search for best permutation
            col_keys = pg["column_keys"]
            row_keys = pg["row_keys"]

            ref_key = col_keys[0] if col_keys else None
            if ref_key is None or ref_key not in weights:
                continue

            w_ref = weights[ref_key]
            policy = mp.get_policy_for_weight(ref_key, w_ref, policies)

            if policy.action != "sign_q1" or not policy.permutation:
                continue

            # Search for best permutation on the shared dimension (columns of ref)
            best_perm, best_method, best_mse = permutation.find_best_permutation(
                w_ref,
                config.group_size,
                quantizer.sign_quantize,
                methods=config.permutation_methods,
            )
            mx.eval(best_perm)

            if config.verbose:
                layer = pg.get("layer", "?")
                print(f"  Layer {layer} {pg['space']}: best={best_method} mse={best_mse:.4f}")

            # Apply permutation to all matrices sharing this subspace
            inv_perm = permutation.inverse_permutation(best_perm)

            # Column keys: permute columns
            for k in col_keys:
                if k in weights:
                    weights[k] = permutation.apply_input_permutation(weights[k], best_perm)
                    mx.eval(weights[k])

            # Row keys: permute rows
            for k in row_keys:
                if k in weights:
                    weights[k] = permutation.apply_output_permutation(weights[k], best_perm)
                    mx.eval(weights[k])

            applied_perms[f"{pg['space']}_{pg.get('layer', 0)}"] = best_method

    # ── Step 3: Per-weight rotation + quantization ────────────────────────
    if config.verbose:
        print("[ROTQ] Quantizing weights...")

    for key in sorted(weights.keys()):
        W = weights[key]
        policy = mp.get_policy_for_weight(key, W, policies)
        base = key[:-len(".weight")] if key.endswith(".weight") else key

        if policy.action == "skip":
            out_weights[key] = W
            stats["n_skip"] += 1
            continue

        if policy.action == "affine4":
            # Standard affine quantization
            W_f = W.astype(mx.float32)
            weight, scales, biases = mx.quantize(
                W_f, group_size=config.affine_group_size, bits=config.affine_bits
            )
            mx.eval(weight, scales, biases)
            out_weights[f"{base}.weight"] = weight
            out_weights[f"{base}.scales"] = scales
            out_weights[f"{base}.biases"] = biases
            affine4_bases.add(base)
            stats["n_affine4"] += 1
            continue

        # action == "sign_q1" (or quantize at target bits)
        W_f = W.astype(mx.float32)

        # Optional rotation search (within the weight's own input space)
        rot_family = "identity"
        if config.rotation and policy.rotation:
            rotate_fn, inverse_fn, family, seed, rot_mse = rotation.search_rotation(
                W_f,
                config.group_size,
                quantize_fn,
                families=policy.rotation_families,
                n_candidates=policy.rotation_candidates,
                block_size=policy.rotation_block_size,
                base_seed=config.seed,
                quantize_bits=qbits,
            )
            rot_family = family
            # Apply rotation (this is folded INTO the weight)
            W_f = rotation.apply_rotation_to_weight(W_f, rotate_fn, axis="input")
            mx.eval(W_f)

            if config.verbose and family != "identity":
                print(f"  {key}: rotation={family} seed={seed} mse={rot_mse:.4f}")

        # Quantize at target bit-width
        packed, scales, biases = quantize_fn(W_f, config.group_size)
        mx.eval(packed, scales, biases)

        out_weights[f"{base}.weight"] = packed
        out_weights[f"{base}.scales"] = scales
        out_weights[f"{base}.biases"] = biases

        # Store rotation signs for RotatedLinear inference
        if rot_family != "identity" and rot_family.endswith("hadamard"):
            bs = policy.rotation_block_size or config.rotation_block_size
            signs = rotation.random_signs(bs, seed=seed)
            mx.eval(signs)
            out_weights[f"{base}.signs"] = signs
            rotated_bases[base] = {
                "family": rot_family,
                "seed": seed,
                "block_size": bs,
            }

        stats["n_quantized"] += 1

        if config.verbose and stats["n_quantized"] % 20 == 0:
            elapsed = time.time() - t0
            bits_label = f"q{qbits}"
            print(f"  [{stats['n_quantized']:3d} {bits_label}] {key} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    if config.verbose:
        print(f"\n[ROTQ] Done in {elapsed:.1f}s:")
        print(f"  q{qbits}: {stats['n_quantized']}")
        print(f"  affine{config.affine_bits}: {stats['n_affine4']}")
        print(f"  skip:    {stats['n_skip']}")
        if applied_perms:
            print(f"  permutations applied: {len(applied_perms)}")
        if rotated_bases:
            print(f"  rotations (RotatedLinear): {len(rotated_bases)}")

    return out_weights, affine4_bases, rotated_bases


# ── Export pipeline ──────────────────────────────────────────────────────────


def rotq_export(
    model_path: str,
    output_path: str,
    config: RotqConfig,
):
    """Full ROTQ pipeline: load BF16 model → transform → quantize → export.

    The output model uses plain Q1 format identical to sign_quantize_gemma4.py.

    Args:
        model_path: Path to BF16 MLX model directory.
        output_path: Output directory for quantized model.
        config: ROTQ configuration.
    """
    model_dir = Path(model_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load weights ──────────────────────────────────────────────────────
    shard_files = sorted(glob.glob(str(model_dir / "model*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No model*.safetensors in {model_dir}")

    if config.verbose:
        print(f"[ROTQ] Loading {len(shard_files)} shard(s) from {model_dir}...")

    all_weights = {}
    for sf in shard_files:
        all_weights.update(mx.load(sf))

    if config.verbose:
        print(f"  Loaded {len(all_weights)} tensors.")

    # ── Process ───────────────────────────────────────────────────────────
    out_weights, affine4_bases, rotated_bases = rotq_process_weights(all_weights, config)

    # ── Strip multimodal components (text-only export) ──────────────────
    _MULTIMODAL_PREFIXES = (
        "model.vision_tower", "model.audio_tower",
        "model.multi_modal_projector",
        "model.embed_audio", "model.embed_vision",
        "vision_tower", "audio_tower",
        "multi_modal_projector",
        "embed_audio", "embed_vision",
    )
    n_before = len(out_weights)
    out_weights = {
        k: v for k, v in out_weights.items()
        if not k.startswith(_MULTIMODAL_PREFIXES)
    }
    n_stripped = n_before - len(out_weights)
    if n_stripped and config.verbose:
        print(f"[ROTQ] Stripped {n_stripped} multimodal tensors (text-only export).")

    # ── Save weights ──────────────────────────────────────────────────────
    if config.verbose:
        print(f"[ROTQ] Saving to {out_dir}...")

    _save_sharded(out_weights, out_dir, max_shard_bytes=2 * 1024**3)

    # ── Write config ──────────────────────────────────────────────────────
    with open(model_dir / "config.json") as f:
        model_config = json.load(f)

    # Quantization config with the correct target bits
    if rotated_bases:
        # Use rotation_quantization path: RotatedLinear for rotated layers,
        # standard QuantizedLinear for affine4 layers (inferred from weights).
        # Do NOT set "quantization" — that would convert rotated layers to
        # QuantizedLinear before RotatedLinear can replace them.
        model_config["rotation_quantization"] = {
            "bits": config.quantize_bits,
            "group_size": config.group_size,
        }
    else:
        # No rotated layers — plain quantization path
        quant_cfg = {
            "group_size": config.group_size,
            "bits": config.quantize_bits,
            "mode": "affine",
        }
        # Add per-layer overrides for affine4 layers.
        # Keys must match the model's parameter paths (post-sanitize),
        # not the HF weight keys.  Gemma 4 sanitize:
        #   "model.language_model.X" → "language_model.model.X"
        #   embed_tokens_per_layer (stacked) → .0, .1, ... (split)
        n_layers = model_config.get("text_config", {}).get(
            "num_hidden_layers", None
        )
        override = {
            "group_size": config.affine_group_size,
            "bits": config.affine_bits,
        }
        for base in sorted(affine4_bases):
            key = base
            if key.startswith("model."):
                key = key[len("model."):]
            if key.startswith("language_model.") and not key.startswith(
                "language_model.model."
            ):
                key = "language_model.model." + key[len("language_model."):]
            # Expand stacked PLE key to per-layer keys (sanitize splits it)
            if "embed_tokens_per_layer" in key and n_layers and not any(
                c.isdigit() for c in key.split("embed_tokens_per_layer")[-1]
            ):
                for i in range(n_layers):
                    quant_cfg[f"{key}.{i}"] = override
            else:
                quant_cfg[key] = override
        model_config["quantization"] = quant_cfg

    if not rotated_bases:
        model_config["quantization_config"] = {
            "group_size": config.group_size,
            "bits": config.quantize_bits,
            "mode": "affine",
        }

    # Add ROTQ metadata (informational only, not used by loader)
    model_config["rotq_config"] = {
        "equalization": config.equalization,
        "equalization_method": config.equalization_method,
        "permutation": config.permutation,
        "rotation": config.rotation,
        "rotation_families": config.rotation_families,
        "scale_method": config.scale_method,
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # ── Copy tokenizer and model files ────────────────────────────────────
    for fname in model_dir.iterdir():
        if fname.name.startswith("tokenizer") or fname.name in (
            "special_tokens_map.json",
            "gemma4.py",
            "chat_template.jinja",
            "generation_config.json",
        ):
            shutil.copy2(fname, out_dir / fname.name)

    # Copy model file if present
    model_file = model_config.get("model_file")
    if model_file and (model_dir / model_file).exists():
        shutil.copy2(model_dir / model_file, out_dir / model_file)

    if config.verbose:
        print(f"[ROTQ] Export complete → {out_dir}")


def _save_sharded(
    weights: dict[str, mx.array],
    out_dir: Path,
    max_shard_bytes: int = 2 * 1024**3,
):
    """Save weights as sharded safetensors files."""
    try:
        from mlx.utils import save_safetensors
    except ImportError:
        mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
        return

    # Estimate sizes and split into shards
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted(weights.keys()):
        w = weights[key]
        w_bytes = w.size * w.dtype.size
        if current_size + w_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = w
        current_size += w_bytes

    if current_shard:
        shards.append(current_shard)

    # Save each shard
    index_map = {}
    shard_files = []

    if len(shards) == 1:
        fname = "model.safetensors"
        mx.save_safetensors(str(out_dir / fname), shards[0])
        shard_files.append(fname)
        for k in shards[0]:
            index_map[k] = fname
    else:
        for i, shard in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            mx.save_safetensors(str(out_dir / fname), shard)
            shard_files.append(fname)
            for k in shard:
                index_map[k] = fname

        # Write index file
        index = {
            "metadata": {"total_size": sum(w.size * w.dtype.size for w in weights.values())},
            "weight_map": index_map,
        }
        with open(out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
