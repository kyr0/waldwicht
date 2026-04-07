"""Sign-based quantizer with pluggable scale selection for ROTQ.

All scale methods produce the same packed format:
  - packed_bits: uint32, one sign bit per weight
  - mlx_scales = 2 * scale_per_group
  - mlx_biases = -scale_per_group

This is identical to the format in sign_quantize_gemma4.py and
compatible with MLX's affine QuantizedLinear / quantized_matmul kernel.

Scale methods:
  mean_abs    — s = mean(|w|) per group (baseline)
  median_abs  — s = median(|w|) per group (robust to outliers)
  percentile  — s = percentile(|w|, p) per group
  rms         — s = sqrt(mean(w²)) per group (RMS scale)
  output_aware — minimize ||X @ W - X @ W_q||² (requires activations)
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx


def sign_quantize(
    w: mx.array,
    group_size: int = 128,
    scale_method: str = "mean_abs",
    *,
    activations: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Sign-based 1-bit quantization with pluggable scale selection.

    Args:
        w: (out_features, in_features) weight matrix.
        group_size: Number of weights per group.
        scale_method: Scale selection method.
        activations: (..., in_features) calibration activations.
            Required for "output_aware" method.

    Returns:
        (packed_bits, mlx_scales, mlx_biases) in MLX quantized format.
    """
    rows, cols = w.shape
    assert cols % group_size == 0, f"cols {cols} not divisible by group_size {group_size}"

    w_f = w.astype(mx.float32)
    n_groups = cols // group_size
    w_grouped = w_f.reshape(rows, n_groups, group_size)

    # Sign bits: 1 for non-negative, 0 for negative
    sign_bits = (w_grouped >= 0).astype(mx.uint32)

    # Compute scale per group
    scale = _compute_scale(w_grouped, scale_method, activations=activations)

    # MLX format: dequantized = mlx_scale * bit + mlx_bias
    # For sign quant: bit=1 → +scale, bit=0 → -scale
    # So: mlx_scale = 2*scale, mlx_bias = -scale
    mlx_scales = (2 * scale).astype(mx.float16)  # (rows, n_groups)
    mlx_biases = (-scale).astype(mx.float16)  # (rows, n_groups)

    # Pack 32 sign bits into each uint32
    packed = _pack_sign_bits(sign_bits, group_size)

    return packed, mlx_scales, mlx_biases


def _compute_scale(
    w_grouped: mx.array,
    method: str,
    *,
    activations: Optional[mx.array] = None,
) -> mx.array:
    """Compute per-group scale factor.

    Args:
        w_grouped: (rows, n_groups, group_size) weight groups.
        method: Scale selection method.
        activations: Optional calibration activations.

    Returns:
        (rows, n_groups) scale factors.
    """
    if method == "mean_abs":
        return mx.mean(mx.abs(w_grouped), axis=2)

    elif method == "median_abs":
        # Median of absolute values — robust to outliers
        abs_w = mx.abs(w_grouped)
        # MLX doesn't have median, use sorted middle element
        sorted_abs = mx.sort(abs_w, axis=2)
        mid = w_grouped.shape[2] // 2
        return sorted_abs[:, :, mid]

    elif method == "percentile":
        # 75th percentile of absolute values
        abs_w = mx.abs(w_grouped)
        sorted_abs = mx.sort(abs_w, axis=2)
        p75_idx = int(w_grouped.shape[2] * 0.75)
        return sorted_abs[:, :, p75_idx]

    elif method == "rms":
        # Root mean square
        return mx.sqrt(mx.mean(w_grouped * w_grouped, axis=2))

    elif method == "output_aware":
        # Minimize ||X @ W - X @ W_q||² per group
        # This reduces to: s* = (X @ |W| . sign(W)) / (X @ 1)
        # Simplified: use activation magnitudes as importance weights
        if activations is None:
            raise ValueError("output_aware scale requires activations")
        return _output_aware_scale(w_grouped, activations)

    else:
        raise ValueError(f"Unknown scale method: {method}")


def _output_aware_scale(
    w_grouped: mx.array,
    activations: mx.array,
) -> mx.array:
    """Activation-weighted scale for sign quantization.

    For sign quantization W_q = s * sign(W), the output-aware optimal
    scale per group minimizes:
        sum_x ||x @ W_g - x @ (s * sign(W_g))||²

    This has closed-form solution per group g:
        s_g = sum_x sum_j |x_j * w_j| / sum_x sum_j x_j²

    Where j indexes within the group. We approximate using calibration
    activation statistics.

    Args:
        w_grouped: (rows, n_groups, group_size) weight groups.
        activations: (..., in_features) calibration activations.

    Returns:
        (rows, n_groups) output-aware scales.
    """
    rows, n_groups, group_size = w_grouped.shape

    # Compute per-channel activation energy
    a_f = activations.astype(mx.float32).reshape(-1, activations.shape[-1])
    # Mean squared activation per channel: (in_features,)
    act_energy = mx.mean(a_f * a_f, axis=0)

    # Reshape to match groups
    in_features = n_groups * group_size
    act_energy_grouped = act_energy[:in_features].reshape(n_groups, group_size)

    # Weighted absolute value of weights
    abs_w = mx.abs(w_grouped)  # (rows, n_groups, group_size)
    weighted_abs = abs_w * act_energy_grouped[None, :, :]  # broadcast over rows

    # Scale = sum(weighted_abs) / sum(act_energy) per group
    numerator = mx.sum(weighted_abs, axis=2)  # (rows, n_groups)
    denominator = mx.sum(act_energy_grouped, axis=1, keepdims=True) + 1e-12  # (1, n_groups) → broadcast
    denominator = mx.broadcast_to(denominator, numerator.shape)

    return numerator / denominator


def _pack_sign_bits(
    sign_bits: mx.array,
    group_size: int,
) -> mx.array:
    """Pack sign bits into uint32 words.

    Args:
        sign_bits: (rows, n_groups, group_size) uint32 with values 0 or 1.
        group_size: Number of bits per group.

    Returns:
        (rows, n_groups * ints_per_group) packed uint32 array.
    """
    rows, n_groups, gs = sign_bits.shape
    el_per_int = 32
    ints_per_group = gs // el_per_int

    # Reshape: (rows, n_groups, ints_per_group, 32)
    bits = sign_bits.reshape(rows, n_groups, ints_per_group, el_per_int)

    # Pack: bit 0 is LSB
    shifts = mx.arange(el_per_int, dtype=mx.uint32).reshape(1, 1, 1, el_per_int)
    packed = mx.sum(bits << shifts, axis=3).astype(mx.uint32)

    # Flatten groups: (rows, n_groups * ints_per_group)
    return packed.reshape(rows, n_groups * ints_per_group)


def dequantize_sign(
    packed: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 128,
) -> mx.array:
    """Dequantize sign-quantized weights (for offline evaluation).

    Uses MLX's built-in dequantize which handles the affine format.
    """
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=1)
