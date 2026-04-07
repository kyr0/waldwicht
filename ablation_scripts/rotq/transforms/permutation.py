"""Channel permutation search for ROTQ.

Reorders hidden channels so that channels with similar magnitude /
sensitivity / statistical profile fall into the same contiguous g128
groups.  This is a function-preserving transform: permuting input
dimensions of one matrix and output dimensions of the preceding matrix
leaves the composed function unchanged.

Key insight from synthetic experiments:
  - Magnitude-sorted permutation reduced grouped sign MSE from 0.56 to 0.32
    (43% reduction) — one of the highest-ROI single operations.
"""
from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx


def magnitude_sort_permutation(w: mx.array, group_size: int = 128) -> mx.array:
    """Sort channels by column L2 norm.

    Channels with similar magnitude end up in the same group, reducing
    within-group dynamic range.

    Args:
        w: (out_features, in_features) weight matrix.
        group_size: Quantization group size.

    Returns:
        (in_features,) permutation index array.
    """
    w_f = w.astype(mx.float32)
    col_norms = mx.sqrt(mx.sum(w_f * w_f, axis=0))  # (in_features,)
    perm = mx.argsort(col_norms)
    return perm


def abs_mean_sort_permutation(w: mx.array, group_size: int = 128) -> mx.array:
    """Sort channels by column mean absolute value.

    Args:
        w: (out_features, in_features) weight matrix.
        group_size: Quantization group size.

    Returns:
        (in_features,) permutation index array.
    """
    w_f = w.astype(mx.float32)
    col_abs_mean = mx.mean(mx.abs(w_f), axis=0)
    return mx.argsort(col_abs_mean)


def variance_sort_permutation(w: mx.array, group_size: int = 128) -> mx.array:
    """Sort channels by column variance.

    Groups channels with similar variance together, so group scales
    are more uniform.

    Args:
        w: (out_features, in_features) weight matrix.
        group_size: Quantization group size.

    Returns:
        (in_features,) permutation index array.
    """
    w_f = w.astype(mx.float32)
    col_var = mx.var(w_f, axis=0)
    return mx.argsort(col_var)


def sensitivity_sort_permutation(
    w: mx.array,
    activations: mx.array,
    group_size: int = 128,
) -> mx.array:
    """Sort channels by activation-weighted sensitivity.

    Sensitivity per channel j = mean(|a_j|) * mean(|w_j|), reflecting
    how much each channel contributes to the output in practice.

    Args:
        w: (out_features, in_features) weight matrix.
        activations: (..., in_features) activation tensor from calibration.
        group_size: Quantization group size.

    Returns:
        (in_features,) permutation index array.
    """
    w_f = w.astype(mx.float32)
    a_f = activations.astype(mx.float32).reshape(-1, activations.shape[-1])

    weight_mag = mx.mean(mx.abs(w_f), axis=0)  # (in_features,)
    act_mag = mx.mean(mx.abs(a_f), axis=0)  # (in_features,)
    sensitivity = weight_mag * act_mag

    return mx.argsort(sensitivity)


def interleave_permutation(base_perm: mx.array, group_size: int = 128) -> mx.array:
    """Interleave a sorted permutation to balance within-group variance.

    Given a sorted permutation [small...large], interleave to create
    groups where each group contains a mix of magnitudes but neighboring
    groups have similar overall scale.

    Strategy: assign element i to group (i % n_groups), then sort within
    each group. This ensures each group gets elements from across the
    magnitude spectrum.

    Args:
        base_perm: (dim,) sorted permutation from magnitude_sort or similar.
        group_size: Quantization group size.

    Returns:
        (dim,) interleaved permutation.
    """
    dim = base_perm.shape[0]
    n_groups = dim // group_size

    # Work in Python for correctness (permutation is done once offline)
    perm_list = base_perm.tolist()
    groups = [[] for _ in range(n_groups)]

    for i, idx in enumerate(perm_list):
        groups[i % n_groups].append(idx)

    # Flatten back
    result = []
    for g in groups:
        result.extend(g)

    # Handle remainder (if dim not divisible by group_size)
    remainder = dim - n_groups * group_size
    if remainder > 0:
        result.extend(perm_list[n_groups * group_size:])

    return mx.array(result, dtype=mx.int32)


def apply_input_permutation(w: mx.array, perm: mx.array) -> mx.array:
    """Permute input (column) dimensions of a weight matrix.

    Args:
        w: (out_features, in_features) weight matrix.
        perm: (in_features,) permutation indices.

    Returns:
        (out_features, in_features) permuted weight matrix.
    """
    return w[:, perm]


def apply_output_permutation(w: mx.array, perm: mx.array) -> mx.array:
    """Permute output (row) dimensions of a weight matrix.

    Args:
        w: (out_features, in_features) weight matrix.
        perm: (out_features,) permutation indices.

    Returns:
        (out_features, in_features) permuted weight matrix.
    """
    return w[perm, :]


def inverse_permutation(perm: mx.array) -> mx.array:
    """Compute the inverse of a permutation.

    If perm[i] = j, then inv_perm[j] = i.

    Args:
        perm: (n,) permutation array.

    Returns:
        (n,) inverse permutation array.
    """
    n = perm.shape[0]
    inv = mx.zeros((n,), dtype=mx.int32)
    # MLX scatter: inv[perm[i]] = i
    indices = perm.astype(mx.int32)
    values = mx.arange(n, dtype=mx.int32)
    # Use Python for correctness (one-time offline operation)
    perm_list = perm.tolist()
    inv_list = [0] * n
    for i, p in enumerate(perm_list):
        inv_list[int(p)] = i
    return mx.array(inv_list, dtype=mx.int32)


def find_best_permutation(
    w: mx.array,
    group_size: int,
    sign_quantize_fn: Callable[[mx.array, int], tuple],
    *,
    activations: Optional[mx.array] = None,
    methods: list[str] | None = None,
) -> tuple[mx.array, str, float]:
    """Try multiple permutation strategies and return the best.

    Args:
        w: (out_features, in_features) weight matrix.
        group_size: Quantization group size.
        sign_quantize_fn: Function(w, group_size) → (packed, scales, biases).
            Used to evaluate quantization quality.
        activations: Optional activation data for sensitivity-based methods.
        methods: Subset of methods to try. Default: all applicable methods.

    Returns:
        (best_perm, best_method, best_mse) — the best permutation found.
    """
    from ..metrics import relative_mse as _rel_mse

    if methods is None:
        methods = ["identity", "magnitude", "abs_mean", "variance"]
        if activations is not None:
            methods.append("sensitivity")

    in_features = w.shape[1]
    best_perm = mx.arange(in_features, dtype=mx.int32)
    best_method = "identity"
    best_mse = float("inf")

    strategy_map = {
        "identity": lambda: mx.arange(in_features, dtype=mx.int32),
        "magnitude": lambda: magnitude_sort_permutation(w, group_size),
        "abs_mean": lambda: abs_mean_sort_permutation(w, group_size),
        "variance": lambda: variance_sort_permutation(w, group_size),
    }
    if activations is not None:
        strategy_map["sensitivity"] = lambda: sensitivity_sort_permutation(
            w, activations, group_size
        )

    for name in methods:
        if name not in strategy_map:
            continue

        perm = strategy_map[name]()
        mx.eval(perm)

        # Apply permutation and quantize
        w_perm = apply_input_permutation(w, perm)
        packed, scales, biases = sign_quantize_fn(w_perm, group_size)

        # Dequantize to measure error
        w_deq = mx.dequantize(packed, scales, biases, group_size=group_size, bits=1)
        mx.eval(w_deq)

        # Inverse permute the dequantized weights for fair comparison
        inv_perm = inverse_permutation(perm)
        w_deq_orig_order = apply_input_permutation(w_deq, inv_perm)

        mse = _rel_mse(w, w_deq_orig_order)

        if mse < best_mse:
            best_mse = mse
            best_perm = perm
            best_method = name

    return best_perm, best_method, best_mse
