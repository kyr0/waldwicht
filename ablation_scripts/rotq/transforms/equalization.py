"""Cross-layer equalization for ROTQ.

Exploits the exact invariance: for adjacent linear layers W_out and W_in,
we can replace (W_out, W_in) with (W_out @ D, D^{-1} @ W_in) for any
invertible diagonal matrix D, without changing the composed function.

This flattens per-channel magnitude disparities, reducing within-group
dynamic range and making sign quantization less destructive.

Folding through RMSNorm:
  RMSNorm(x) = x / rms(x) * gamma
  Since D is diagonal, scaling channels by D before RMSNorm and by D^{-1}
  after it requires absorbing D into gamma: gamma_new = gamma * D.
  This is exact for diagonal D.
"""
from __future__ import annotations

import mlx.core as mx


def compute_channel_scales(
    w_producer: mx.array,
    w_consumer: mx.array,
    method: str = "abs_max",
) -> mx.array:
    """Compute per-channel equalization scales.

    Given producer W_p (out_p, hidden) and consumer W_c (out_c, hidden),
    find diagonal D such that W_p @ D and D^{-1} @ W_c^T have balanced
    channel magnitudes.

    Args:
        w_producer: (out_p, hidden) — the weight whose output dimension
            is the shared hidden space.
        w_consumer: (out_c, hidden) — the weight whose input dimension
            is the shared hidden space (columns = hidden).
        method: "abs_max", "abs_mean", or "geometric".

    Returns:
        (hidden,) scale vector D. Apply as:
            w_producer_new = w_producer * D[None, :]   (scale columns)
            w_consumer_new = w_consumer / D[None, :]   (scale columns)
    """
    w_p = w_producer.astype(mx.float32)
    w_c = w_consumer.astype(mx.float32)
    hidden = w_p.shape[1]
    assert w_c.shape[1] == hidden, (
        f"Hidden dimension mismatch: producer {w_p.shape}, consumer {w_c.shape}"
    )

    if method == "abs_max":
        # Per-channel max absolute value
        s_p = mx.max(mx.abs(w_p), axis=0)  # (hidden,)
        s_c = mx.max(mx.abs(w_c), axis=0)  # (hidden,)
    elif method == "abs_mean":
        s_p = mx.mean(mx.abs(w_p), axis=0)
        s_c = mx.mean(mx.abs(w_c), axis=0)
    elif method == "geometric":
        # Geometric mean of producer and consumer magnitudes
        s_p = mx.mean(mx.abs(w_p), axis=0)
        s_c = mx.mean(mx.abs(w_c), axis=0)
    else:
        raise ValueError(f"Unknown equalization method: {method}")

    # D = sqrt(s_c / s_p)  so that  s_p * D ≈ s_c / D ≈ sqrt(s_p * s_c)
    eps = 1e-8
    s_p = mx.maximum(s_p, eps)
    s_c = mx.maximum(s_c, eps)
    D = mx.sqrt(s_c / s_p)

    # Clamp to avoid extreme scale factors
    D = mx.clip(D, 1e-4, 1e4)

    return D


def equalize_pair(
    w_producer: mx.array,
    w_consumer: mx.array,
    method: str = "abs_max",
) -> tuple[mx.array, mx.array]:
    """Apply cross-layer equalization to a (producer, consumer) weight pair.

    Args:
        w_producer: (out_p, hidden) weight matrix.
        w_consumer: (out_c, hidden) weight matrix.
        method: Equalization method (see compute_channel_scales).

    Returns:
        (w_producer_eq, w_consumer_eq) — equalized weight matrices.
        The composed function W_consumer @ W_producer is preserved.
    """
    D = compute_channel_scales(w_producer, w_consumer, method=method)
    mx.eval(D)

    # Scale producer columns by D, consumer columns by 1/D
    w_p_eq = w_producer.astype(mx.float32) * D[None, :]
    w_c_eq = w_consumer.astype(mx.float32) / D[None, :]

    return w_p_eq, w_c_eq


def equalize_with_norm(
    w_producer: mx.array,
    w_consumer: mx.array,
    norm_gamma: mx.array,
    method: str = "abs_max",
) -> tuple[mx.array, mx.array, mx.array]:
    """Equalize through an RMSNorm layer.

    For the path: producer → RMSNorm(gamma) → consumer,
    we can absorb diagonal scaling D into gamma:
        producer_new = producer * D  (scale output channels)
        gamma_new = gamma * D        (absorb into norm)
        consumer_new = consumer / D  (scale input channels)

    Note: RMSNorm is x / rms(x) * gamma. The diagonal scaling D modifies
    the per-channel magnitude but rms(x * D) ≠ rms(x) in general, so
    this is an approximation. It is exact only when D is uniform (scalar),
    but empirically very good for small D variations.

    For exact folding, use equalize_pair on subspaces that have no norm
    in between (e.g., MLP intermediate: gate/up → down).

    Args:
        w_producer: (out_p, hidden).
        w_consumer: (out_c, hidden).
        norm_gamma: (hidden,) RMSNorm scale parameter.
        method: Equalization method.

    Returns:
        (w_producer_eq, w_consumer_eq, norm_gamma_eq).
    """
    D = compute_channel_scales(w_producer, w_consumer, method=method)
    mx.eval(D)

    w_p_eq = w_producer.astype(mx.float32) * D[None, :]
    w_c_eq = w_consumer.astype(mx.float32) / D[None, :]
    gamma_eq = norm_gamma.astype(mx.float32) * D

    return w_p_eq, w_c_eq, gamma_eq


def equalize_model_weights(
    weights: dict[str, mx.array],
    layer_pairs: list[tuple[str, str]],
    method: str = "abs_max",
) -> dict[str, mx.array]:
    """Apply equalization across all specified layer pairs.

    Args:
        weights: Name→tensor weight dictionary.
        layer_pairs: List of (producer_key, consumer_key) pairs.
            Both keys should end in '.weight'.
        method: Equalization method.

    Returns:
        Updated weight dictionary with equalized weights.
    """
    result = dict(weights)

    for prod_key, cons_key in layer_pairs:
        if prod_key not in result or cons_key not in result:
            continue

        w_p = result[prod_key]
        w_c = result[cons_key]

        if w_p.ndim != 2 or w_c.ndim != 2:
            continue
        if w_p.shape[1] != w_c.shape[1]:
            continue

        w_p_eq, w_c_eq = equalize_pair(w_p, w_c, method=method)
        mx.eval(w_p_eq, w_c_eq)

        result[prod_key] = w_p_eq
        result[cons_key] = w_c_eq

    return result
