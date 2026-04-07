"""Quantization quality metrics for ROTQ evaluation.

Three levels of measurement:
  1. Weight-domain: MSE, sign agreement, group statistics
  2. Layer-domain: hidden-state and logit comparison
  3. End-to-end: perplexity, generation sanity
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# ── Weight-domain metrics ────────────────────────────────────────────────────


def relative_mse(w_orig: mx.array, w_quant: mx.array) -> float:
    """Relative mean squared error: MSE(w_orig, w_quant) / MSE(w_orig, 0)."""
    diff = (w_orig - w_quant).astype(mx.float32)
    mse = mx.mean(diff * diff).item()
    ref = mx.mean(w_orig.astype(mx.float32) ** 2).item()
    return mse / max(ref, 1e-12)


def sign_agreement(w_orig: mx.array, w_quant: mx.array) -> float:
    """Fraction of elements where sign(w_orig) == sign(w_quant)."""
    s_orig = (w_orig >= 0).astype(mx.int32)
    s_quant = (w_quant >= 0).astype(mx.int32)
    return mx.mean((s_orig == s_quant).astype(mx.float32)).item()


def mean_abs_error(w_orig: mx.array, w_quant: mx.array) -> float:
    """Mean absolute error."""
    return mx.mean(mx.abs((w_orig - w_quant).astype(mx.float32))).item()


def max_abs_error(w_orig: mx.array, w_quant: mx.array) -> float:
    """Maximum absolute error."""
    return mx.max(mx.abs((w_orig - w_quant).astype(mx.float32))).item()


def group_dynamic_range(w: mx.array, group_size: int) -> mx.array:
    """Per-group dynamic range (max/min of absolute values).

    Returns (n_rows, n_groups) array of range ratios.
    """
    w_f = w.astype(mx.float32)
    rows, cols = w_f.shape
    n_groups = cols // group_size
    w_grouped = mx.abs(w_f.reshape(rows, n_groups, group_size))
    g_max = mx.max(w_grouped, axis=2)
    g_min = mx.min(w_grouped, axis=2) + 1e-12
    return g_max / g_min


def mean_group_dynamic_range(w: mx.array, group_size: int) -> float:
    """Mean of per-group dynamic range across all groups."""
    return mx.mean(group_dynamic_range(w, group_size)).item()


def weight_kurtosis(w: mx.array) -> float:
    """Excess kurtosis of flattened weight distribution."""
    w_f = w.astype(mx.float32).reshape(-1)
    mu = mx.mean(w_f)
    diff = w_f - mu
    var = mx.mean(diff ** 2)
    kurt = mx.mean(diff ** 4) / (var ** 2 + 1e-12)
    return kurt.item() - 3.0  # excess kurtosis


# ── Layer-domain metrics ─────────────────────────────────────────────────────


def layer_output_mse(states_orig: mx.array, states_quant: mx.array) -> float:
    """MSE between original and quantized hidden states."""
    diff = (states_orig - states_quant).astype(mx.float32)
    return mx.mean(diff * diff).item()


def layer_output_relative_mse(states_orig: mx.array, states_quant: mx.array) -> float:
    """Relative MSE of hidden states."""
    diff = (states_orig - states_quant).astype(mx.float32)
    mse = mx.mean(diff * diff).item()
    ref = mx.mean(states_orig.astype(mx.float32) ** 2).item()
    return mse / max(ref, 1e-12)


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Mean cosine similarity across last dimension."""
    a_f = a.astype(mx.float32)
    b_f = b.astype(mx.float32)
    dot = mx.sum(a_f * b_f, axis=-1)
    norm_a = mx.sqrt(mx.sum(a_f * a_f, axis=-1) + 1e-12)
    norm_b = mx.sqrt(mx.sum(b_f * b_f, axis=-1) + 1e-12)
    return mx.mean(dot / (norm_a * norm_b)).item()


# ── Logit-domain metrics ────────────────────────────────────────────────────


def logit_kl_divergence(logits_orig: mx.array, logits_quant: mx.array) -> float:
    """KL(p_orig || p_quant) averaged over positions.

    p_orig = softmax(logits_orig), q = softmax(logits_quant).
    KL = sum(p * log(p/q)).
    """
    # Shift for numerical stability
    lo = logits_orig.astype(mx.float32)
    lq = logits_quant.astype(mx.float32)
    lo = lo - mx.max(lo, axis=-1, keepdims=True)
    lq = lq - mx.max(lq, axis=-1, keepdims=True)

    log_p = lo - mx.logsumexp(lo, axis=-1, keepdims=True)
    log_q = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
    p = mx.exp(log_p)

    kl = mx.sum(p * (log_p - log_q), axis=-1)
    return mx.mean(kl).item()


def topk_overlap(logits_orig: mx.array, logits_quant: mx.array, k: int = 10) -> float:
    """Mean overlap of top-k predictions across positions.

    Returns fraction of top-k tokens that appear in both sets.
    """
    # Get top-k indices for each position
    topk_orig = mx.argpartition(logits_orig, kth=-k, axis=-1)[..., -k:]
    topk_quant = mx.argpartition(logits_quant, kth=-k, axis=-1)[..., -k:]

    # Sort for consistent comparison
    topk_orig = mx.sort(topk_orig, axis=-1)
    topk_quant = mx.sort(topk_quant, axis=-1)

    # Count overlaps position by position
    batch_dims = logits_orig.shape[:-1]
    n_positions = 1
    for d in batch_dims:
        n_positions *= d

    topk_orig_flat = topk_orig.reshape(n_positions, k)
    topk_quant_flat = topk_quant.reshape(n_positions, k)

    overlaps = 0.0
    for i in range(n_positions):
        set_orig = set(topk_orig_flat[i].tolist())
        set_quant = set(topk_quant_flat[i].tolist())
        overlaps += len(set_orig & set_quant) / k

    return overlaps / max(n_positions, 1)


def ce_increase(
    logits_orig: mx.array,
    logits_quant: mx.array,
    targets: mx.array,
) -> float:
    """Increase in cross-entropy loss from quantization.

    Returns CE(quant) - CE(orig) using targets as ground truth.
    """

    def _ce(logits, targets):
        log_probs = logits.astype(mx.float32) - mx.logsumexp(
            logits.astype(mx.float32), axis=-1, keepdims=True
        )
        # Gather log-probs for target tokens
        n_pos = targets.shape[-1]
        # Flatten for indexing
        lp_flat = log_probs.reshape(-1, logits.shape[-1])
        t_flat = targets.reshape(-1)
        # Index: lp_flat[i, t_flat[i]]
        nll = -mx.mean(
            mx.take_along_axis(lp_flat, t_flat[:, None], axis=1).squeeze(1)
        )
        return nll.item()

    return _ce(logits_quant, targets) - _ce(logits_orig, targets)


def top1_match(logits_orig: mx.array, logits_quant: mx.array) -> float:
    """Fraction of positions where argmax agrees."""
    pred_orig = mx.argmax(logits_orig, axis=-1)
    pred_quant = mx.argmax(logits_quant, axis=-1)
    return mx.mean((pred_orig == pred_quant).astype(mx.float32)).item()


# ── Summary helpers ──────────────────────────────────────────────────────────


def weight_metrics_summary(
    w_orig: mx.array,
    w_quant: mx.array,
    group_size: int = 128,
) -> dict:
    """Compute all weight-domain metrics in one call."""
    return {
        "relative_mse": relative_mse(w_orig, w_quant),
        "sign_agreement": sign_agreement(w_orig, w_quant),
        "mean_abs_error": mean_abs_error(w_orig, w_quant),
        "max_abs_error": max_abs_error(w_orig, w_quant),
        "kurtosis_orig": weight_kurtosis(w_orig),
        "mean_group_dynamic_range": mean_group_dynamic_range(w_orig, group_size),
    }


def logit_metrics_summary(
    logits_orig: mx.array,
    logits_quant: mx.array,
    targets: mx.array | None = None,
) -> dict:
    """Compute all logit-domain metrics in one call."""
    result = {
        "kl_divergence": logit_kl_divergence(logits_orig, logits_quant),
        "top1_match": top1_match(logits_orig, logits_quant),
        "top10_overlap": topk_overlap(logits_orig, logits_quant, k=10),
    }
    if targets is not None:
        result["ce_increase"] = ce_increase(logits_orig, logits_quant, targets)
    return result
