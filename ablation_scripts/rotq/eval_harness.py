"""Evaluation harness for ROTQ — orchestrates metrics collection.

Compares quantized models against a BF16 reference at weight, layer,
logit, and generation levels.  Produces JSON reports suitable for
ablation studies.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import calibration, metrics


def evaluate_weight_quality(
    weights_orig: dict[str, mx.array],
    weights_quant: dict[str, mx.array],
    *,
    group_size: int = 128,
    keys: list[str] | None = None,
) -> dict:
    """Compare original and quantized weight dictionaries.

    Args:
        weights_orig: Name→tensor map of original FP weights.
        weights_quant: Name→tensor map of dequantized quantized weights.
        group_size: Group size used for quantization.
        keys: Subset of keys to evaluate. If None, evaluate all shared 2D keys.

    Returns:
        Dict with per-layer and aggregate weight metrics.
    """
    if keys is None:
        keys = [
            k for k in weights_orig
            if k in weights_quant
            and weights_orig[k].ndim == 2
            and weights_quant[k].ndim == 2
        ]

    per_layer = {}
    agg_mse = []
    agg_sign = []

    for k in sorted(keys):
        w_o = weights_orig[k]
        w_q = weights_quant[k]
        if w_o.shape != w_q.shape:
            continue

        m = metrics.weight_metrics_summary(w_o, w_q, group_size)
        per_layer[k] = m
        agg_mse.append(m["relative_mse"])
        agg_sign.append(m["sign_agreement"])

    agg = {}
    if agg_mse:
        agg["mean_relative_mse"] = sum(agg_mse) / len(agg_mse)
        agg["max_relative_mse"] = max(agg_mse)
        agg["mean_sign_agreement"] = sum(agg_sign) / len(agg_sign)
        agg["n_layers"] = len(agg_mse)

    return {"aggregate": agg, "per_layer": per_layer}


def evaluate_logit_quality(
    model_orig,
    model_quant,
    tokens: mx.array,
) -> dict:
    """Compare logit outputs of two models on the same tokens.

    Args:
        model_orig: Reference (FP) model.
        model_quant: Quantized model.
        tokens: (batch, seq_len) calibration tokens.

    Returns:
        Dict with logit-domain metrics.
    """
    logits_orig = calibration.capture_logits(model_orig, tokens)
    logits_quant = calibration.capture_logits(model_quant, tokens)

    # Use tokens shifted by 1 as targets for CE measurement
    targets = tokens[:, 1:]
    lo_trim = logits_orig[:, :-1, :]
    lq_trim = logits_quant[:, :-1, :]

    result = metrics.logit_metrics_summary(lo_trim, lq_trim, targets)
    return result


def evaluate_layer_quality(
    model_orig,
    model_quant,
    tokens: mx.array,
    layer_indices: list[int] | None = None,
) -> dict:
    """Compare hidden-state outputs of two models at specified layers.

    Args:
        model_orig: Reference (FP) model.
        model_quant: Quantized model.
        tokens: (batch, seq_len) calibration tokens.
        layer_indices: Which layers to compare. If None, compares all.

    Returns:
        Dict with per-layer and aggregate hidden-state metrics.
    """
    outputs_orig = calibration.capture_layer_outputs(model_orig, tokens, layer_indices)
    outputs_quant = calibration.capture_layer_outputs(model_quant, tokens, layer_indices)

    per_layer = {}
    agg_cos = []

    for idx in sorted(outputs_orig.keys()):
        if idx not in outputs_quant:
            continue
        s_o = outputs_orig[idx]
        s_q = outputs_quant[idx]
        cos = metrics.cosine_similarity(s_o, s_q)
        rmse = metrics.layer_output_relative_mse(s_o, s_q)
        per_layer[idx] = {"cosine_similarity": cos, "relative_mse": rmse}
        agg_cos.append(cos)

    agg = {}
    if agg_cos:
        agg["mean_cosine_similarity"] = sum(agg_cos) / len(agg_cos)
        agg["min_cosine_similarity"] = min(agg_cos)
        agg["n_layers"] = len(agg_cos)

    return {"aggregate": agg, "per_layer": per_layer}


def full_evaluation(
    model_orig,
    model_quant,
    tokenizer,
    *,
    weights_orig: dict[str, mx.array] | None = None,
    weights_quant: dict[str, mx.array] | None = None,
    group_size: int = 128,
    calibration_text_path: str | None = None,
    n_samples: int = 4,
    seq_len: int = 256,
    layer_indices: list[int] | None = None,
) -> dict:
    """Run the full evaluation suite.

    Args:
        model_orig: Reference (FP) model.
        model_quant: Quantized model.
        tokenizer: Tokenizer for calibration data.
        weights_orig: Original weight dict (for weight-domain metrics).
        weights_quant: Dequantized weight dict (for weight-domain metrics).
        group_size: Quantization group size.
        calibration_text_path: Path to calibration text file.
        n_samples: Number of calibration samples.
        seq_len: Calibration sequence length.
        layer_indices: Layers to compare for hidden-state metrics.

    Returns:
        Comprehensive evaluation dict.
    """
    tokens = calibration.load_calibration_tokens(
        tokenizer,
        text_path=calibration_text_path,
        n_samples=n_samples,
        seq_len=seq_len,
    )

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Weight-domain
    if weights_orig is not None and weights_quant is not None:
        report["weight_quality"] = evaluate_weight_quality(
            weights_orig, weights_quant, group_size=group_size
        )

    # Logit-domain
    print("  Evaluating logit quality...")
    report["logit_quality"] = evaluate_logit_quality(model_orig, model_quant, tokens)

    # Layer-domain
    if layer_indices is not None:
        print("  Evaluating layer quality...")
        report["layer_quality"] = evaluate_layer_quality(
            model_orig, model_quant, tokens, layer_indices
        )

    return report


def print_report(report: dict, file=None):
    """Pretty-print an evaluation report."""
    import sys
    out = file or sys.stdout

    print("=" * 72, file=out)
    print("ROTQ Evaluation Report", file=out)
    print("=" * 72, file=out)

    if "weight_quality" in report:
        wq = report["weight_quality"]["aggregate"]
        print(f"\nWeight Quality ({wq.get('n_layers', '?')} layers):", file=out)
        print(f"  Mean relative MSE:    {wq.get('mean_relative_mse', 'N/A'):.4f}", file=out)
        print(f"  Max relative MSE:     {wq.get('max_relative_mse', 'N/A'):.4f}", file=out)
        print(f"  Mean sign agreement:  {wq.get('mean_sign_agreement', 'N/A'):.4f}", file=out)

    if "logit_quality" in report:
        lq = report["logit_quality"]
        print(f"\nLogit Quality:", file=out)
        print(f"  KL divergence:   {lq.get('kl_divergence', 'N/A'):.6f}", file=out)
        print(f"  Top-1 match:     {lq.get('top1_match', 'N/A'):.4f}", file=out)
        print(f"  Top-10 overlap:  {lq.get('top10_overlap', 'N/A'):.4f}", file=out)
        if "ce_increase" in lq:
            print(f"  CE increase:     {lq['ce_increase']:.6f}", file=out)

    if "layer_quality" in report:
        ly = report["layer_quality"]["aggregate"]
        print(f"\nLayer Quality ({ly.get('n_layers', '?')} layers):", file=out)
        print(f"  Mean cosine sim: {ly.get('mean_cosine_similarity', 'N/A'):.6f}", file=out)
        print(f"  Min cosine sim:  {ly.get('min_cosine_similarity', 'N/A'):.6f}", file=out)

    print("=" * 72, file=out)
