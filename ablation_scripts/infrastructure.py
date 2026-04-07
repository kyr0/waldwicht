#!/usr/bin/env python3
"""Shared infrastructure for calibration-aware PLE quantization studies.

Provides model loading, quantization, evaluation, answer quality generation,
and activation capture utilities used by all study scripts.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ── Paths & constants ────────────────────────────────────────────────────────

BF16_PATH = "hf/gemma-4-E2B-it"
OPTIMAL_PATH = "mlx/gemma-4-E2B-q3-g64-optimal"
CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "optimal", "calibration", "calibration_e2b.json"
)
AQ_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "answer_quality_calibration.json"
)
TMP_DIR = "tmp-cap-e2b"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_LAYERS = 35

PPL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning, there was nothing but darkness and silence. "
    "Then the universe expanded rapidly, creating matter and energy. "
    "Stars formed, planets coalesced, and life emerged on at least one world. "
    "Humans developed language, tools, agriculture, and eventually technology. "
    "Today we stand at the threshold of artificial general intelligence."
)

# Supported bit-widths in MLX Metal kernels
VALID_BITS = (1, 2, 3, 4, 5, 6, 8)

# The top-down ladder (conservative → aggressive)
BIT_LADDER = (6, 5, 4, 3, 2)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(path: str):
    """Load model and tokenizer via mlx_lm."""
    return load(path)


def load_bf16_weights(path: str = BF16_PATH) -> dict[str, mx.array]:
    """Load BF16 weight dict from safetensors shards."""
    shards = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    weights = {}
    for s in shards:
        weights.update(mx.load(s))
    return weights


# ── Model size ───────────────────────────────────────────────────────────────

def model_size_gb(path: str) -> float:
    """Sum safetensors file sizes in GB."""
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path) if f.endswith(".safetensors")
    ) / 1e9


# ── PPL computation ──────────────────────────────────────────────────────────

def compute_ppl(model, tokenizer, text: str = PPL_TEXT) -> float:
    tokens = tokenizer.encode(text, return_tensors=None)
    token_ids = tokens if isinstance(tokens, list) else list(tokens)
    if len(token_ids) < 2:
        return float("inf")
    x = mx.array(token_ids)[None, :]
    logits = model(x)
    mx.eval(logits)
    logits = logits[:, :-1, :]
    targets = mx.array(token_ids[1:])[None, :]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = 0.0
    for i in range(targets.shape[1]):
        nll -= log_probs[0, i, targets[0, i].item()].item()
    return float(mx.exp(mx.array(nll / targets.shape[1])).item())


def token_overlap(ref_tokens: list, hyp_tokens: list) -> float:
    """Token-level F1."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    common = ref_set & hyp_set
    if not common:
        return 0.0
    precision = len(common) / len(hyp_set)
    recall = len(common) / len(ref_set)
    return 2 * precision * recall / (precision + recall)


# ── Quantization utilities ───────────────────────────────────────────────────

def requantize_tensors(
    bf16_weights: dict,
    target_keys_prefix: str,
    bits: int,
    group_size: int,
) -> tuple[dict, int]:
    """Re-quantize specific tensors from BF16 at given bits/group_size.

    Returns (dict with .weight/.scales/.biases, count of matched weights).
    """
    out = {}
    matched = [k for k in bf16_weights
               if target_keys_prefix in k and k.endswith(".weight")]
    for k in matched:
        w = bf16_weights[k].astype(mx.float32)
        if w.ndim != 2 or w.shape[-1] < group_size or w.shape[-1] % group_size != 0:
            out[k] = bf16_weights[k]
            continue
        weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(weight, scales, biases)
        base = k[: -len(".weight")]
        out[f"{base}.weight"] = weight
        out[f"{base}.scales"] = scales
        out[f"{base}.biases"] = biases
    return out, len(matched)


def requantize_all_linear(
    bf16_weights: dict,
    bits: int,
    group_size: int,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[dict, int]:
    """Re-quantize ALL 2-D weight tensors from BF16.

    Skips tensors matching any exclude_prefix. Returns quantized dict +
    count of quantized weights.
    """
    out = {}
    count = 0
    for k in bf16_weights:
        if not k.endswith(".weight"):
            continue
        if any(ep in k for ep in exclude_prefixes):
            continue
        w = bf16_weights[k].astype(mx.float32)
        if w.ndim != 2 or w.shape[-1] < group_size or w.shape[-1] % group_size != 0:
            out[k] = bf16_weights[k]
            continue
        weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(weight, scales, biases)
        base = k[: -len(".weight")]
        out[f"{base}.weight"] = weight
        out[f"{base}.scales"] = scales
        out[f"{base}.biases"] = biases
        count += 1
    return out, count


# ── Sharded save ─────────────────────────────────────────────────────────────

def save_sharded(weights: dict, out_dir: str, max_shard_bytes: int = 2 * 1024**3):
    """Save weights as sharded safetensors."""
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

    index_map = {}
    if len(shards) == 1:
        fname = "model.safetensors"
        mx.save_safetensors(os.path.join(out_dir, fname), shards[0])
        for k in shards[0]:
            index_map[k] = fname
    else:
        for i, shard in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            mx.save_safetensors(os.path.join(out_dir, fname), shard)
            for k in shard:
                index_map[k] = fname
        index = {
            "metadata": {"total_size": sum(w.size * w.dtype.size for w in weights.values())},
            "weight_map": index_map,
        }
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


# ── Config patching ──────────────────────────────────────────────────────────

def patch_config_quantization(config_path: str, global_bits: int, global_gs: int,
                              overrides: dict | None = None):
    """Rewrite the quantization section in config.json.

    Args:
        config_path: Path to config.json.
        global_bits: Default bits for all weights.
        global_gs: Default group_size for all weights.
        overrides: Optional dict of {key_substring: {"bits": N, "group_size": G}}
                   that override specific weight groups.
    """
    with open(config_path) as f:
        config = json.load(f)

    # Start fresh — remove all per-layer overrides from old config
    quant = {"bits": global_bits, "group_size": global_gs}

    if overrides:
        for key_pattern, qcfg in overrides.items():
            quant[key_pattern] = qcfg

    config["quantization"] = quant
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_variant(
    variant_path: str,
    calibration_path: str = CALIBRATION_PATH,
    max_eval_samples: int = 64,
) -> dict:
    """Load variant, compute PPL and compare against calibration baseline.

    Returns dict with ppl, exact_match_pct, avg_token_overlap, n_samples.
    """
    model, tok = load(variant_path)
    ppl = compute_ppl(model, tok)

    with open(calibration_path) as f:
        cal = json.load(f)

    sampler = make_sampler(temp=0.0)
    samples = cal["samples"][:max_eval_samples]
    exact_matches = 0
    overlap_scores = []

    for s in samples:
        msgs = [{"role": "user", "content": s["prompt"]}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tok, prompt=chat, max_tokens=200,
                            sampler=sampler, verbose=False)

        ref_first = s["response"].strip().split("\n")[0].strip()
        hyp_first = response.strip().split("\n")[0].strip()
        if ref_first == hyp_first:
            exact_matches += 1

        ref_toks = tok.encode(s["response"])
        hyp_toks = tok.encode(response)
        if isinstance(ref_toks, list) and isinstance(hyp_toks, list):
            overlap_scores.append(token_overlap(ref_toks, hyp_toks))

    del model, tok
    gc.collect()
    mx.clear_cache()

    return {
        "ppl": round(ppl, 1),
        "exact_match": exact_matches,
        "exact_match_pct": round(100 * exact_matches / len(samples), 1),
        "avg_token_overlap": (
            round(sum(overlap_scores) / len(overlap_scores), 3)
            if overlap_scores else 0
        ),
        "n_samples": len(samples),
    }


# ── Answer quality generation ────────────────────────────────────────────────

def run_answer_quality(
    variant_path: str,
    out_path: str,
    aq_calibration_path: str = AQ_CALIBRATION_PATH,
) -> list[dict]:
    """Generate responses to the 8 answer-quality prompts.

    Saves JSON with the same schema as answer_quality_calibration.json but
    with scores set to None (to be filled in by manual LLM-judge scoring).

    Returns the list of result dicts.
    """
    with open(aq_calibration_path) as f:
        aq_prompts = json.load(f)

    model, tok = load(variant_path)
    sampler = make_sampler(temp=0.0)
    results = []

    for item in aq_prompts:
        prompt = item["prompt"]
        msgs = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        response = generate(model, tok, prompt=chat, max_tokens=512,
                            sampler=sampler, verbose=False)
        elapsed = time.time() - t0

        out_toks = tok.encode(response)
        n_tokens = len(out_toks) if isinstance(out_toks, list) else out_toks.shape[0]

        results.append({
            "id": item["id"],
            "prompt": prompt,
            "response": response,
            "tokens": n_tokens,
            "time_s": round(elapsed, 2),
            "tok_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else 0,
            "correctness": None,
            "completion": None,
            "reasoning_hygiene": None,
        })

    del model, tok
    gc.collect()
    mx.clear_cache()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Answer quality responses saved to {out_path}")
    return results


def check_quality_gate(scored_path: str, threshold: float = 9.0) -> dict:
    """Validate that a scored answer-quality JSON meets the quality gate.

    Returns dict with per-dimension averages, pass/fail status, and
    whether all items have been scored.
    """
    with open(scored_path) as f:
        items = json.load(f)

    dims = ("correctness", "completion", "reasoning_hygiene")
    total = len(items)
    avgs = {}
    scored_counts = {}
    for d in dims:
        scores = [it[d] for it in items if it[d] is not None]
        scored_counts[d] = len(scores)
        avgs[d] = sum(scores) / len(scores) if scores else None

    all_scored = all(scored_counts[d] == total for d in dims)
    passed = all_scored and all(avgs[d] is not None and avgs[d] >= threshold for d in dims)
    return {"averages": avgs, "threshold": threshold, "passed": passed, "all_scored": all_scored}


# ── Uniform model export ─────────────────────────────────────────────────────

def export_uniform_quantized(
    bf16_path: str,
    out_path: str,
    bits: int,
    group_size: int,
) -> float:
    """Quantize all linear layers uniformly from BF16 and export.

    Copies non-weight files (config.json, tokenizer, etc.) from the
    existing optimal model, replaces all weights, patches config.

    Returns model size in GB.
    """
    # Copy optimal model structure (configs, tokenizer files)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(OPTIMAL_PATH, out_path)

    # Remove old weight files
    for f in os.listdir(out_path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_path, f))

    # Load BF16 and quantize everything
    bf16_weights = load_bf16_weights(bf16_path)
    quantized, n = requantize_all_linear(bf16_weights, bits, group_size)

    # Copy non-weight tensors (1-D biases, norms, etc.)
    for k, v in bf16_weights.items():
        if k not in quantized and not any(
            k.endswith(s) for s in (".scales", ".biases")
        ):
            # Check it's not a derivative of something we already quantized
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

    # Patch config
    patch_config_quantization(
        os.path.join(out_path, "config.json"),
        global_bits=bits,
        global_gs=group_size,
    )

    return model_size_gb(out_path)


# ── Component-specific re-quantization ───────────────────────────────────────

# Component groups: each maps a human-readable name to a list of weight key
# substrings that identify that group.
COMPONENT_GROUPS = {
    "ple_embeddings": ["embed_tokens_per_layer"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "ple_gate_proj": ["per_layer_input_gate", "per_layer_projection"],
    "main_embed_lmhead": ["embed_tokens", "lm_head"],
}


def export_mixed_quantized(
    bf16_path: str,
    out_path: str,
    default_bits: int,
    default_gs: int,
    component_overrides: dict[str, tuple[int, int]],
) -> float:
    """Quantize with per-component bit/group_size overrides.

    Args:
        bf16_path: Path to BF16 source model.
        out_path: Output directory.
        default_bits: Default bits for all weights.
        default_gs: Default group_size for all weights.
        component_overrides: Dict of {component_name: (bits, group_size)}.
            Component names must be keys from COMPONENT_GROUPS.

    Returns model size in GB.
    """
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(OPTIMAL_PATH, out_path)

    for f in os.listdir(out_path):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_path, f))

    bf16_weights = load_bf16_weights(bf16_path)

    # Build a map: weight_key → (bits, group_size)
    key_to_quant = {}
    for k in bf16_weights:
        if not k.endswith(".weight"):
            continue
        assigned = False
        for comp_name, (comp_bits, comp_gs) in component_overrides.items():
            prefixes = COMPONENT_GROUPS[comp_name]
            # Check if this weight belongs to this component
            for pfx in prefixes:
                # For main_embed_lmhead, exclude per_layer variants
                if comp_name == "main_embed_lmhead":
                    if pfx in k and "per_layer" not in k:
                        key_to_quant[k] = (comp_bits, comp_gs)
                        assigned = True
                        break
                else:
                    if pfx in k:
                        key_to_quant[k] = (comp_bits, comp_gs)
                        assigned = True
                        break
            if assigned:
                break
        if not assigned:
            key_to_quant[k] = (default_bits, default_gs)

    # Quantize each weight at its assigned bits/group_size
    quantized = {}
    for k, (bits, gs) in key_to_quant.items():
        w = bf16_weights[k].astype(mx.float32)
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
        if k not in quantized and not any(
            k.endswith(s) for s in (".scales", ".biases")
        ):
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

    # Build config overrides
    config_overrides = {}
    for comp_name, (comp_bits, comp_gs) in component_overrides.items():
        for pfx in COMPONENT_GROUPS[comp_name]:
            config_overrides[pfx] = {"bits": comp_bits, "group_size": comp_gs}
        # Disambiguation: "embed_tokens" substring also matches "embed_tokens_per_layer"
        # Add explicit PLE entry at default bits to ensure longest-match wins
        if comp_name == "main_embed_lmhead" and "ple_embeddings" not in component_overrides:
            config_overrides["embed_tokens_per_layer"] = {
                "bits": default_bits, "group_size": default_gs
            }

    patch_config_quantization(
        os.path.join(out_path, "config.json"),
        global_bits=default_bits,
        global_gs=default_gs,
        overrides=config_overrides,
    )

    return model_size_gb(out_path)


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup_variant(path: str):
    """Remove a temporary variant directory."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    gc.collect()
    mx.clear_cache()


def ensure_dirs():
    """Create results and tmp directories."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)


# ── Results logging ──────────────────────────────────────────────────────────

def log_result(results_path: str, entry: dict):
    """Append a result entry to a JSON list file."""
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)


def print_result(name: str, size_gb: float, metrics: dict, baseline: dict | None = None):
    """Pretty-print a single experiment result."""
    ppl_str = f"PPL: {metrics['ppl']}"
    if baseline:
        delta = metrics["ppl"] - baseline["ppl"]
        ppl_str += f" (Δ {delta:+.0f})"
    print(f"  {name}: {size_gb:.2f} GB | {ppl_str} | "
          f"Exact: {metrics['exact_match']}/{metrics['n_samples']} "
          f"({metrics['exact_match_pct']}%) | "
          f"Overlap: {metrics['avg_token_overlap']:.3f}")
