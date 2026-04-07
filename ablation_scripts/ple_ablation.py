#!/usr/bin/env python3
"""PLE/Embedding shrinkage ablation: test aggressive quantization on embeddings.

Takes the existing optimal models, re-quantizes ONLY the PLE and embed_tokens
tensors at lower bits / larger group sizes, and compares inference against
the 64-sample calibration baseline.

Experiments (per model):
  A. PLE 2-bit g64   (from 3-bit g64 for E2B / 4-bit g64 for E4B)
  B. PLE 2-bit g128  (also increase group size)
  C. embed_tokens 2-bit g64
  D. embed_tokens 2-bit g128
  E. Both PLE + embed at 2-bit g64
  F. Both PLE + embed at 2-bit g128
  G. PLE 3-bit g128 (just increase group size, E4B only since E2B is already 3-bit g64)

Metrics:
  - Model size (GB)
  - Perplexity
  - Exact-match rate vs calibration (greedy responses)
  - ROUGE-L / token overlap vs calibration

Usage:
    .venv/bin/python optimal/ple_ablation.py [--model e2b|e4b|both]
"""
import argparse
import copy
import gc
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS = {
    "e2b": {
        "optimal": "mlx/gemma-4-E2B-q3-g64-optimal",
        "bf16": "hf/gemma-4-E2B-it",
        "calibration": os.path.join(os.path.dirname(__file__), "calibration", "calibration_e2b.json"),
        "tmp_dir": "tmp-ple-e2b",
    },
    "e4b": {
        "optimal": "mlx/gemma-4-E4B-e4ne4g64-optimal",
        "bf16": "mlx/gemma-4-E4B-it-bf16",
        "calibration": os.path.join(os.path.dirname(__file__), "calibration", "calibration_e4b.json"),
        "tmp_dir": "tmp-ple-e4b",
    },
}

# ── PPL text (same as all ablations) ─────────────────────────────────────────

PPL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning, there was nothing but darkness and silence. "
    "Then the universe expanded rapidly, creating matter and energy. "
    "Stars formed, planets coalesced, and life emerged on at least one world. "
    "Humans developed language, tools, agriculture, and eventually technology. "
    "Today we stand at the threshold of artificial general intelligence."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_size_gb(path: str) -> float:
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path) if f.endswith(".safetensors")
    ) / 1e9


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
    """Simple token-level F1 (proxy for ROUGE-L)."""
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


def requantize_tensors(
    bf16_weights: dict,
    target_keys_prefix: str,
    bits: int,
    group_size: int,
) -> dict:
    """Re-quantize specific tensors from BF16 at given bits/group_size.

    Returns dict with .weight, .scales, .biases for each matched base key.
    """
    out = {}
    # Find all raw weight keys matching the prefix (BF16 has just .weight)
    # The BF16 model may have keys like "model.language_model.embed_tokens_per_layer.weight"
    # or "language_model.model.embed_tokens_per_layer.0.weight" etc.
    matched = [k for k in bf16_weights if target_keys_prefix in k and k.endswith(".weight")]

    for k in matched:
        w = bf16_weights[k].astype(mx.float32)
        if w.ndim < 2:
            out[k] = bf16_weights[k]
            continue
        weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(weight, scales, biases)
        base = k[:-len(".weight")]
        out[f"{base}.weight"] = weight
        out[f"{base}.scales"] = scales
        out[f"{base}.biases"] = biases

    return out, len(matched)


def create_variant(
    optimal_path: str,
    bf16_path: str,
    out_path: str,
    ple_bits: int | None,
    ple_gs: int | None,
    embed_bits: int | None,
    embed_gs: int | None,
    model_name: str,
):
    """Create a model variant by patching PLE/embed quantization."""
    # Copy the optimal model
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    shutil.copytree(optimal_path, out_path)

    # Load BF16 weights for re-quantization
    bf16_shards = sorted(glob.glob(os.path.join(bf16_path, "model*.safetensors")))
    if not bf16_shards:
        bf16_shards = sorted(glob.glob(os.path.join(bf16_path, "*.safetensors")))
    bf16_weights = {}
    for s in bf16_shards:
        bf16_weights.update(mx.load(s))

    # Load current optimal weights
    opt_shards = sorted(glob.glob(os.path.join(out_path, "model*.safetensors")))
    opt_weights = {}
    for s in opt_shards:
        opt_weights.update(mx.load(s))

    # Track changes
    patches = {}
    config_overrides = {}

    if ple_bits is not None and ple_gs is not None:
        new_ple, n = requantize_tensors(bf16_weights, "embed_tokens_per_layer", ple_bits, ple_gs)
        if n == 0:
            print(f"  WARNING: No PLE weights found in BF16 for re-quantization")
        else:
            patches.update(new_ple)
            print(f"  PLE: re-quantized {n} weight tensors → {ple_bits}-bit g{ple_gs}")

    if embed_bits is not None and embed_gs is not None:
        # Be careful not to match embed_tokens_per_layer
        new_emb = {}
        for k in bf16_weights:
            if "embed_tokens" in k and "per_layer" not in k and k.endswith(".weight"):
                w = bf16_weights[k].astype(mx.float32)
                if w.ndim < 2:
                    new_emb[k] = bf16_weights[k]
                    continue
                weight, scales, biases = mx.quantize(w, group_size=embed_gs, bits=embed_bits)
                mx.eval(weight, scales, biases)
                base = k[:-len(".weight")]
                new_emb[f"{base}.weight"] = weight
                new_emb[f"{base}.scales"] = scales
                new_emb[f"{base}.biases"] = biases
        if new_emb:
            patches.update(new_emb)
            print(f"  embed_tokens: re-quantized → {embed_bits}-bit g{embed_gs}")
        else:
            print(f"  WARNING: No embed_tokens weights found in BF16")

    # Apply patches
    for k, v in patches.items():
        # Map BF16 key to optimal model key space
        # BF16 may use "model.language_model.X" while optimal uses that or
        # "language_model.model.X" — try both
        if k in opt_weights:
            opt_weights[k] = v
        else:
            # Try key transformations
            alt_k = k
            if alt_k.startswith("model."):
                alt_k = alt_k[len("model."):]
            if alt_k.startswith("language_model.") and not alt_k.startswith("language_model.model."):
                alt_k = "language_model.model." + alt_k[len("language_model."):]

            if alt_k in opt_weights:
                opt_weights[alt_k] = v
            else:
                # Try removing model. prefix entirely
                for opt_k in list(opt_weights.keys()):
                    # Match by the embedding-specific suffix
                    if k.split("embed_tokens")[-1] == opt_k.split("embed_tokens")[-1] and "embed_tokens" in opt_k:
                        opt_weights[opt_k] = v
                        break

    # Clean up old shard files
    for s in opt_shards:
        os.remove(s)

    # Save
    _save_sharded(opt_weights, out_path)

    # Update config.json quantization overrides
    config_path = os.path.join(out_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    quant = config.get("quantization", {})
    # Update PLE overrides
    if ple_bits is not None:
        for qk in list(quant.keys()):
            if "embed_tokens_per_layer" in qk:
                quant[qk] = {"group_size": ple_gs, "bits": ple_bits}
    if embed_bits is not None:
        for qk in list(quant.keys()):
            if "embed_tokens" in qk and "per_layer" not in qk:
                quant[qk] = {"group_size": embed_gs, "bits": embed_bits}
    config["quantization"] = quant

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Cleanup BF16
    del bf16_weights, opt_weights, patches
    gc.collect()
    mx.clear_cache()

    return model_size_gb(out_path)


def _save_sharded(weights: dict, out_dir: str, max_shard_bytes: int = 2 * 1024**3):
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


def evaluate_variant(
    variant_path: str,
    calibration_path: str,
    max_eval_samples: int = 64,
) -> dict:
    """Load variant, compute PPL and compare against calibration."""
    model, tok = load(variant_path)

    # PPL
    ppl = compute_ppl(model, tok)

    # Load calibration
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

        # Exact match (first line)
        ref_first = s["response"].strip().split("\n")[0].strip()
        hyp_first = response.strip().split("\n")[0].strip()
        if ref_first == hyp_first:
            exact_matches += 1

        # Token overlap
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
        "avg_token_overlap": round(sum(overlap_scores) / len(overlap_scores), 3) if overlap_scores else 0,
        "n_samples": len(samples),
    }


# ── Experiment configurations ─────────────────────────────────────────────────

def get_experiments(model_name: str):
    """Return list of (name, ple_bits, ple_gs, embed_bits, embed_gs) tuples."""
    configs = [
        # (name, ple_bits, ple_gs, embed_bits, embed_gs)
        ("ple-2bit-g64",    2, 64,   None, None),
        ("ple-2bit-g128",   2, 128,  None, None),
        ("emb-2bit-g64",    None, None, 2, 64),
        ("emb-2bit-g128",   None, None, 2, 128),
        ("both-2bit-g64",   2, 64,   2, 64),
        ("both-2bit-g128",  2, 128,  2, 128),
    ]
    if model_name == "e4b":
        # E4B PLE is 4-bit g64 — also try 3-bit (E2B's current setting)
        configs.insert(0, ("ple-3bit-g64", 3, 64, None, None))
        configs.insert(1, ("ple-3bit-g128", 3, 128, None, None))
    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model_ablation(model_name: str):
    paths = MODELS[model_name]
    experiments = get_experiments(model_name)

    print(f"\n{'#'*80}")
    print(f"  PLE Ablation: {model_name.upper()}")
    print(f"  Optimal: {paths['optimal']} ({model_size_gb(paths['optimal']):.2f} GB)")
    print(f"  BF16:    {paths['bf16']}")
    print(f"  Configs: {len(experiments)}")
    print(f"{'#'*80}")

    # Baseline evaluation
    print(f"\n── Baseline (current optimal) ──")
    baseline_size = model_size_gb(paths["optimal"])
    baseline = evaluate_variant(paths["optimal"], paths["calibration"])
    baseline["size_gb"] = baseline_size
    print(f"  Size: {baseline_size:.2f} GB | PPL: {baseline['ppl']} | "
          f"Exact: {baseline['exact_match']}/{baseline['n_samples']} ({baseline['exact_match_pct']}%) | "
          f"Overlap: {baseline['avg_token_overlap']:.3f}")

    results = [{"name": "baseline", "config": "current optimal", **baseline}]

    # Run experiments
    for exp_name, ple_bits, ple_gs, emb_bits, emb_gs in experiments:
        print(f"\n── {exp_name} ──")
        out_path = os.path.join(paths["tmp_dir"], f"{model_name}-{exp_name}")
        os.makedirs(paths["tmp_dir"], exist_ok=True)

        size = create_variant(
            paths["optimal"], paths["bf16"], out_path,
            ple_bits, ple_gs, emb_bits, emb_gs, model_name,
        )
        print(f"  Size: {size:.2f} GB (Δ {size - baseline_size:+.2f} GB, {(size/baseline_size - 1)*100:+.1f}%)")

        metrics = evaluate_variant(out_path, paths["calibration"])
        metrics["size_gb"] = size
        delta_ppl = metrics["ppl"] - baseline["ppl"]
        print(f"  PPL: {metrics['ppl']} (Δ {delta_ppl:+.0f}) | "
              f"Exact: {metrics['exact_match']}/{metrics['n_samples']} ({metrics['exact_match_pct']}%) | "
              f"Overlap: {metrics['avg_token_overlap']:.3f}")

        results.append({"name": exp_name, **metrics})

        # Clean up variant to save disk
        shutil.rmtree(out_path, ignore_errors=True)
        gc.collect()
        mx.clear_cache()

    # Summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY: {model_name.upper()} PLE/Embed Ablation")
    print(f"{'='*90}")
    print(f"  {'Config':<20s} {'Size':>7s} {'Δ Size':>8s} {'PPL':>8s} {'Δ PPL':>8s} {'Exact%':>7s} {'Overlap':>8s}")
    print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
    for r in results:
        ds = r["size_gb"] - baseline_size
        dp = r["ppl"] - baseline["ppl"]
        print(f"  {r['name']:<20s} {r['size_gb']:>6.2f}G {ds:>+7.2f}G {r['ppl']:>8.0f} {dp:>+8.0f} "
              f"{r['exact_match_pct']:>6.1f}% {r['avg_token_overlap']:>7.3f}")

    # Save results
    out_json = os.path.join(os.path.dirname(__file__), f"ple_ablation_{model_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_json}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e2b", "e4b", "both"], default="both")
    args = parser.parse_args()

    models = ["e2b", "e4b"] if args.model == "both" else [args.model]
    all_results = {}
    for m in models:
        all_results[m] = run_model_ablation(m)

    print("\n✓ PLE ablation complete.")


if __name__ == "__main__":
    main()
