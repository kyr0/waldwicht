#!/usr/bin/env python3
"""Phase 2: Configuration knobs sweep for Gemma 4 E2B.

Tests the impact of:
  1. Group size (32, 64, 128) for middle-layer 3-bit weights
  2. Edge layer count (2, 3, 4, 5 layers each side promoted to 4-bit)
  3. Near-edge promotion (keep 3-bit vs promote to 4-bit)

For each config, exports a model, measures size, PPL, and runs 5 chat-templated
generation prompts at temp=0.

E2B has 35 decoder layers (vs E4B's 42), so the edge/near-edge ranges differ:
  - e4: edge=L0-3,L31-34  (8 of 35 = 23%)
  - e4+ne4: above + ne=L4-7,L27-30  (16 of 35 = 46%)

Usage:
    uv run python scripts_E2B/ablation_q3_knobs.py
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts_E2B.e2b_config import (
    BF16_PATH, TMP_DIR, E2B_N_LAYERS, patch_module_policy,
)
from scripts_E2B.eval_utils import (
    PROMPTS, model_size_gb, compute_ppl,
    test_generation, print_generation_results, avg_tok_per_sec,
)

from mlx_lm import load
from rotq.pipeline import RotqConfig, rotq_export
from rotq import module_policy as mp

# ── Patch for E2B ─────────────────────────────────────────────────────────────
patch_module_policy()


# ── Export Helper ─────────────────────────────────────────────────────────────

def export_model(name: str, edge_layers: int, group_size: int,
                 near_edge_4bit: bool = False) -> str:
    """Export a q3-plain variant and return its path."""
    out_path = os.path.join(TMP_DIR, f"e2b-ablation-{name}")

    # Monkey-patch edge layer count
    orig_edge = mp.EDGE_LAYERS
    mp.EDGE_LAYERS = edge_layers

    # If near-edge promotion, monkey-patch classify_weight
    _orig_classify = mp.classify_weight
    if near_edge_4bit:
        def _patched_classify(key, w):
            cls = _orig_classify(key, w)
            if cls == "sign_q1":
                tier = mp.layer_tier(key)
                if tier == "near_edge":
                    return "affine4"
            return cls
        mp.classify_weight = _patched_classify

    config = RotqConfig(
        equalization=False,
        permutation=False,
        rotation=False,
        group_size=group_size,
        quantize_bits=3,
        affine_bits=4,
        affine_group_size=64,
        seed=42,
        verbose=False,
    )

    rotq_export(BF16_PATH, out_path, config)

    # Restore
    mp.EDGE_LAYERS = orig_edge
    mp.classify_weight = _orig_classify

    return out_path


# ── Configurations ────────────────────────────────────────────────────────────

CONFIGS = [
    # (name, edge_layers, group_size, near_edge_4bit)

    # Baseline: e4 with g128 (same ratio as E4B winner pre-NE-promotion)
    ("e4-g128",           4, 128, False),

    # Group size sweep (edge=4 fixed)
    ("e4-g64",            4,  64, False),
    ("e4-g32",            4,  32, False),

    # Edge count sweep (g128 fixed)
    ("e2-g128",           2, 128, False),
    ("e3-g128",           3, 128, False),
    ("e5-g128",           5, 128, False),

    # Near-edge promotion (g128 + near-edge at 4-bit)
    ("e4-g128-ne4bit",    4, 128, True),
    ("e3-g128-ne4bit",    3, 128, True),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n = E2B_N_LAYERS
    print("=" * 90)
    print("GEMMA 4 E2B — PHASE 2: CONFIGURATION KNOBS SWEEP")
    print("=" * 90)
    print(f"  Model: Gemma 4 E2B ({n} decoder layers, hidden=1536)")
    print(f"  BF16:  {BF16_PATH}")
    print(f"  Tmp:   {TMP_DIR}")
    print()

    os.makedirs(TMP_DIR, exist_ok=True)
    all_results = []

    for name, edge, gs, ne4 in CONFIGS:
        print(f"\n{'─' * 90}")
        label = f"{name} (edge={edge}, g{gs}"
        if ne4:
            ne = mp.NEAR_EDGE_LAYERS
            n_4bit = 2 * (edge + ne)
            n_3bit = n - n_4bit
            label += f", near-edge→4bit, {n_4bit} at 4-bit, {n_3bit} at 3-bit"
        else:
            n_4bit = 2 * edge
            n_3bit = n - n_4bit
            label += f", {n_4bit} at 4-bit, {n_3bit} at 3-bit"
        label += ")"
        print(f"CONFIG: {label}")
        print(f"{'─' * 90}")

        # Export
        t0 = time.time()
        path = export_model(name, edge, gs, ne4)
        export_time = time.time() - t0
        size = model_size_gb(path)
        print(f"  Exported in {export_time:.1f}s — {size:.2f} GB")

        # Load
        model, tok = load(path)

        # PPL
        ppl = compute_ppl(model, tok)
        print(f"  PPL = {ppl:.2f}")

        # Generation
        gen_results = test_generation(model, tok, max_tokens=60)
        print_generation_results(gen_results)

        result = {
            "name": name,
            "edge_layers": edge,
            "group_size": gs,
            "near_edge_4bit": ne4,
            "n_4bit_layers": n_4bit,
            "n_3bit_layers": n_3bit,
            "size_gb": round(size, 3),
            "ppl": round(ppl, 2),
            "generation": gen_results,
        }
        all_results.append(result)

        # Free
        del model, tok
        gc.collect()

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("SUMMARY — E2B KNOBS SWEEP")
    print(f"{'=' * 90}")
    print(f"  {'Config':<22} {'Edge':>4} {'GS':>4} {'NE4':>4} {'4bit':>4} {'3bit':>4} "
          f"{'Size':>7} {'PPL':>10} {'Speed':>8}")
    print(f"  {'-'*22} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*7} {'-'*10} {'-'*8}")
    for r in sorted(all_results, key=lambda x: x["ppl"]):
        ne = "yes" if r["near_edge_4bit"] else "no"
        speed = avg_tok_per_sec(r["generation"])
        print(f"  {r['name']:<22} {r['edge_layers']:>4} {r['group_size']:>4} {ne:>4} "
              f"{r['n_4bit_layers']:>4} {r['n_3bit_layers']:>4} "
              f"{r['size_gb']:>6.2f}G {r['ppl']:>10.2f} {speed:>6.1f}t/s")

    # Save JSON
    out_json = os.path.join(os.path.dirname(__file__), "..", "ablation_e2b_q3_knobs.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
