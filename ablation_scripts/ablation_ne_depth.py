#!/usr/bin/env python3
"""Phase 3: Near-edge depth sweep for Gemma 4 E2B.

Builds on Phase 2 winner. Sweeps:
  1. Near-edge layer count: 2, 3, 4, 6 layers each side
  2. Near-edge group size: g64 vs g128
  3. Cross-checking with runner-up edge count from Phase 2

E2B has 35 decoder layers. With e4+ne4:
  - Edge: L0-3, L31-34 (8 layers at 4-bit)
  - Near-edge: L4-7, L27-30 (8 layers at 4-bit)
  - Middle: L8-26 (19 layers at 3-bit)
  - 46% of layers at 4-bit (vs E4B's 38%)

Note: ne=3 is included since 35 layers don't divide as evenly as E4B's 42.

Usage:
    uv run python scripts_E2B/ablation_ne_depth.py
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
N = E2B_N_LAYERS  # 35


# ── Export Helper ─────────────────────────────────────────────────────────────

def export_model(name: str, edge_layers: int, near_edge_layers: int,
                 near_edge_gs: int = 128) -> str:
    """Export a q3-plain variant with variable near-edge depth/group size."""
    out_path = os.path.join(TMP_DIR, f"e2b-ne-{name}")

    # Save originals
    orig_edge = mp.EDGE_LAYERS
    orig_ne = mp.NEAR_EDGE_LAYERS
    _orig_classify = mp.classify_weight

    mp.EDGE_LAYERS = edge_layers
    mp.NEAR_EDGE_LAYERS = near_edge_layers

    # Patch classify_weight to promote near-edge to affine4
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
        group_size=128,           # middle layers: always g128
        quantize_bits=3,
        affine_bits=4,
        affine_group_size=near_edge_gs,  # both edge and near-edge use this
        seed=42,
        verbose=False,
    )

    rotq_export(BF16_PATH, out_path, config)

    # Restore
    mp.EDGE_LAYERS = orig_edge
    mp.NEAR_EDGE_LAYERS = orig_ne
    mp.classify_weight = _orig_classify

    return out_path


# ── Configurations ────────────────────────────────────────────────────────────

CONFIGS = [
    # (name, edge_layers, near_edge_layers, near_edge_group_size)

    # ── Primary sweep: edge=4 (E4B winner), vary near-edge depth ──
    ("e4-ne2-g128", 4, 2, 128),
    ("e4-ne3-g128", 4, 3, 128),   # E2B-specific: odd depth for 35 layers
    ("e4-ne4-g128", 4, 4, 128),
    ("e4-ne6-g128", 4, 6, 128),

    # ── Near-edge group size: g64 vs g128 ──
    ("e4-ne4-g64",  4, 4, 64),
    ("e4-ne3-g64",  4, 3, 64),    # E2B-specific: test odd depth with g64

    # ── Cross-check with e3 (runner-up candidate) ──
    ("e3-ne4-g64",  3, 4, 64),
    ("e3-ne6-g128", 3, 6, 128),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("GEMMA 4 E2B — PHASE 3: NEAR-EDGE DEPTH ABLATION")
    print("=" * 90)
    print(f"  Model: Gemma 4 E2B ({N} decoder layers, hidden=1536)")
    print(f"  All configs: 3-bit g128 middle, 4-bit edge + near-edge")
    print()

    os.makedirs(TMP_DIR, exist_ok=True)
    all_results = []

    for name, edge, ne, ne_gs in CONFIGS:
        print(f"\n{'─' * 90}")
        ne_end_low = edge + ne
        ne_end_high = N - edge - ne
        label = (f"{name}: edge={edge} (L0–{edge-1}, L{N-edge}–{N-1}), "
                 f"near-edge={ne} (L{edge}–{ne_end_low-1}, L{ne_end_high}–{N-edge-1}), "
                 f"ne_gs={ne_gs}")
        print(f"CONFIG: {label}")

        n_4bit = 2 * (edge + ne)
        n_3bit = N - n_4bit
        pct_4bit = n_4bit / N * 100
        print(f"  Layers: {n_4bit} at 4-bit ({pct_4bit:.0f}%), {n_3bit} at 3-bit")
        print(f"{'─' * 90}")

        # Export
        t0 = time.time()
        path = export_model(name, edge, ne, ne_gs)
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
            "near_edge_layers": ne,
            "near_edge_gs": ne_gs,
            "n_4bit_layers": n_4bit,
            "n_3bit_layers": n_3bit,
            "pct_4bit": round(pct_4bit, 1),
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
    print("SUMMARY — E2B NEAR-EDGE DEPTH ABLATION")
    print(f"{'=' * 90}")
    print(f"  {'Config':<18} {'Edge':>4} {'NE':>3} {'GS':>4} {'4bit':>4} {'3bit':>4} "
          f"{'%4bit':>5} {'Size':>7} {'PPL':>10} {'Speed':>8}")
    print(f"  {'-'*18} {'-'*4} {'-'*3} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*5} {'-'*7} {'-'*10} {'-'*8}")
    for r in sorted(all_results, key=lambda x: x["ppl"]):
        speed = avg_tok_per_sec(r["generation"])
        print(f"  {r['name']:<18} {r['edge_layers']:>4} {r['near_edge_layers']:>3} "
              f"{r['near_edge_gs']:>4} {r['n_4bit_layers']:>4} {r['n_3bit_layers']:>4} "
              f"{r['pct_4bit']:>4.0f}% {r['size_gb']:>6.2f}G {r['ppl']:>10.2f} "
              f"{speed:>6.1f}t/s")

    # Save JSON
    out_json = os.path.join(os.path.dirname(__file__), "..", "ablation_e2b_ne_depth.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
