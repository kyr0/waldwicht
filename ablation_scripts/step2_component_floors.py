#!/usr/bin/env python3
"""Step 2: Per-component quality floors.

Starting from the uniform floor established in Step 1, demote one component
group at a time one rung below. If a group passes at floor-1, test floor-2,
etc. This finds each component's independent minimum bit-width.

Component groups:
  - ple_embeddings:    embed_tokens_per_layer (35 tables)
  - attention:         q_proj, k_proj, v_proj, o_proj
  - mlp:              gate_proj, up_proj, down_proj
  - ple_gate_proj:    per_layer_input_gate, per_layer_projection
  - main_embed_lmhead: embed_tokens, lm_head

Usage:
    .venv/bin/python calibration-aware-ple/step2_component_floors.py --floor-bits 4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, VALID_BITS,
    COMPONENT_GROUPS, ensure_dirs,
    evaluate_variant, export_mixed_quantized,
    run_answer_quality, check_quality_gate,
    log_result, print_result, cleanup_variant,
)


def bits_below(current: int) -> list[int]:
    """Return valid bit-widths strictly below `current`, descending."""
    return sorted([b for b in VALID_BITS if b < current], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Per-component quality floors")
    parser.add_argument("--floor-bits", type=int, required=True,
                        help="Uniform quality floor from Step 1 (e.g. 4)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Group size (default: 64)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip 64-prompt eval (only do AQ generation)")
    parser.add_argument("--components", nargs="+", default=None,
                        help="Only test these component groups (default: all)")
    args = parser.parse_args()

    ensure_dirs()
    results_path = os.path.join(RESULTS_DIR, "component_floors_e2b.json")

    components = args.components or list(COMPONENT_GROUPS.keys())

    print(f"{'#' * 70}")
    print(f"  Step 2: Per-Component Quality Floors (E2B)")
    print(f"  Uniform floor: {args.floor_bits}-bit g{args.group_size}")
    print(f"  Components: {', '.join(components)}")
    print(f"{'#' * 70}")

    # For each component, try demoting one rung at a time
    for comp_name in components:
        lower_bits = bits_below(args.floor_bits)
        if not lower_bits:
            print(f"\n  {comp_name}: already at minimum bit-width, skipping")
            continue

        print(f"\n{'═' * 60}")
        print(f"  Component: {comp_name}  ({', '.join(COMPONENT_GROUPS[comp_name])})")
        print(f"{'═' * 60}")

        comp_floor = args.floor_bits  # Will be lowered if demotion passes

        for test_bits in lower_bits:
            rung_name = f"{comp_name}-{test_bits}bit-g{args.group_size}"
            out_path = os.path.join(TMP_DIR, rung_name)
            aq_out = os.path.join(RESULTS_DIR, f"answer_quality_{rung_name}.json")

            # Resume: if AQ file already exists and is scored, skip export/eval
            if os.path.exists(aq_out):
                gate = check_quality_gate(aq_out)
                if gate.get("all_scored", False):
                    if gate["passed"]:
                        print(f"\n  [resume] {comp_name} at {test_bits}-bit "
                              f"already scored → PASSED: {gate['averages']}")
                        comp_floor = test_bits
                        continue
                    else:
                        print(f"\n  [resume] {comp_name} at {test_bits}-bit "
                              f"already scored → FAILED: {gate['averages']}")
                        break

            print(f"\n  Testing {comp_name} at {test_bits}-bit "
                  f"(rest at {args.floor_bits}-bit)...")

            overrides = {comp_name: (test_bits, args.group_size)}

            t0 = time.time()
            size = export_mixed_quantized(
                BF16_PATH, out_path,
                default_bits=args.floor_bits,
                default_gs=args.group_size,
                component_overrides=overrides,
            )
            print(f"  Exported in {time.time() - t0:.1f}s — {size:.2f} GB")

            if not args.skip_eval:
                t0 = time.time()
                metrics = evaluate_variant(out_path)
                metrics["size_gb"] = size
                print(f"  Evaluated in {time.time() - t0:.1f}s")
                print_result(rung_name, size, metrics)
            else:
                metrics = {"size_gb": size, "ppl": None, "exact_match_pct": None,
                           "avg_token_overlap": None, "n_samples": None, "exact_match": None}

            # Generate AQ
            print(f"  Generating answer quality responses...")
            run_answer_quality(out_path, aq_out)

            entry = {
                "name": rung_name,
                "component": comp_name,
                "test_bits": test_bits,
                "floor_bits": args.floor_bits,
                "group_size": args.group_size,
                **metrics,
            }
            log_result(results_path, entry)

            cleanup_variant(out_path)

            # Check gate if scores already filled in
            if os.path.exists(aq_out):
                gate = check_quality_gate(aq_out)
                if not gate.get("all_scored", False):
                    # Scores not yet filled in
                    print(f"\n  ⏸  Score {aq_out} then re-run")
                    break
                elif gate["passed"]:
                    print(f"  ✓ {comp_name} at {test_bits}-bit PASSED: {gate['averages']}")
                    comp_floor = test_bits
                    print(f"    → Trying {comp_name} at next lower rung...")
                    continue
                else:
                    print(f"  ✗ {comp_name} at {test_bits}-bit FAILED: {gate['averages']}")
                    print(f"    → {comp_name} floor = {comp_floor}-bit")
                    break

        print(f"\n  ▸ {comp_name} floor: {comp_floor}-bit")

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {results_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
