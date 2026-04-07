#!/usr/bin/env python3
"""Step 1: Uniform bit-width ladder (top-down sieve).

Quantize the entire E2B model uniformly at 6→5→4→3→2 bits.
At each rung, generate answer-quality responses for manual LLM-judge scoring.
Only proceed to the next lower rung if the previous one passes.

Usage:
    .venv/bin/python -m calibration-aware-ple.step1_uniform_ladder [--start-bits 6]
    # or directly:
    .venv/bin/python calibration-aware-ple/step1_uniform_ladder.py [--start-bits 6]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, BIT_LADDER,
    ensure_dirs, model_size_gb, evaluate_variant,
    export_uniform_quantized, run_answer_quality,
    check_quality_gate, log_result, print_result, cleanup_variant,
)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Uniform bit-width ladder")
    parser.add_argument("--start-bits", type=int, default=6,
                        help="Starting bit-width (default: 6)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Group size for quantization (default: 64)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip 64-prompt eval (only do answer quality generation)")
    parser.add_argument("--auto-gate", action="store_true",
                        help="If scored AQ files exist, auto-check gate and continue")
    args = parser.parse_args()

    ensure_dirs()
    results_path = os.path.join(RESULTS_DIR, "uniform_ladder_e2b.json")

    # Filter ladder to start from requested bits
    ladder = [b for b in BIT_LADDER if b <= args.start_bits]

    print(f"{'#' * 70}")
    print(f"  Step 1: Uniform Bit-Width Ladder (E2B)")
    print(f"  BF16 source: {BF16_PATH}")
    print(f"  Ladder: {' → '.join(str(b) for b in ladder)} bits")
    print(f"  Group size: {args.group_size}")
    print(f"{'#' * 70}")

    for bits in ladder:
        rung_name = f"uniform-{bits}bit-g{args.group_size}"
        out_path = os.path.join(TMP_DIR, rung_name)
        aq_out = os.path.join(RESULTS_DIR, f"answer_quality_{rung_name}.json")

        print(f"\n{'─' * 60}")
        print(f"  Rung: {bits}-bit g{args.group_size}")
        print(f"{'─' * 60}")

        # ── Export ────────────────────────────────────────────────────
        t0 = time.time()
        size = export_uniform_quantized(BF16_PATH, out_path, bits, args.group_size)
        print(f"  Exported in {time.time() - t0:.1f}s — {size:.2f} GB")

        # ── 64-prompt calibration eval ────────────────────────────────
        if not args.skip_eval:
            t0 = time.time()
            metrics = evaluate_variant(out_path)
            metrics["size_gb"] = size
            print(f"  Evaluated in {time.time() - t0:.1f}s")
            print_result(rung_name, size, metrics)
        else:
            metrics = {"size_gb": size, "ppl": None, "exact_match_pct": None,
                       "avg_token_overlap": None, "n_samples": None, "exact_match": None}

        # ── Answer quality generation (8 prompts) ────────────────────
        print(f"  Generating answer quality responses...")
        run_answer_quality(out_path, aq_out)

        # ── Log result ────────────────────────────────────────────────
        entry = {"name": rung_name, "bits": bits, "group_size": args.group_size, **metrics}
        log_result(results_path, entry)

        # ── Cleanup variant (keep results) ────────────────────────────
        cleanup_variant(out_path)

        # ── Check quality gate if auto mode and scored file exists ────
        if args.auto_gate and os.path.exists(aq_out):
            gate = check_quality_gate(aq_out)
            if gate["passed"]:
                print(f"  ✓ Quality gate PASSED: {gate['averages']}")
                print(f"  → Proceeding to next rung...")
            else:
                print(f"  ✗ Quality gate FAILED: {gate['averages']}")
                print(f"  → STOPPING. Floor is {bits}-bit (last passing rung).")
                break
        else:
            print(f"\n  ⏸  Manual quality gate required.")
            print(f"     Score the file: {aq_out}")
            print(f"     Then re-run with --auto-gate --start-bits {bits - 1 if bits > 2 else bits}")

            # If this is not the last rung, pause to let user score
            if bits != ladder[-1]:
                print(f"     (fill in correctness/completion/reasoning_hygiene scores,")
                print(f"      then re-run to continue the ladder)")
                break

    print(f"\n{'=' * 70}")
    print(f"  Results saved to {results_path}")
    print(f"  Answer quality files in {RESULTS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
