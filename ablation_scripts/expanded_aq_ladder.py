"""Run expanded AQ benchmark on larger configs that passed the math gate.

Tests three configs in ascending size order:
  1. ple2+gate3+embed2  (2.53 GB) — attn=4, mlp=4, ple=2, gate=3, embed=2
  2. uniform-4bit-g64   (3.22 GB) — all 4-bit
  3. uniform-5bit-g64   (3.86 GB) — all 5-bit
"""
import sys, os, time, json

sys.path.insert(0, "calibration-aware-ple")
from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, ensure_dirs,
    export_mixed_quantized, export_uniform_quantized,
    run_answer_quality, cleanup_variant,
)

ensure_dirs()

EXPANDED_PROMPTS = os.path.join(
    os.path.dirname(__file__), "..", "answer_quality_expanded.json"
)

CONFIGS = [
    {
        "name": "expanded-aq-ple2-gate3-embed2-2.53gb",
        "type": "mixed",
        "default_bits": 4, "default_gs": 64,
        "overrides": {
            "ple_embeddings": (2, 64),
            "ple_gate_proj": (3, 64),
            "main_embed_lmhead": (2, 64),
        },
    },
    {
        "name": "expanded-aq-uniform-4bit-g64",
        "type": "uniform",
        "bits": 4, "gs": 64,
    },
    {
        "name": "expanded-aq-uniform-5bit-g64",
        "type": "uniform",
        "bits": 5, "gs": 64,
    },
]

for cfg in CONFIGS:
    name = cfg["name"]
    aq_out = os.path.join(RESULTS_DIR, f"answer_quality_{name}.json")

    if os.path.exists(aq_out):
        print(f"[SKIP] {name} — already exists")
        continue

    out_path = os.path.join(TMP_DIR, name)

    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    # Export
    t0 = time.time()
    if cfg["type"] == "mixed":
        size = export_mixed_quantized(
            BF16_PATH, out_path,
            default_bits=cfg["default_bits"],
            default_gs=cfg["default_gs"],
            component_overrides=cfg["overrides"],
        )
    else:
        size = export_uniform_quantized(
            BF16_PATH, out_path,
            bits=cfg["bits"],
            group_size=cfg["gs"],
        )
    print(f"  Exported: {size:.2f} GB ({time.time()-t0:.1f}s)")

    # Run expanded AQ
    print(f"  Running expanded AQ (20 prompts)...")
    t1 = time.time()
    run_answer_quality(out_path, aq_out, aq_calibration_path=EXPANDED_PROMPTS)
    print(f"  AQ done ({time.time()-t1:.1f}s)")

    cleanup_variant(out_path)
    print(f"  Cleaned up.")

print("\n\nAll configs complete.")
