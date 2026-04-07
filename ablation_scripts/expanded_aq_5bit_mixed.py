"""Run expanded AQ on 5-bit mixed configs (relative weighting transfer).

Applies the winning 2.32 GB relative offsets to a 5-bit base:
  Variant A: attn=5, mlp=5, ple=3, gate=4, embed=3  (2.53 GB pattern)
  Variant B: attn=5, mlp=4, ple=3, gate=4, embed=3  (2.32 GB pattern)

Also tests the full winning pattern on 4-bit (for completeness):
  Variant C: attn=4, mlp=3, ple=2, gate=3, embed=2  (= 2.32 GB, already in results)
"""
import sys, os, time, json

sys.path.insert(0, "calibration-aware-ple")
from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, ensure_dirs,
    export_mixed_quantized, run_answer_quality, cleanup_variant,
)

ensure_dirs()

EXPANDED_PROMPTS = os.path.join(
    os.path.dirname(__file__), "..", "answer_quality_expanded.json"
)

CONFIGS = [
    {
        "name": "expanded-aq-5bit-mixed-A-ple3-gate4-embed3",
        "label": "5-bit base, ple3+gate4+embed3 (2.53 GB pattern)",
        "default_bits": 5, "default_gs": 64,
        "overrides": {
            "ple_embeddings": (3, 64),
            "ple_gate_proj": (4, 64),
            "main_embed_lmhead": (3, 64),
        },
    },
    {
        "name": "expanded-aq-5bit-mixed-B-mlp4-ple3-gate4-embed3",
        "label": "5-bit base, mlp4+ple3+gate4+embed3 (2.32 GB pattern)",
        "default_bits": 5, "default_gs": 64,
        "overrides": {
            "mlp": (4, 64),
            "ple_embeddings": (3, 64),
            "ple_gate_proj": (4, 64),
            "main_embed_lmhead": (3, 64),
        },
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
    print(f"Config: {cfg['label']}")
    print(f"  Name: {name}")
    print(f"  Default: {cfg['default_bits']}-bit g{cfg['default_gs']}")
    print(f"  Overrides: {cfg['overrides']}")
    print(f"{'='*60}")

    t0 = time.time()
    size = export_mixed_quantized(
        BF16_PATH, out_path,
        default_bits=cfg["default_bits"],
        default_gs=cfg["default_gs"],
        component_overrides=cfg["overrides"],
    )
    print(f"  Exported: {size:.2f} GB ({time.time()-t0:.1f}s)")

    print(f"  Running expanded AQ (20 prompts)...")
    t1 = time.time()
    run_answer_quality(out_path, aq_out, aq_calibration_path=EXPANDED_PROMPTS)
    print(f"  AQ done ({time.time()-t1:.1f}s)")

    cleanup_variant(out_path)
    print(f"  Cleaned up.")

print("\n\nAll 5-bit mixed configs complete.")
