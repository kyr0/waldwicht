"""Run expanded AQ benchmark on the winning 2.32 GB config.

Re-exports the optimal config, generates responses to 20 expanded prompts
(function calling, code, translation, reasoning, creative writing),
then cleans up.
"""
import sys, os, time, json
sys.path.insert(0, "calibration-aware-ple")
from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, ensure_dirs,
    export_mixed_quantized, run_answer_quality, cleanup_variant
)

ensure_dirs()

EXPANDED_PROMPTS = os.path.join(
    os.path.dirname(__file__), "..", "answer_quality_expanded.json"
)
VARIANT_NAME = "expanded-aq-optimal-2.32gb"
OUT_PATH = os.path.join(TMP_DIR, VARIANT_NAME)
AQ_OUT = os.path.join(RESULTS_DIR, f"answer_quality_{VARIANT_NAME}.json")

if os.path.exists(AQ_OUT):
    print(f"Already exists: {AQ_OUT}")
    sys.exit(0)

# Re-export the winning config
print("Exporting optimal 2.32 GB config...")
t0 = time.time()
size = export_mixed_quantized(
    BF16_PATH, OUT_PATH,
    default_bits=4, default_gs=64,
    component_overrides={
        "mlp": (3, 64),
        "ple_embeddings": (2, 64),
        "ple_gate_proj": (3, 64),
        "main_embed_lmhead": (2, 64),
    }
)
print(f"  {size:.2f} GB ({time.time()-t0:.1f}s)")

# Run expanded AQ
print(f"Running expanded AQ ({EXPANDED_PROMPTS})...")
run_answer_quality(OUT_PATH, AQ_OUT, aq_calibration_path=EXPANDED_PROMPTS)
print("  Expanded AQ done")

cleanup_variant(OUT_PATH)
print("Done.")
