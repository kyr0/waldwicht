import sys, os, time, json
sys.path.insert(0, "calibration-aware-ple")
from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, ensure_dirs,
    export_mixed_quantized, run_answer_quality, cleanup_variant
)

ensure_dirs()

# Strategy: demote ONLY one of the big groups (attn/mlp) + small components
combos = [
    # MLP at 3-bit + small components demoted, attention stays at 4-bit
    ("combo-mlp3-ple2-gate3-embed2", 4, {
        "mlp": (3,64), "ple_embeddings": (2,64),
        "ple_gate_proj": (3,64), "main_embed_lmhead": (2,64)
    }),
    # Attention at 3-bit + small components demoted, MLP stays at 4-bit
    ("combo-attn3-ple2-gate3-embed2", 4, {
        "attention": (3,64), "ple_embeddings": (2,64),
        "ple_gate_proj": (3,64), "main_embed_lmhead": (2,64)
    }),
    # Only small components demoted, both attention+mlp at 4-bit
    ("combo-ple2-gate3-embed2", 4, {
        "ple_embeddings": (2,64), "ple_gate_proj": (3,64),
        "main_embed_lmhead": (2,64)
    }),
    # Just MLP at 3-bit, nothing else changed
    ("combo-mlp3-only", 4, {"mlp": (3,64)}),
]

for name, default_bits, overrides in combos:
    out_path = os.path.join(TMP_DIR, name)
    aq_out = os.path.join(RESULTS_DIR, f"answer_quality_{name}.json")
    
    if os.path.exists(aq_out):
        print(f"\n  [{name}] Already exists, skipping")
        continue
    
    print(f"\n  {name}: default={default_bits}bit, {overrides}")
    t0 = time.time()
    size = export_mixed_quantized(BF16_PATH, out_path, default_bits, 64, overrides)
    print(f"  {size:.2f} GB ({time.time()-t0:.1f}s)")
    
    run_answer_quality(out_path, aq_out)
    print(f"  AQ done")
    cleanup_variant(out_path)

print("\nDone.")