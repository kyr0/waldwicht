import sys, os, time, json
sys.path.insert(0, "calibration-aware-ple")
from infrastructure import (
    BF16_PATH, TMP_DIR, RESULTS_DIR, ensure_dirs,
    export_mixed_quantized, run_answer_quality, cleanup_variant
)

ensure_dirs()

# Test combinations: start with biggest savings (attn+mlp at 3-bit)
combos = [
    # name, default_bits, overrides
    ("combo-attn3-mlp3", 4, {"attention": (3,64), "mlp": (3,64)}),
    ("combo-attn3-mlp3-ple2", 4, {"attention": (3,64), "mlp": (3,64), "ple_embeddings": (2,64)}),
    ("combo-attn3-mlp3-ple2-gate3", 4, {"attention": (3,64), "mlp": (3,64), "ple_embeddings": (2,64), "ple_gate_proj": (3,64)}),
    ("combo-attn3-mlp3-ple2-gate3-embed2", 4, {"attention": (3,64), "mlp": (3,64), "ple_embeddings": (2,64), "ple_gate_proj": (3,64), "main_embed_lmhead": (2,64)}),
]

for name, default_bits, overrides in combos:
    out_path = os.path.join(TMP_DIR, name)
    aq_out = os.path.join(RESULTS_DIR, f"answer_quality_{name}.json")
    
    if os.path.exists(aq_out):
        print(f"\n  [{name}] Already exists, skipping")
        continue
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  default={default_bits}bit, overrides: {overrides}")
    
    t0 = time.time()
    size = export_mixed_quantized(BF16_PATH, out_path, default_bits, 64, overrides)
    print(f"  Exported in {time.time()-t0:.1f}s — {size:.2f} GB")
    
    print(f"  Generating AQ...")
    run_answer_quality(out_path, aq_out)
    print(f"  Done")
    
    cleanup_variant(out_path)

print("\nAll combination variants exported and AQ generated.")