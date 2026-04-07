"""Gemma 4 E2B architecture constants and module-policy patching.

E2B architecture (from HuggingFace config.json):
  - 35 decoder layers (vs E4B's 42)
  - hidden_size: 1536 (vs 2560)
  - intermediate_size: 6144 (vs 10240)
  - num_attention_heads: 8,  num_key_value_heads: 1 (vs 8/2)
  - head_dim: 256 (sliding), 512 (global)
  - layer_types: 4 sliding + 1 full, repeating ×7
  - num_kv_shared_layers: 20
  - PLE: 35 × Embedding(262144, 256) + gating
  - per_layer_model_projection: (8960, 1536)  [35 × 256]
  - All dims divisible by 128: 1536/128=12 ✓
  - Total params: 5.1B (2.3B effective)
  - BF16 size: ~10 GB
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rotq import module_policy as mp

# ── E2B Architecture Constants ────────────────────────────────────────────────

E2B_N_LAYERS = 35
E2B_HIDDEN_SIZE = 1536
E2B_INTERMEDIATE_SIZE = 6144

# ── Paths ─────────────────────────────────────────────────────────────────────

BF16_PATH = "hf/gemma-4-E2B-it"
OUTPUT_BASE = "mlx/gemma-4-E2B-ablation"
TMP_DIR = "tmp-e2b"


def patch_module_policy(n_layers: int = E2B_N_LAYERS):
    """Monkey-patch rotq.module_policy for E2B's 35-layer architecture.

    Must be called before any classify_weight() or layer_tier() calls.
    Consistent with the E4B ablation pattern where scripts patch mp.EDGE_LAYERS.
    """
    mp.N_LAYERS = n_layers
