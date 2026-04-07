"""Module-specific transform policies for Gemma 4 ROTQ.

Different modules in the Gemma 4 architecture have different folding
constraints and quantization sensitivity.  This module defines per-family
policies and identifies foldable layer pairs.

Architecture constants (Gemma 4 E4B):
  - 42 decoder layers
  - Edge layers (0-1, 40-41): more sensitive → 4-bit affine, no transforms
  - Embeddings / PLE: 4-bit affine, no transforms
  - Middle decoder layers: full transform stack

Foldable subspaces:
  - MLP intermediate (gate/up output → down input): NO norm in between,
    BUT gate*up elementwise product prevents full rotation.
    Only permutation is safe here.
  - Attention head space: rotation within each head is compatible with
    scaled dot-product attention.
  - Hidden space across layers: diagonal scaling (equalization) folds
    through RMSNorm; full rotation does NOT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


# ── Architecture constants ───────────────────────────────────────────────────

N_LAYERS = 42  # Gemma 4 E4B decoder layers
EDGE_LAYERS = 4  # Edge layers each side → affine4 (most sensitive)
NEAR_EDGE_LAYERS = 4  # Near-edge layers each side → sign_q1 with extra rotation budget


# ── Module transform policy ──────────────────────────────────────────────────


@dataclass
class ModulePolicy:
    """Transform policy for a module family."""

    # What to do with this module
    action: str = "sign_q1"  # "sign_q1", "affine4", "skip"

    # Transform toggles
    equalization: bool = False
    permutation: bool = False
    rotation: bool = False

    # Transform parameters
    equalization_method: str = "abs_max"
    permutation_methods: list[str] = field(
        default_factory=lambda: ["magnitude", "abs_mean", "variance"]
    )
    rotation_families: list[str] = field(
        default_factory=lambda: ["block_hadamard", "signed_hadamard"]
    )
    rotation_candidates: int = 8
    rotation_block_size: Optional[int] = 128

    # Scale method for sign quantization
    scale_method: str = "mean_abs"

    # Group size
    group_size: int = 128


# ── Weight classification ────────────────────────────────────────────────────

_SKIP_SUBSTRINGS = (
    "norm.weight",
    "norm.bias",
    "layer_scalar",
)

_AFFINE4_SUBSTRINGS = (
    "embed_tokens",
    "lm_head",
    "per_layer_model_projection",
    "per_layer_input_gate",
)

_ATTENTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def _layer_index(key: str) -> Optional[int]:
    """Extract decoder layer index from weight key."""
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def classify_weight(key: str, w: mx.array) -> str:
    """Classify a weight for the ROTQ pipeline.

    Returns: "sign_q1", "affine4", or "skip".
    """
    if w.ndim < 2:
        return "skip"

    # Skip non-text modality weights (audio tower, vision tower) and
    # weights with ndim > 2 (conv kernels, position embeddings)
    if w.ndim > 2:
        return "skip"
    if "audio_tower" in key or "vision_tower" in key:
        return "skip"

    for sub in _SKIP_SUBSTRINGS:
        if sub in key:
            return "skip"

    for sub in _AFFINE4_SUBSTRINGS:
        if sub in key:
            return "affine4"

    if not key.endswith(".weight"):
        return "skip"

    # Edge layers → 4-bit affine (most sensitive to quantization)
    idx = _layer_index(key)
    if idx is not None and (idx < EDGE_LAYERS or idx >= N_LAYERS - EDGE_LAYERS):
        return "affine4"

    return "sign_q1"


def layer_tier(key: str) -> str:
    """Classify layer position into edge / near_edge / middle.

    Based on APEX finding that edge layers are disproportionately
    sensitive to quantization noise.  For our 42-layer model:
      - edge (0-3, 38-41):       affine4, no transforms
      - near_edge (4-7, 34-37):  sign_q1 with more rotation budget
      - middle (8-33):           sign_q1 with standard rotation budget
    """
    idx = _layer_index(key)
    if idx is None:
        return "middle"  # non-layer weights default to middle treatment
    if idx < EDGE_LAYERS or idx >= N_LAYERS - EDGE_LAYERS:
        return "edge"
    if idx < EDGE_LAYERS + NEAR_EDGE_LAYERS or idx >= N_LAYERS - EDGE_LAYERS - NEAR_EDGE_LAYERS:
        return "near_edge"
    return "middle"


def module_family(key: str) -> str:
    """Identify which module family a weight belongs to.

    Returns one of: "attention", "mlp", "embedding", "other".
    """
    suffix = key.rsplit(".", 1)[-1] if "." in key else key

    # Check without .weight suffix
    clean_key = key[:-len(".weight")] if key.endswith(".weight") else key
    clean_suffix = clean_key.rsplit(".", 1)[-1] if "." in clean_key else clean_key

    if clean_suffix in _ATTENTION_SUFFIXES:
        return "attention"
    if clean_suffix in _MLP_SUFFIXES:
        return "mlp"
    if any(sub in key for sub in ("embed_tokens", "lm_head")):
        return "embedding"
    return "other"


# ── Default policies ─────────────────────────────────────────────────────────


def default_gemma4_policies() -> dict[str, ModulePolicy]:
    """Default ROTQ policies for Gemma 4 E4B module families.

    Returns:
        Dict mapping module family → ModulePolicy.
        Keys are "family" or "family:tier" for position-aware overrides.
    """
    return {
        # Embeddings: 4-bit affine, no transforms
        "embedding": ModulePolicy(
            action="affine4",
            equalization=False,
            permutation=False,
            rotation=False,
            group_size=64,
        ),
        # ── Attention: middle layers (default) ──
        "attention": ModulePolicy(
            action="sign_q1",
            equalization=True,
            permutation=True,
            rotation=True,
            rotation_families=["block_hadamard"],
            rotation_candidates=4,
            rotation_block_size=128,
            scale_method="mean_abs",
        ),
        # ── Attention: near-edge layers — NO rotation ──
        # Near-edge k_proj/v_proj already have baseline MSE ~0.322 which is
        # BELOW the rotation floor (~0.36). Rotation destroys their naturally
        # favorable sign distribution (-12% worse). Empirically validated.
        "attention:near_edge": ModulePolicy(
            action="sign_q1",
            equalization=True,
            permutation=True,
            rotation=False,
            scale_method="mean_abs",
        ),
        # ── MLP: middle layers (default) ──
        # No rotation due to gate*up elementwise product
        "mlp": ModulePolicy(
            action="sign_q1",
            equalization=True,
            permutation=True,
            rotation=False,
            scale_method="mean_abs",
        ),
        # ── MLP: near-edge layers ──
        "mlp:near_edge": ModulePolicy(
            action="sign_q1",
            equalization=True,
            permutation=True,
            rotation=False,
            scale_method="mean_abs",
        ),
        # Edge layers & other: 4-bit affine
        "other": ModulePolicy(
            action="affine4",
            equalization=False,
            permutation=False,
            rotation=False,
            group_size=64,
        ),
    }


def get_policy_for_weight(
    key: str,
    w: mx.array,
    policies: dict[str, ModulePolicy] | None = None,
) -> ModulePolicy:
    """Get the transform policy for a specific weight.

    Args:
        key: Weight key string.
        w: Weight tensor.
        policies: Policy dict. If None, uses default_gemma4_policies().

    Returns:
        ModulePolicy for this weight.
    """
    if policies is None:
        policies = default_gemma4_policies()

    classification = classify_weight(key, w)
    if classification == "skip":
        return ModulePolicy(action="skip")
    if classification == "affine4":
        return policies.get("embedding", ModulePolicy(action="affine4", group_size=64))

    # For sign_q1, look up by module family with tier override
    family = module_family(key)
    tier = layer_tier(key)

    # Try tier-specific policy first, then fall back to family default
    tier_key = f"{family}:{tier}"
    if tier_key in policies:
        return policies[tier_key]
    return policies.get(family, policies.get("other", ModulePolicy()))


# ── Foldable layer pairs ────────────────────────────────────────────────────


def identify_equalization_pairs(n_layers: int = N_LAYERS) -> list[tuple[str, str]]:
    """Identify weight pairs for cross-layer equalization.

    Returns pairs (producer_key, consumer_key) where both share a hidden
    dimension that can be equalized via diagonal scaling.

    Pairs:
      - Within each layer's MLP: (gate_proj, down_proj) and (up_proj, down_proj)
        share the intermediate dimension. But gate*up prevents simple pairwise
        equalization. We equalize down_proj input with gate/up output individually.
      - Across layers: o_proj output feeds (through RMSNorm) into next layer's
        q/k/v_proj. Equalization through RMSNorm is approximate but effective.
    """
    pairs = []
    prefix = "language_model.model.layers"

    for i in range(n_layers):
        # MLP: equalize gate/up outputs with down input
        # gate_proj: (intermediate, hidden) — output dim = intermediate
        # up_proj: (intermediate, hidden)
        # down_proj: (hidden, intermediate) — input dim = intermediate
        # The shared dimension is intermediate (columns of gate/up = columns of down after transpose)
        # Actually gate_proj.weight shape is (intermediate, hidden), down_proj.weight shape is (hidden, intermediate)
        # For equalization of intermediate space: gate_proj rows ↔ down_proj columns
        gate_key = f"{prefix}.{i}.mlp.gate_proj.weight"
        up_key = f"{prefix}.{i}.mlp.up_proj.weight"
        down_key = f"{prefix}.{i}.mlp.down_proj.weight"
        pairs.append((gate_key, down_key))
        pairs.append((up_key, down_key))

    return pairs


def identify_permutation_groups(n_layers: int = N_LAYERS) -> list[dict]:
    """Identify groups of weights that must share a permutation.

    When permuting a hidden dimension, all matrices reading from or
    writing to that dimension must agree on the same permutation.

    Returns a list of dicts:
      {
        "space": "mlp_intermediate" or "hidden" or "head",
        "layer": int,
        "column_keys": [keys whose columns share this space],
        "row_keys": [keys whose rows share this space],
      }
    """
    groups = []
    prefix = "language_model.model.layers"

    for i in range(n_layers):
        # MLP intermediate space:
        #   gate_proj (intermediate, hidden) — rows = intermediate
        #   up_proj (intermediate, hidden) — rows = intermediate
        #   down_proj (hidden, intermediate) — columns = intermediate
        groups.append({
            "space": "mlp_intermediate",
            "layer": i,
            "row_keys": [
                f"{prefix}.{i}.mlp.gate_proj.weight",
                f"{prefix}.{i}.mlp.up_proj.weight",
            ],
            "column_keys": [
                f"{prefix}.{i}.mlp.down_proj.weight",
            ],
        })

    return groups
