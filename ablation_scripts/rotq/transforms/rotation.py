"""Structured orthogonal rotation search for ROTQ.

Applies foldable orthogonal transforms within norm-free subspaces to
spread anisotropy and outliers more evenly across grouped quantization
coordinates.

Supported families:
  - Hadamard (fixed, deterministic)
  - Signed Hadamard (Hadamard × random diagonal signs)
  - Block Hadamard (blockwise within larger dimensions)
  - Householder products (product of sparse Householder reflectors)

All transforms are O(d log d) or better and exactly foldable.

Independent implementation — does NOT import from turboquant_rotation.py.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx


# ── Core transforms ──────────────────────────────────────────────────────────


def walsh_hadamard_transform(x: mx.array) -> mx.array:
    """Fast Walsh-Hadamard Transform via butterfly operations.

    O(d log d). Input last dimension must be power of 2.
    Normalized by 1/sqrt(d).

    Args:
        x: (..., d) where d is power of 2.

    Returns:
        (..., d) WHT-transformed array.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Dimension must be power of 2, got {d}"

    h = 1
    while h < d:
        x_reshaped = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        even = x_reshaped[..., 0, :]
        odd = x_reshaped[..., 1, :]
        new_even = even + odd
        new_odd = even - odd
        x = mx.stack([new_even, new_odd], axis=-2).reshape(*x.shape[:-1], d)
        h *= 2

    return x * (1.0 / math.sqrt(d))


def inverse_wht(x: mx.array) -> mx.array:
    """Inverse Walsh-Hadamard Transform. WHT is self-inverse (up to scale)."""
    return walsh_hadamard_transform(x)


def random_signs(d: int, seed: int = 42) -> mx.array:
    """Generate random ±1 diagonal vector.

    Args:
        d: Dimension.
        seed: Random seed for reproducibility.

    Returns:
        (d,) array of ±1 values.
    """
    key = mx.random.key(seed)
    mask = mx.random.bernoulli(p=0.5, shape=(d,), key=key)
    return mx.where(mask, mx.array(1.0), mx.array(-1.0))


def signed_hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    """Signed Hadamard: WHT(diag(signs) @ x).

    More effective than plain WHT because the random signs break
    structure-dependent correlations before the Hadamard mixing.

    Args:
        x: (..., d).
        signs: (d,) ±1 diagonal.

    Returns:
        (..., d) transformed array.
    """
    return walsh_hadamard_transform(x * signs)


def inverse_signed_hadamard(x: mx.array, signs: mx.array) -> mx.array:
    """Inverse signed Hadamard: diag(signs) @ WHT(x).

    Since WHT is self-inverse and diag(signs)^{-1} = diag(signs):
        inverse = diag(signs) @ WHT(x)
    """
    return inverse_wht(x) * signs


def block_hadamard_transform(
    x: mx.array, block_size: int, signs: mx.array | None = None
) -> mx.array:
    """Blockwise Hadamard transform.

    Divides the last dimension into blocks of block_size and applies
    WHT (optionally with signs) independently to each block.

    Args:
        x: (..., d) where d is divisible by block_size.
        block_size: Size of each block (must be power of 2).
        signs: Optional (block_size,) sign vector.

    Returns:
        (..., d) block-transformed array.
    """
    d = x.shape[-1]
    assert int(d) % int(block_size) == 0, f"d={d} not divisible by block_size={block_size}"
    n_blocks = d // block_size

    # Reshape to isolate blocks
    x_blocks = x.reshape(*x.shape[:-1], n_blocks, block_size)

    if signs is not None:
        x_blocks = x_blocks * signs

    x_transformed = walsh_hadamard_transform(x_blocks)

    return x_transformed.reshape(*x.shape[:-1], d)


def inverse_block_hadamard(
    x: mx.array, block_size: int, signs: mx.array | None = None
) -> mx.array:
    """Inverse blockwise Hadamard transform."""
    d = x.shape[-1]
    n_blocks = d // block_size

    x_blocks = x.reshape(*x.shape[:-1], n_blocks, block_size)
    x_inv = walsh_hadamard_transform(x_blocks)

    if signs is not None:
        x_inv = x_inv * signs

    return x_inv.reshape(*x.shape[:-1], d)


def householder_reflect(x: mx.array, v: mx.array) -> mx.array:
    """Apply Householder reflection: H(v) @ x = x - 2 * v * (v^T @ x) / (v^T @ v).

    Args:
        x: (..., d) input.
        v: (d,) Householder vector.

    Returns:
        (..., d) reflected array.
    """
    v_f = v.astype(mx.float32)
    x_f = x.astype(mx.float32)
    vtv = mx.sum(v_f * v_f)
    vtx = mx.sum(x_f * v_f, axis=-1, keepdims=True)
    return x_f - 2.0 * v_f * vtx / (vtv + 1e-12)


def householder_product_transform(
    x: mx.array, vectors: list[mx.array]
) -> mx.array:
    """Apply product of Householder reflections: H(v_k) @ ... @ H(v_1) @ x.

    Args:
        x: (..., d) input.
        vectors: List of (d,) Householder vectors.

    Returns:
        (..., d) transformed array.
    """
    result = x.astype(mx.float32)
    for v in vectors:
        result = householder_reflect(result, v)
    return result


def inverse_householder_product(
    x: mx.array, vectors: list[mx.array]
) -> mx.array:
    """Inverse of Householder product: H(v_1) @ ... @ H(v_k) @ x.

    Each Householder reflection is self-inverse, so the inverse of the
    product is the product in reverse order.
    """
    result = x.astype(mx.float32)
    for v in reversed(vectors):
        result = householder_reflect(result, v)
    return result


# ── Rotation application to weight matrices ──────────────────────────────────


def apply_rotation_to_weight(
    w: mx.array,
    rotation_fn: Callable[[mx.array], mx.array],
    axis: str = "input",
) -> mx.array:
    """Apply a rotation to a weight matrix along specified axis.

    Args:
        w: (out_features, in_features) weight matrix.
        rotation_fn: Function that transforms (..., d) → (..., d).
        axis: "input" rotates columns (input space),
              "output" rotates rows (output space).

    Returns:
        Rotated weight matrix.
    """
    w_f = w.astype(mx.float32)

    if axis == "input":
        # Rotate columns: W_new[i, :] = rotation_fn(W[i, :])
        return rotation_fn(w_f)
    elif axis == "output":
        # Rotate rows: W_new[:, j] = rotation_fn(W[:, j])
        # Equivalent to rotation_fn(W^T)^T
        return rotation_fn(w_f.T).T
    else:
        raise ValueError(f"axis must be 'input' or 'output', got {axis}")


# ── Rotation search ──────────────────────────────────────────────────────────


def _make_rotation_candidate(
    family: str,
    dim: int,
    seed: int,
    block_size: int | None = None,
) -> tuple[Callable, Callable]:
    """Create (forward, inverse) rotation functions for a given family.

    Returns:
        (rotate_fn, inverse_fn) pair.
    """
    if family == "hadamard":
        return walsh_hadamard_transform, inverse_wht

    elif family == "signed_hadamard":
        signs = random_signs(dim, seed=seed)
        mx.eval(signs)
        fwd = lambda x: signed_hadamard_transform(x, signs)
        inv = lambda x: inverse_signed_hadamard(x, signs)
        return fwd, inv

    elif family == "block_hadamard":
        bs = block_size or min(128, dim)
        signs = random_signs(bs, seed=seed)
        mx.eval(signs)
        fwd = lambda x: block_hadamard_transform(x, bs, signs)
        inv = lambda x: inverse_block_hadamard(x, bs, signs)
        return fwd, inv

    elif family == "householder":
        # Generate random Householder vectors
        n_reflectors = min(4, dim)
        vectors = []
        for i in range(n_reflectors):
            key = mx.random.key(seed + i)
            v = mx.random.normal(shape=(dim,), key=key)
            v = v / mx.sqrt(mx.sum(v * v) + 1e-12)
            mx.eval(v)
            vectors.append(v)
        fwd = lambda x: householder_product_transform(x, vectors)
        inv = lambda x: inverse_householder_product(x, vectors)
        return fwd, inv

    else:
        raise ValueError(f"Unknown rotation family: {family}")


def search_rotation(
    w: mx.array,
    group_size: int,
    sign_quantize_fn: Callable[[mx.array, int], tuple],
    *,
    families: list[str] | None = None,
    n_candidates: int = 8,
    block_size: int | None = None,
    base_seed: int = 42,
    quantize_bits: int = 1,
) -> tuple[Callable, Callable, str, int, float]:
    """Search over structured rotation candidates for best quantization.

    Args:
        w: (out_features, in_features) weight matrix.
        group_size: Quantization group size.
        sign_quantize_fn: Function(w, group_size) → (packed, scales, biases).
        families: Rotation families to try. Default: all.
        n_candidates: Number of random seeds per family.
        block_size: Block size for block_hadamard family.
        base_seed: Base random seed.
        quantize_bits: Bit-width for dequantization (1 for sign, 2-8 for affine).

    Returns:
        (best_rotate_fn, best_inverse_fn, best_family, best_seed, best_mse).
    """
    from ..metrics import relative_mse as _rel_mse

    if families is None:
        # Check if dimension is power of 2 for full Hadamard
        in_dim = w.shape[1]
        families = ["block_hadamard", "signed_hadamard"]
        if in_dim > 0 and (in_dim & (in_dim - 1)) == 0:
            families.insert(0, "hadamard")
        families.append("householder")

    best_rotate = None
    best_inverse = None
    best_family = "identity"
    best_seed = base_seed
    best_mse = float("inf")

    # Evaluate identity (no rotation) baseline
    packed, scales, biases = sign_quantize_fn(w, group_size)
    w_deq = mx.dequantize(packed, scales, biases, group_size=group_size, bits=quantize_bits)
    mx.eval(w_deq)
    best_mse = _rel_mse(w, w_deq)
    best_rotate = lambda x: x
    best_inverse = lambda x: x

    for family in families:
        seeds = [base_seed] if family == "hadamard" else [
            base_seed + i for i in range(n_candidates)
        ]
        for seed in seeds:
            try:
                in_dim = w.shape[1]
                fwd, inv = _make_rotation_candidate(
                    family, in_dim, seed, block_size=block_size
                )

                # Rotate, quantize, dequantize, inverse-rotate
                w_rot = apply_rotation_to_weight(w, fwd, axis="input")
                mx.eval(w_rot)

                packed, scales, biases = sign_quantize_fn(w_rot, group_size)
                w_deq_rot = mx.dequantize(
                    packed, scales, biases, group_size=group_size, bits=quantize_bits
                )
                w_deq = apply_rotation_to_weight(w_deq_rot, inv, axis="input")
                mx.eval(w_deq)

                mse = _rel_mse(w, w_deq)

                if mse < best_mse:
                    best_mse = mse
                    best_rotate = fwd
                    best_inverse = inv
                    best_family = family
                    best_seed = seed

            except Exception:
                continue

    return best_rotate, best_inverse, best_family, best_seed, best_mse
