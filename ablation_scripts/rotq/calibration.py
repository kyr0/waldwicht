"""Calibration data loading and activation capture for ROTQ.

Provides utilities to:
  - Load calibration text from a local file or generate simple defaults
  - Run forward passes to capture hidden states at layer boundaries
  - Capture final logits for quality evaluation
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx


def load_calibration_tokens(
    tokenizer,
    *,
    text_path: Optional[str] = None,
    n_samples: int = 32,
    seq_len: int = 512,
) -> mx.array:
    """Load and tokenize calibration text.

    Args:
        tokenizer: HuggingFace-style tokenizer with .encode() method.
        text_path: Path to a text file. If None, uses a built-in default.
        n_samples: Number of samples to return.
        seq_len: Sequence length per sample.

    Returns:
        (n_samples, seq_len) int32 token array.
    """
    if text_path is not None:
        text = Path(text_path).read_text(encoding="utf-8")
    else:
        text = _DEFAULT_CALIBRATION_TEXT

    # Tokenize the entire text
    token_ids = tokenizer.encode(text)
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)

    total_needed = n_samples * seq_len
    if len(token_ids) < total_needed:
        # Repeat to fill
        repeats = (total_needed // len(token_ids)) + 1
        token_ids = (token_ids * repeats)[:total_needed]
    else:
        token_ids = token_ids[:total_needed]

    tokens = mx.array(token_ids, dtype=mx.int32).reshape(n_samples, seq_len)
    return tokens


def capture_layer_inputs(
    model,
    tokens: mx.array,
    layer_indices: list[int] | None = None,
) -> dict[int, mx.array]:
    """Run forward pass and capture inputs to specified decoder layers.

    Uses MLX's module hooks to intercept activations at layer boundaries.
    Returns a dict mapping layer_index → activation tensor.

    Args:
        model: An mlx_lm model (must have .language_model.layers).
        tokens: (batch, seq_len) token ids.
        layer_indices: Which layers to capture. If None, captures all.

    Returns:
        Dict mapping layer index → (batch, seq_len, hidden_size) activation.
    """
    lang_model = model.language_model if hasattr(model, "language_model") else model
    layers = lang_model.layers

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    captured = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inputs):
            # DecoderLayer.__call__ receives (x, ...) — capture x
            if isinstance(inputs, tuple) and len(inputs) > 0:
                captured[idx] = inputs[0]
            elif isinstance(inputs, mx.array):
                captured[idx] = inputs
        return hook

    # Register hooks
    for idx in layer_indices:
        h = layers[idx].register_input_hook(make_hook(idx))
        hooks.append(h)

    try:
        _ = model(tokens)
        mx.eval(list(captured.values()))
    finally:
        for h in hooks:
            h.remove()

    return captured


def capture_logits(model, tokens: mx.array) -> mx.array:
    """Run forward pass and return logits.

    Args:
        model: An mlx_lm model.
        tokens: (batch, seq_len) token ids.

    Returns:
        (batch, seq_len, vocab_size) logits.
    """
    logits = model(tokens)
    mx.eval(logits)
    return logits


def capture_layer_outputs(
    model,
    tokens: mx.array,
    layer_indices: list[int] | None = None,
) -> dict[int, mx.array]:
    """Run forward pass and capture outputs from specified decoder layers.

    Uses output hooks to intercept the return value of each layer.

    Args:
        model: An mlx_lm model (must have .language_model.layers).
        tokens: (batch, seq_len) token ids.
        layer_indices: Which layers to capture. If None, captures all.

    Returns:
        Dict mapping layer index → (batch, seq_len, hidden_size) output.
    """
    lang_model = model.language_model if hasattr(model, "language_model") else model
    layers = lang_model.layers

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    captured = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inputs, output):
            # DecoderLayer returns hidden states
            if isinstance(output, tuple):
                captured[idx] = output[0]
            else:
                captured[idx] = output
        return hook

    for idx in layer_indices:
        h = layers[idx].register_output_hook(make_hook(idx))
        hooks.append(h)

    try:
        _ = model(tokens)
        mx.eval(list(captured.values()))
    finally:
        for h in hooks:
            h.remove()

    return captured


# ── Default calibration text ─────────────────────────────────────────────────

_DEFAULT_CALIBRATION_TEXT = """The field of artificial intelligence has seen tremendous progress in recent years, particularly in the area of large language models. These models are trained on vast amounts of text data and can generate human-like text, answer questions, and perform a variety of language tasks.

One of the key challenges in deploying large language models is their significant computational and memory requirements. A model with billions of parameters requires substantial GPU memory and processing power for inference. This has led to extensive research in model compression techniques, including quantization, pruning, and knowledge distillation.

Quantization reduces the precision of model weights from floating-point representations to lower-bit formats. For example, converting 32-bit floating-point weights to 8-bit integers can reduce memory usage by 4x while maintaining reasonable quality. More aggressive quantization to 4-bit, 2-bit, or even 1-bit representations offers even greater compression but presents significant quality challenges.

The key insight behind effective low-bit quantization is that not all weights are equally important, and the distribution of weight values within a neural network layer has significant structure that can be exploited. Modern quantization methods go beyond simple rounding by considering the statistical properties of weights and their impact on model outputs.

Group quantization divides weight matrices into small groups (typically 32 to 128 elements) and assigns separate scaling factors to each group. This allows the quantizer to adapt to local variations in magnitude across the weight matrix. The choice of group size represents a tradeoff between compression ratio and approximation quality.

Rotation-based quantization methods apply orthogonal transformations to weight matrices before quantization. The key observation is that the same mathematical function can be represented in many different coordinate systems, and some coordinate systems are much more amenable to quantization than others. By finding a good rotation, we can spread outlier values more evenly across coordinates, making the weight distribution more uniform within each quantization group.

The Walsh-Hadamard Transform is a particularly useful orthogonal transformation for this purpose. It can be computed efficiently in O(d log d) time and has the property of spreading energy evenly across dimensions. When combined with random diagonal signs, it creates a randomized rotation that tends to Gaussianize the distribution of weight values within each group.

Cross-layer equalization is another important technique that exploits the mathematical properties of neural networks. Because two adjacent linear layers W1 and W2 can be equivalently represented as (W1 * D) and (D^{-1} * W2) for any invertible diagonal matrix D, we can redistribute the magnitude of weights between layers to make each layer individually easier to quantize.

Channel permutation provides yet another degree of freedom. Because the internal ordering of hidden dimensions is arbitrary as long as all connected layers agree on the ordering, we can reorder channels so that channels with similar magnitudes and quantization sensitivity fall into the same groups. This simple operation can significantly improve group quantization quality at zero computational cost during inference.

In natural language processing, transformer architectures have become dominant. The self-attention mechanism allows these models to capture long-range dependencies in text, while the feed-forward network layers provide additional capacity for processing information. Each transformer block typically contains attention projections (Q, K, V, O) and MLP projections (gate, up, down), all of which are candidates for quantization.

The mathematical foundations of information theory tell us that there are fundamental limits on how well we can approximate a signal with a given number of bits. For a Gaussian distribution quantized to 1 bit with a symmetric threshold at zero, the optimal scale factor is the mean absolute value, and the achievable relative MSE is bounded below by 1 - 2/pi, approximately 36.3 percent. This means that even with perfect preprocessing, we cannot achieve arbitrarily low quantization error at 1 bit.

However, the practical error can often exceed this theoretical minimum substantially when the weight distribution is anisotropic, has outliers, or when group quantization forces unrelated channels to share a single scale factor. This gap between theoretical minimum and practical error is exactly where geometry optimization methods like rotation and permutation can make significant improvements.

Research in this area continues to advance rapidly, with new methods being proposed for both training-aware and post-training quantization. The holy grail is a method that achieves high compression ratios while maintaining model quality, requires no additional training, and adds no runtime overhead. Such a method would make large language models accessible on consumer devices and edge hardware.

Machine learning systems have also found applications in computer vision, robotics, drug discovery, weather prediction, and many other domains. The ability to process and generate text has opened up possibilities for automated code generation, document summarization, translation between languages, and creative writing assistance.

The development of efficient quantization methods is crucial for the democratization of AI technology, enabling researchers and developers with limited computational resources to work with state-of-the-art models. As models continue to grow in size, the importance of compression techniques will only increase."""
