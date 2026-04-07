"""ROTQ — Foldable geometry optimization for grouped sign quantization.

Post-training, training-free pipeline that applies function-preserving
transforms (equalization → permutation → structured rotation) to model
weights offline, then sign-quantizes and exports plain 1-bit sign+scale
weights with NO runtime transforms.

The output is indistinguishable from plain Q1 at inference — standard
QuantizedLinear, no RotatedLinear, no .signs keys.
"""
