#!/usr/bin/env python3
"""Smoke test a text embedding model by embedding one input string."""

import argparse
import inspect
import json
import math
import sys

import mlx.core as mx
from mlx_embeddings import load
from mlx_embeddings.utils import prepare_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a text with an MLX embedding model and print a JSON summary."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local model path or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to embed.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--preview-dims",
        type=int,
        default=8,
        help="How many embedding dimensions to include in the preview output.",
    )
    return parser.parse_args()


def count_tokens(processor, text: str, max_length: int) -> int | None:
    tokenizer = getattr(processor, "_tokenizer", processor)

    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            return len(tokens)
        if hasattr(tokens, "ids"):
            return len(tokens.ids)
        if hasattr(tokens, "shape"):
            return int(tokens.shape[-1]) if tokens.ndim > 0 else 1

    if callable(tokenizer):
        encoded = tokenizer(
            [text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask.sum())

    return None


def main() -> int:
    args = parse_args()

    print(f"=> Loading embedding model from {args.model} ...", file=sys.stderr)
    model, processor = load(args.model)

    print("=> Generating embedding ...", file=sys.stderr)
    inputs = prepare_inputs(
        processor,
        None,
        [args.text],
        args.max_length,
        True,
        True,
    )

    if isinstance(inputs, mx.array):
        outputs = model(inputs)
    else:
        model_inputs = dict(inputs)
        call_parameters = inspect.signature(model.__call__).parameters
        if "input_ids" in model_inputs and "input_ids" not in call_parameters and "inputs" in call_parameters:
            model_inputs["inputs"] = model_inputs.pop("input_ids")
        outputs = model(**model_inputs)

    embedding_array = getattr(outputs, "text_embeds", None)
    if embedding_array is None:
        embedding_array = getattr(outputs, "pooler_output", None)
    if embedding_array is None:
        raise RuntimeError("Model output did not contain text_embeds or pooler_output")

    mx.eval(embedding_array)

    vector = embedding_array[0] if embedding_array.ndim > 1 else embedding_array
    vector_list = vector.tolist()
    preview_dims = max(args.preview_dims, 0)
    token_count = count_tokens(processor, args.text, args.max_length)
    l2_norm = math.sqrt(sum(value * value for value in vector_list))

    result = {
        "model": args.model,
        "text": args.text,
        "dimensions": len(vector_list),
        "token_count": token_count,
        "l2_norm": l2_norm,
        "embedding_preview": vector_list[:preview_dims],
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())