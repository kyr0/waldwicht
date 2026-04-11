#!/usr/bin/env python3
"""Run MTEB with an MLX embedding model and write summary artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mteb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from mlx_embeddings import load
from mlx_embeddings.utils import prepare_inputs
from mteb.cache import ResultCache
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ScoringFunction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTEB with an MLX embedding model and write summary files."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local model path or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--benchmark",
        default="MTEB(eng, v2)",
        help="Benchmark name, for example 'MTEB(eng, v2)'.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional explicit task list to run instead of the named benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated benchmark artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for raw MTEB cache/results. Defaults to <output-dir>/cache.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Path to summary JSON output. Defaults to <output-dir>/benchmark_results.json.",
    )
    parser.add_argument(
        "--results-md",
        default=None,
        help="Path to markdown output. Defaults to <output-dir>/benchmark_results.md.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional path to mteb_v2_eval_prompts.json.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size passed through to MTEB encoding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any previous cached task results before running.",
    )
    return parser.parse_args()


def resolve_local_model_path(model_name: str) -> Path | None:
    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return None


def resolve_prompt_file(model_name: str, prompt_file: str | None) -> Path | None:
    if prompt_file:
        candidate = Path(prompt_file).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Prompt file not found: {candidate}")
        return candidate

    local_model_path = resolve_local_model_path(model_name)
    if local_model_path is not None:
        candidate = local_model_path / "mteb_v2_eval_prompts.json"
        if candidate.is_file():
            return candidate

    if "/" in model_name:
        try:
            return Path(
                hf_hub_download(model_name, filename="mteb_v2_eval_prompts.json")
            ).resolve()
        except Exception:
            return None

    return None


def sanitize_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def model_reference(model_name: str) -> str | None:
    if resolve_local_model_path(model_name) is None and "/" in model_name:
        return f"https://huggingface.co/{model_name}"
    return None


def display_model_name(model_name: str) -> str:
    local_model_path = resolve_local_model_path(model_name)
    if local_model_path is not None:
        return local_model_path.name
    return model_name.split("/")[-1]


def format_task_type(task_type: Any) -> str:
    value = getattr(task_type, "value", task_type)
    text = str(value or "Unknown").replace("_", " ")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def build_local_summary_table(
    benchmark,
    benchmark_results: mteb.BenchmarkResults,
) -> pd.DataFrame:
    data = benchmark_results.to_dataframe(format="long")
    if data.empty:
        return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

    task_type_by_name = {
        task.metadata.name: format_task_type(task.metadata.type)
        for task in benchmark.tasks
    }

    data = data.copy()
    data["task_type"] = data["task_name"].map(task_type_by_name).fillna("Unknown")

    per_task = data.pivot(index="model_name", columns="task_name", values="score")
    type_means = (
        data.groupby(["model_name", "task_type"], dropna=False)["score"]
        .mean()
        .unstack("task_type")
    )

    overall_mean = per_task.mean(skipna=False, axis=1)
    typed_mean = (
        type_means.mean(skipna=False, axis=1) if not type_means.empty else overall_mean
    )

    summary = type_means.copy()
    summary.insert(0, "Mean (Task)", overall_mean)
    summary.insert(1, "Mean (TaskType)", typed_mean)
    summary = summary.sort_values("Mean (Task)", ascending=False)
    summary.insert(
        0,
        "Rank (Mean Task)",
        summary["Mean (Task)"].rank(ascending=False, method="dense").astype(int),
    )
    summary = summary.reset_index()
    summary["model_name"] = summary["model_name"].map(display_model_name)
    return summary.rename(columns={"model_name": "Model"})


def build_summary_table(
    benchmark,
    benchmark_results: mteb.BenchmarkResults,
) -> pd.DataFrame:
    try:
        return benchmark_results.get_benchmark_result()
    except KeyError as exc:
        print(
            f"=> Falling back to local summary table for unregistered model metadata: {exc}",
            file=sys.stderr,
        )
        return build_local_summary_table(benchmark, benchmark_results)


class MLXMTEBEncoder(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        *,
        max_length: int = 512,
        prompt_file: Path | None = None,
        **_: Any,
    ):
        self.max_length = max_length
        self.model_name = model_name
        self.local_model_path = resolve_local_model_path(model_name)
        self.model_id = (
            str(self.local_model_path) if self.local_model_path is not None else model_name
        )
        self.model, self.processor = load(model_name)
        self.call_parameters = inspect.signature(self.model.__call__).parameters
        self.task_prompts = self._load_task_prompts(prompt_file)
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": self.model_id,
                "revision": revision or "no_revision_available",
                "framework": ["Sentence Transformers"],
                "reference": model_reference(model_name),
                "max_tokens": max_length,
                "similarity_fn_name": ScoringFunction.COSINE,
                "open_weights": True,
                "model_type": ["dense"],
                "use_instructions": bool(self.task_prompts),
            }
        )

    def _load_task_prompts(self, prompt_file: Path | None) -> dict[str, str]:
        if prompt_file is None:
            return {}
        data = json.loads(prompt_file.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Prompt file must contain a JSON object: {prompt_file}")
        return {str(key): str(value) for key, value in data.items()}

    def _extract_texts(self, batch: Any) -> list[str]:
        value = batch
        if isinstance(batch, dict):
            if "text" in batch:
                value = batch["text"]
            elif len(batch) == 1:
                value = next(iter(batch.values()))
            else:
                raise ValueError(f"Unsupported MTEB batch keys: {sorted(batch.keys())}")

        if isinstance(value, str):
            return [value]
        if hasattr(value, "tolist") and not isinstance(value, list):
            value = value.tolist()
        return [str(item) for item in value]

    def _resolve_instruction(self, task_name: str, prompt_type: Any) -> str | None:
        prompt_value = getattr(prompt_type, "value", None)
        candidates: list[str] = []

        if prompt_value:
            candidates.append(f"{task_name}-{prompt_value}")
            if prompt_value == "query":
                candidates.append(task_name)
        else:
            candidates.append(task_name)

        for candidate in candidates:
            instruction = self.task_prompts.get(candidate)
            if instruction:
                return instruction
        return None

    def _apply_instruction(
        self, texts: list[str], task_name: str, prompt_type: Any
    ) -> list[str]:
        instruction = self._resolve_instruction(task_name, prompt_type)
        if not instruction:
            return texts

        prompted: list[str] = []
        for text in texts:
            if "{text}" in instruction:
                prompted.append(instruction.format(text=text))
            elif instruction.startswith("Instruct:") or "Query:" in instruction:
                separator = "" if instruction.endswith((" ", "\n", "\t")) else " "
                prompted.append(f"{instruction}{separator}{text}")
            else:
                prompted.append(f"Instruct: {instruction}\nQuery: {text}")
        return prompted

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = prepare_inputs(
            self.processor,
            None,
            texts,
            self.max_length,
            True,
            True,
        )

        if isinstance(inputs, mx.array):
            outputs = self.model(inputs)
        else:
            model_inputs = dict(inputs)
            if (
                "input_ids" in model_inputs
                and "input_ids" not in self.call_parameters
                and "inputs" in self.call_parameters
            ):
                model_inputs["inputs"] = model_inputs.pop("input_ids")
            outputs = self.model(**model_inputs)

        embedding_array = getattr(outputs, "text_embeds", None)
        if embedding_array is None:
            embedding_array = getattr(outputs, "pooler_output", None)
        if embedding_array is None:
            raise RuntimeError("Model output did not contain text_embeds or pooler_output")

        mx.eval(embedding_array)
        embeddings = np.array(embedding_array.tolist(), dtype=np.float32)
        if (
            self.mteb_model_meta.embed_dim is None
            and embeddings.ndim == 2
            and embeddings.shape[1] > 0
        ):
            self.mteb_model_meta = self.mteb_model_meta.model_copy(
                update={"embed_dim": int(embeddings.shape[1])}
            )
        return embeddings

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        del hf_split, hf_subset, kwargs

        encoded_batches: list[np.ndarray] = []
        for batch in inputs:
            texts = self._extract_texts(batch)
            prompted_texts = self._apply_instruction(
                texts, task_metadata.name, prompt_type
            )
            encoded_batches.append(self._embed_texts(prompted_texts))

        if not encoded_batches:
            dim = int(self.mteb_model_meta.embed_dim or 0)
            return np.empty((0, dim), dtype=np.float32)
        return np.concatenate(encoded_batches, axis=0)


def build_benchmark(args: argparse.Namespace):
    if args.tasks:
        return mteb.Benchmark(
            name=args.benchmark,
            tasks=mteb.get_tasks(tasks=args.tasks),
        )
    return mteb.get_benchmark(args.benchmark)


def render_markdown(
    benchmark_name: str,
    model_name: str,
    prompt_file: Path | None,
    output_dir: Path,
    cache_dir: Path,
    summary_df: pd.DataFrame,
    per_task_df: pd.DataFrame,
) -> str:
    lines = [
        "# MTEB Results",
        "",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Benchmark: {benchmark_name}",
        f"- Model: {model_name}",
        f"- Prompt file: {prompt_file if prompt_file is not None else 'none'}",
        f"- Output directory: {output_dir}",
        f"- Cache directory: {cache_dir}",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Per Task",
        "",
        per_task_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output_dir / "cache"
    )
    results_json = (
        Path(args.results_json).expanduser().resolve()
        if args.results_json
        else output_dir / "benchmark_results.json"
    )
    results_md = (
        Path(args.results_md).expanduser().resolve()
        if args.results_md
        else output_dir / "benchmark_results.md"
    )

    if args.overwrite and cache_dir.exists():
        shutil.rmtree(cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_md.parent.mkdir(parents=True, exist_ok=True)

    prompt_file = resolve_prompt_file(args.model, args.prompt_file)
    benchmark = build_benchmark(args)
    encoder = MLXMTEBEncoder(
        args.model,
        max_length=args.max_length,
        prompt_file=prompt_file,
    )

    print(
        f"=> Running {benchmark.name} with {len(benchmark.tasks)} task(s) using {encoder.model_id} ...",
        file=sys.stderr,
    )
    model_result = mteb.evaluate(
        encoder,
        tasks=benchmark.tasks,
        cache=ResultCache(cache_dir),
        encode_kwargs={"batch_size": args.batch_size},
        show_progress_bar=True,
    )

    benchmark_results = mteb.BenchmarkResults(
        model_results=[model_result],
        benchmark=benchmark,
    )
    summary_df = build_summary_table(benchmark, benchmark_results)
    per_task_df = benchmark._create_per_task_table(benchmark_results)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": benchmark.name,
        "tasks": [task.metadata.name for task in benchmark.tasks],
        "model": encoder.mteb_model_meta.to_dict(),
        "prompt_file": str(prompt_file) if prompt_file is not None else None,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "summary": sanitize_records(summary_df),
        "per_task": sanitize_records(per_task_df),
    }
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    results_md.write_text(
        render_markdown(
            benchmark.name,
            encoder.model_id,
            prompt_file,
            output_dir,
            cache_dir,
            summary_df,
            per_task_df,
        )
    )

    print(
        json.dumps(
            {
                "benchmark": benchmark.name,
                "task_count": len(benchmark.tasks),
                "results_json": str(results_json),
                "results_md": str(results_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())