#!/usr/bin/env python3
"""End-to-end smoke test for oMLX /v1/embeddings with a local or HF-backed model."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch oMLX and smoke-test the /v1/embeddings endpoint."
    )
    parser.add_argument(
        "--model-source",
        required=True,
        help="Local model path or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to embed through the HTTP endpoint.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the temporary oMLX server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8436,
        help="Port to bind the temporary oMLX server to.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for server readiness and HTTP response.",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["trace", "debug", "info", "warning", "error"],
        help="oMLX server log level for the smoke test.",
    )
    return parser.parse_args()


def module_locations(module) -> list[Path]:
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(module_file).resolve())
    for module_path in getattr(module, "__path__", []):
        if module_path:
            locations.append(Path(module_path).resolve())
    return locations


def require_local_module(module, expected_root: Path) -> None:
    expected_root = expected_root.resolve()
    for location in module_locations(module):
        try:
            location.relative_to(expected_root)
            return
        except ValueError:
            continue
    raise RuntimeError(
        f"{module.__name__} is not loaded from {expected_root}: {module_locations(module)}"
    )


def resolve_model_source(model_source: str) -> tuple[Path, str]:
    source_path = Path(model_source).expanduser()
    if source_path.exists():
        return source_path.resolve(), source_path.resolve().name

    snapshot_path = Path(snapshot_download(repo_id=model_source))
    return snapshot_path, model_source.rsplit("/", 1)[-1]


def wait_for_server(base_url: str, model_id: str, timeout: float, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.time() + timeout
    last_error: str | None = None

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"oMLX exited early with code {proc.returncode}.\n\n{tail_log(log_path)}"
            )

        try:
            health = requests.get(f"{base_url}/health", timeout=2)
            if health.ok:
                models = requests.get(f"{base_url}/v1/models", timeout=2)
                if models.ok:
                    data = models.json().get("data", [])
                    if any(entry.get("id") == model_id for entry in data):
                        return
                last_error = f"Model '{model_id}' not listed yet"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(1)

    raise RuntimeError(
        f"Timed out waiting for oMLX server readiness ({last_error}).\n\n{tail_log(log_path)}"
    )


def tail_log(log_path: Path, max_lines: int = 80) -> str:
    if not log_path.exists():
        return "<no server log captured>"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    args = parse_args()

    import mlx_embeddings
    import omlx

    require_local_module(omlx, ROOT / "omlx")
    require_local_module(mlx_embeddings, ROOT / "mlx-embeddings")

    model_path, model_id = resolve_model_source(args.model_source)
    if not (model_path / "config.json").exists():
        raise RuntimeError(f"Model source does not contain config.json: {model_path}")

    work_dir = Path(tempfile.mkdtemp(prefix="omlx-embed-smoke-"))
    base_path = work_dir / "base"
    model_root = work_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    staged_model_path = model_root / model_id
    staged_model_path.symlink_to(model_path, target_is_directory=True)
    log_path = work_dir / "omlx.log"
    proc: subprocess.Popen | None = None
    log_file = None

    cmd = [
        sys.executable,
        "-m",
        "omlx.cli",
        "serve",
        "--model-dir",
        str(model_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--base-path",
        str(base_path),
        "--log-level",
        args.log_level,
        "--no-cache",
    ]

    print(f"=> Using local mlx-embeddings from {module_locations(mlx_embeddings)}", file=sys.stderr)
    print(f"=> Starting oMLX for model {model_id} from {model_path} ...", file=sys.stderr)

    try:
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        base_url = f"http://{args.host}:{args.port}"
        wait_for_server(base_url, model_id, args.timeout, proc, log_path)

        response = requests.post(
            f"{base_url}/v1/embeddings",
            json={
                "model": model_id,
                "input": args.text,
                "encoding_format": "float",
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if not payload.get("data"):
            raise RuntimeError(f"oMLX returned no embeddings: {json.dumps(payload, indent=2)}")

        embedding = payload["data"][0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"oMLX returned invalid embedding payload: {json.dumps(payload, indent=2)}")

        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    finally:
        if proc is not None:
            terminate_process(proc)
        if log_file is not None:
            log_file.close()
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())