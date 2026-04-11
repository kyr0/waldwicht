#!/usr/bin/env python3
"""Verify embedding conversion dependencies in the repo virtual environment."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


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
    locations = module_locations(module)

    for location in locations:
        try:
            location.relative_to(expected_root)
            return
        except ValueError:
            continue

    raise RuntimeError(
        f"{module.__name__} is installed from {locations}, expected under {expected_root}"
    )


def main() -> int:
    try:
        import requests  # noqa: F401
        from PIL import Image  # noqa: F401

        import mlx  # noqa: F401
        import mlx_embeddings
        import mlx_lm
        import mlx_vlm  # noqa: F401

        require_local_module(mlx_lm, ROOT / "mlx-lm")
        require_local_module(mlx_embeddings, ROOT / "mlx-embeddings")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())