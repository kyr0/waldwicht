#!/usr/bin/env python3
"""Run a command with the packaged oMLX Python runtime."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP = ROOT / "omlx" / "packaging" / "dist" / "oMLX.app"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Python command inside the packaged oMLX runtime."
    )
    parser.add_argument(
        "--app",
        default=str(DEFAULT_APP),
        help="Path to the packaged oMLX.app bundle.",
    )
    parser.add_argument(
        "python_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the packaged Python interpreter.",
    )
    return parser.parse_args()


def resolve_runtime(app_bundle: Path) -> tuple[Path, Path, Path, Path, Path]:
    contents_dir = app_bundle / "Contents"
    layers_dir = contents_dir / "Frameworks"
    if not layers_dir.is_dir():
        layers_dir = contents_dir / "Python"

    python_home = layers_dir / "cpython-3.12"
    python_exe = python_home / "bin" / "python3"
    app_site = layers_dir / "app-omlx-app" / "lib" / "python3.12" / "site-packages"
    framework_site = (
        layers_dir / "framework-mlx-framework" / "lib" / "python3.12" / "site-packages"
    )

    required_paths = {
        "app bundle": app_bundle,
        "contents directory": contents_dir,
        "layer directory": layers_dir,
        "python home": python_home,
        "python executable": python_exe,
        "app site-packages": app_site,
        "framework site-packages": framework_site,
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    return contents_dir, python_home, python_exe, app_site, framework_site


def main() -> int:
    args = parse_args()
    python_args = list(args.python_args)
    if python_args and python_args[0] == "--":
        python_args = python_args[1:]

    if not python_args:
        print("No packaged Python command provided.", file=sys.stderr)
        return 2

    app_bundle = Path(args.app).expanduser().resolve()
    contents_dir, python_home, python_exe, app_site, framework_site = resolve_runtime(
        app_bundle
    )

    env = os.environ.copy()
    pythonpath_entries = [
        str(contents_dir / "Resources"),
        str(app_site),
        str(framework_site),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])

    env["PYTHONHOME"] = str(python_home)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run([str(python_exe), *python_args], env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())