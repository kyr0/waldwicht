#!/usr/bin/env python3
"""Export the three Waldwicht model variants for distribution.

- kyr0/Gemma-4-Waldwicht-Winzling   (2.96 GB) — 5-bit mixed B
- kyr0/Gemma-4-Waldwicht-Sproessling (3.17 GB) — 5-bit mixed A
- kyr0/Gemma-4-Waldwicht-Juengling  (3.86 GB) — uniform 5-bit g64

Usage:
  python export.py [--model NAME] [--output-dir DIR]
"""
import argparse
import json
import shutil
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "calibration-aware-ple"))
from infrastructure import (
    BF16_PATH, HF_HOME, export_mixed_quantized, export_uniform_quantized
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(HF_HOME), "mlx")
MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "model_configs")

# Map full model name → config subfolder
VARIANT_MAP = {
    "Gemma-4-Waldwicht-Winzling": "winzling",
    "Gemma-4-Waldwicht-Sproessling": "sproessling",
    "Gemma-4-Waldwicht-Juengling": "juengling",
}


def load_quant_config(variant_key: str) -> dict:
    """Load waldwicht_quantization.json for a model variant."""
    cfg_path = os.path.join(MODEL_CONFIGS_DIR, variant_key, "waldwicht_quantization.json")
    with open(cfg_path) as f:
        return json.load(f)


def copy_model_configs(variant_key: str, out_path: str):
    """Copy README.md, config.json, and scripts/ from model_configs into the output dir."""
    cfg_dir = os.path.join(MODEL_CONFIGS_DIR, variant_key)
    for fname in ("README.md", "config.json"):
        src = os.path.join(cfg_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, out_path)
    # Copy scripts/ from model_scaffold
    scaffold_dir = os.path.join(os.path.dirname(__file__), "..", "model_scaffold")
    scripts_src = os.path.join(scaffold_dir, "scripts")
    if os.path.isdir(scripts_src):
        scripts_dst = os.path.join(out_path, "scripts")
        if os.path.exists(scripts_dst):
            shutil.rmtree(scripts_dst)
        shutil.copytree(scripts_src, scripts_dst)


def export_model(name: str, output_dir: str):
    variant_key = VARIANT_MAP[name]
    qcfg = load_quant_config(variant_key)

    out_path = os.path.join(output_dir, name)
    print(f"\n{'='*60}")
    print(f"Exporting {name}...")
    print(f"  Output: {out_path}")
    t0 = time.time()

    if qcfg["type"] == "mixed":
        overrides = {
            comp: (spec["bits"], spec["group_size"])
            for comp, spec in qcfg["overrides"].items()
        }
        size = export_mixed_quantized(
            BF16_PATH, out_path,
            default_bits=qcfg["default_bits"],
            default_gs=qcfg["default_group_size"],
            component_overrides=overrides,
        )
    else:
        size = export_uniform_quantized(
            BF16_PATH, out_path,
            bits=qcfg["default_bits"],
            group_size=qcfg["default_group_size"],
        )

    # Overwrite config.json and add README.md from model_configs
    copy_model_configs(variant_key, out_path)

    elapsed = time.time() - t0
    print(f"  Done: {size:.2f} GB in {elapsed:.1f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Export Waldwicht model variants")
    parser.add_argument(
        "--model", default="Gemma-4-Waldwicht-Winzling",
        choices=list(VARIANT_MAP.keys()),
        help="Model variant to export (default: Gemma-4-Waldwicht-Winzling)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    export_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
