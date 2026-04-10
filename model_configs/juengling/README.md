---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model: google/gemma-4-E2B-it
tags:
  - mlx
---

<div align="center">
  <img src=https://github.com/kyr0/waldwicht/raw/main/waldwicht_banner.png>
  <br>
  Jüngling - MLX Edition - 3.86 GB
</div>

## Overview

**Waldwicht-Juengling** is the highest-quality quantization of Gemma 4 E2B (2.3B effective parameters), using **uniform 5-bit g64** precision. At **3.86 GB** it achieves near-BF16 quality across all task categories - the result of a 50+ configuration ablation study on Apple Silicon.

This is the **quality-first variant**: when you have the memory budget, Juengling delivers the best scores across code generation, multi-step reasoning, translation, and creative writing. Choose this when quality matters more than size.

> **Part of the Waldwicht family:**
> [Winzling (2.96 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Winzling) - [Sproessling (3.17 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Sproessling) - [Juengling (3.86 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Juengling)

## Quick Start

For a quick test, you can install the a current version of `mlx-lm` from PyPI and run `mlx_lm.generate` with the model name and a prompt:

```bash
pip install mlx-lm
python -m mlx_lm.generate --model kyr0/Gemma-4-Waldwicht-Juengling --prompt "Explain quantum tunneling in simple terms."
```

However, we recommend using the Waldwicht Inference Server.

## Waldwicht Inference Server

Together with this model release, we are also launching the Waldwicht Inference Server, a high-performance server optimized for running our quantized models on Apple Silicon. The server is open-source and available on GitHub: [Waldwicht Inference Server](https://github.com/kyr0/waldwicht)

## Quantization Specification

| Component Group | Weight Key Patterns | Bits | Group Size |
|---|---|---|---|
| All linear weights | All 2-D weight tensors | 5 | 64 |
| Norms, scalars | (not quantized) | BF16 | - |

Uniform 5-bit affine quantization with group size 64. No transforms, no rotation, no calibration data - standard MLX `mx.quantize`.

## Key Metrics

| Metric | Value |
|---|---|
| **Model size** | 3.86 GB |
| **Memory footprint** | 3.52 GB peak (MacBook Air M4 24 GB) |
| **Compression from BF16** | 2.5x |
| **Throughput** | 47.4 tok/s avg (MacBook Air M4 24 GB) |
| **Correctness (20-prompt)** | 9.40 / 10.0 |
| **Completion (20-prompt)** | 9.35 / 10.0 |
| **Reasoning hygiene (20-prompt)** | 9.50 / 10.0 |
| **Expanded AQ gate** | [OK] PASS (threshold >= 7.0) |

## Benchmark Results by Category

Evaluated on a 20-prompt diverse benchmark covering 6 task categories (correctness / completion / reasoning hygiene):

| Category | Scores |
|---|---|
| Function calling (3 prompts) | 9.7 / 9.7 / 9.7 |
| Multi-step tool calling (1) | 7.0 / 9.0 / 8.0 |
| Code generation (4) | 9.8 / 9.5 / 9.8 |
| Translation (4) | 9.0 / 9.2 / 9.0 |
| Multi-step reasoning (4) | 9.2 / 8.5 / 9.5 |
| Creative writing (4) | 10.0 / 10.0 / 10.0 |

### Highlights

- **Code generation** achieves near-perfect 9.8 correctness across all 4 code prompts
- **Multi-step reasoning** scores 9.2 correctness - up from 7.0 at uniform 4-bit, the largest quality jump between bit-widths
- **Translation** scores 9.0 correctness across all 4 translation prompts
- **Creative writing** scores perfect 10.0/10.0/10.0 across all 4 prompts
- **Reasoning quality jumps sharply at 5-bit.** Multi-step reasoning goes from 7.0 (uniform 4-bit) to 9.2 (uniform 5-bit) correctness

## Comparison with Other Variants

| Config | Size | RAM | tok/s | Corr. | Comp. | Reas. | Gate |
|---|---|---|---|---|---|---|---|
| Winzling (5-bit mixed B) | 2.96 GB | 2.63 GB | 51.5 | 8.75 | 8.95 | 9.05 | PASS |
| Sproessling (5-bit mixed A) | 3.17 GB | 2.83 GB | 48.6 | 9.25 | 9.15 | 9.35 | PASS |
| **Juengling (this model)** | **3.86 GB** | **3.52 GB** | **47.4** | **9.40** | **9.35** | **9.50** | **PASS** |

## Why Uniform 5-bit?

Our ablation study tested 50+ configurations including layer-level mixed-precision, component-level mixed-precision, rotation, and various group sizes. Key findings relevant to this variant:

1. **Uniform beats layer-level mixed-precision for E2B.** Unlike the larger E4B model where three-tier mixed-precision was essential, E2B's smaller size and flatter layer sensitivity profile makes uniform quantization optimal at each bit-depth.

2. **Group size g64 is Pareto-optimal.** E2B's narrow hidden dimension (1536) benefits from finer granularity. g128 causes catastrophic failure at lower bit-widths; g32 passes but wastes +0.64 GB. g64 maximizes quality per GB.

3. **5-bit is the quality ceiling.** Going from 4-bit to 5-bit delivers the largest quality jumps in reasoning (+2.2 correctness), translation (+0.2), and code gen (+0.6). Going to 6-bit adds minimal improvement at +0.64 GB cost.

4. **No transforms needed.** Rotation was tested and provides no benefit for E2B uniform quantization. Standard `mx.quantize` is sufficient.

### When to Choose Juengling vs Smaller Variants

| Use Case | Recommended |
|---|---|
| Maximum quality, memory available | **Juengling (3.86 GB)** |
| Best quality within ~3 GB budget | Sproessling (3.17 GB) |
| Maximum compression, diverse tasks | Winzling (2.96 GB) |

### Research Paper

Full details: [Pushing Below 3 GB: Component-Level Mixed-Precision Quantization for Gemma 4 E2B on Apple Silicon](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Juengling/blob/main/PAPER_GEMMA4_E2B_MLX.md) (Homberg, April 2026). 50+ configurations, 4 experimental phases, 20-prompt diverse benchmark with LLM-judge scoring.

<div align="center">
  <img src=https://github.com/kyr0/waldwicht/raw/main/waldwicht_artwork.png>
</div>

# Base model

For the base model documentation, see [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) - the smallest dense model in the Gemma 4 family, with 2.3B effective parameters and a 128K token context window.

<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>
