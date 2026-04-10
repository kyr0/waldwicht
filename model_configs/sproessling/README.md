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
  **Sprössling** - MLX Edition - 3.17 GB
</div>

## Overview

**Waldwicht-Sproessling** is a high-quality 5-bit mixed-precision quantization of Gemma 4 E2B (2.3B effective parameters). At **3.17 GB** it matches or exceeds uniform 4-bit (3.22 GB) on all quality metrics while being slightly smaller — discovered through a 50+ configuration ablation study on Apple Silicon.

This is the **balanced variant**: attention and MLP both stay at 5-bit for maximum reasoning fidelity, while only PLE/embeddings are reduced to 3–4 bit. Choose this over Winzling when you need the best possible quality within a ~3 GB budget.

> **Part of the Waldwicht family:**
> [Winzling (2.96 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Winzling) · [Sproessling (3.17 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Sproessling) · [Juengling (3.86 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Juengling)

## Quick Start

For a quick test, you can install the a current version of `mlx-lm` from PyPI and run `mlx_lm.generate` with the model name and a prompt:

```bash
pip install mlx-lm
python -m mlx_lm.generate --model kyr0/Gemma-4-Waldwicht-Sproessling --prompt "Explain quantum tunneling in simple terms."
```

However, we recommend using the Waldwicht Inference Server.


## Waldwicht Inference Server

Together with this model release, we are also launching the Waldwicht Inference Server, a high-performance server optimized for running our quantized models on Apple Silicon. The server is open-source and available on GitHub: [Waldwicht Inference Server](https://github.com/kyr0/waldwicht)

## Quantization Specification

| Component Group | Weight Key Patterns | Bits | Group Size |
|---|---|---|---|
| Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 5 | 64 |
| MLP | `gate_proj`, `up_proj`, `down_proj` | 5 | 64 |
| PLE embeddings | `embed_tokens_per_layer` | 3 | 64 |
| PLE gate/projection | `per_layer_input_gate`, `per_layer_projection` | 4 | 64 |
| Main embed + LM head | `embed_tokens`, `lm_head` | 3 | 64 |
| Norms, scalars | (not quantized) | BF16 | — |

## Key Metrics

| Metric | Value |
|---|---|
| **Model size** | 3.17 GB |
| **Memory footprint** | 2.83 GB peak (MacBook Air M4 24 GB) |
| **Compression from BF16** | 3.0× |
| **Throughput** | 48.6 tok/s avg (MacBook Air M4 24 GB) |
| **Correctness (20-prompt)** | 9.25 / 10.0 |
| **Completion (20-prompt)** | 9.15 / 10.0 |
| **Reasoning hygiene (20-prompt)** | 9.35 / 10.0 |
| **Expanded AQ gate** | ✅ PASS (threshold ≥ 7.0) |

## Benchmark Results

Evaluated on a 20-prompt diverse benchmark covering 6 task categories: function calling (3), multi-step function calling (1), code generation (4), translation (4), multi-step reasoning (4), and creative writing (4). Each response scored on correctness, completion, and reasoning hygiene (0–10 scale).

**Overall: 9.25 / 9.15 / 9.35** (correctness / completion / reasoning hygiene)

## Comparison with Other Variants

| Config | Size | RAM | tok/s | Corr. | Comp. | Reas. | Gate |
|---|---|---|---|---|---|---|---|
| Winzling (5-bit mixed B) | 2.96 GB | 2.63 GB | 51.5 | 8.75 | 8.95 | 9.05 | PASS |
| **Sproessling (this model)** | **3.17 GB** | **2.83 GB** | **48.6** | **9.25** | **9.15** | **9.35** | **PASS** |
| Juengling (uniform 5-bit) | 3.86 GB | 3.52 GB | 47.4 | 9.40 | 9.35 | 9.50 | PASS |

## How This Config Was Discovered

Sproessling applies the **relative weighting transfer** pattern to a 5-bit base (Section 7.9, Variant A). The component sensitivity ordering — attention > MLP > gate > PLE/embed — was discovered through 28 component-level experiments at 4-bit base, then transferred to 5-bit:

- **Attention and MLP** remain at 5-bit (full base precision) — these carry the reasoning and code generation capability
- **PLE embeddings and main embed/LM-head** drop to 3-bit (−2 offset) — these are lookup tables tolerant of compression
- **PLE gate/projection** drops to 4-bit (−1 offset)

The result: 5-bit precision on the core compute path (attention + MLP) delivers quality that matches or exceeds uniform 4-bit across all 6 task categories, while 3-bit PLE/embeddings keep the total size **0.05 GB below uniform 4-bit**.

### Key Advantages over Winzling (2.96 GB)

- Higher overall quality: +0.50 correctness, +0.20 completion, +0.30 reasoning hygiene
- 5-bit MLP (vs 4-bit in Winzling) preserves reasoning and code generation fidelity
- **Comparable throughput** (48.6 vs 51.5 tok/s)

### Research Paper

Full details: [Pushing Below 3 GB: Component-Level Mixed-Precision Quantization for Gemma 4 E2B on Apple Silicon](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Sproessling/blob/main/PAPER_GEMMA4_E2B_MLX.md) (Homberg, April 2026). 50+ configurations, 4 experimental phases, 20-prompt diverse benchmark with LLM-judge scoring.

<div align="center">
  <img src=https://github.com/kyr0/waldwicht/raw/main/waldwicht_artwork.png>
</div>


# Base model

For the base model documentation, see [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) — the smallest dense model in the Gemma 4 family, with 2.3B effective parameters and a 128K token context window.

<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>
