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
  Winzling - MLX Edition - 2.96 GB
</div>

## Overview

**Waldwicht-Winzling** is the smallest diverse-task-competent quantization of Gemma 4 E2B (2.3B effective parameters). At **2.96 GB** it is **8% smaller than uniform 4-bit** (3.22 GB) while matching its quality on a rigorous 20-prompt benchmark - the result of a 50+ configuration ablation study on Apple Silicon.

This is the **recommended general-use variant** for maximum compression. It uses **5-bit mixed-precision with component-level offsets**: attention stays at 5-bit (the most sensitive component), while MLP drops to 4-bit and PLE/embeddings drop to 3-bit.

> **Part of the Waldwicht family:**
> [Winzling (2.96 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Winzling) - [Sproessling (3.17 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Sproessling) - [Juengling (3.86 GB)](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Juengling)

## Quick Start

For a quick test, you can install the a current version of `mlx-lm` from PyPI and run `mlx_lm.generate` with the model name and a prompt:

```bash
pip install mlx-lm
python -m mlx_lm.generate --model kyr0/Gemma-4-Waldwicht-Winzling --prompt "Explain quantum tunneling in simple terms."
```

However, we recommend using the Waldwicht Inference Server.

## Waldwicht Inference Server

Together with this model release, we are also launching the Waldwicht Inference Server, a high-performance server optimized for running our quantized models on Apple Silicon. The server is open-source and available on GitHub: [Waldwicht Inference Server](https://github.com/kyr0/waldwicht)

## Quantization Specification

| Component Group | Weight Key Patterns | Bits | Group Size |
|---|---|---|---|
| Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 5 | 64 |
| MLP | `gate_proj`, `up_proj`, `down_proj` | 4 | 64 |
| PLE embeddings | `embed_tokens_per_layer` | 3 | 64 |
| PLE gate/projection | `per_layer_input_gate`, `per_layer_projection` | 4 | 64 |
| Main embed + LM head | `embed_tokens`, `lm_head` | 3 | 64 |
| Norms, scalars | (not quantized) | BF16 | - |

## Key Metrics

| Metric | Value |
|---|---|
| **Model size** | 2.96 GB |
| **Memory footprint** | 2.63 GB peak (MacBook Air M4 24 GB) |
| **Compression from BF16** | 3.2x |
| **Throughput** | 51.5 tok/s avg (MacBook Air M4 24 GB) |
| **Correctness (20-prompt)** | 8.75 / 10.0 |
| **Completion (20-prompt)** | 8.95 / 10.0 |
| **Reasoning hygiene (20-prompt)** | 9.05 / 10.0 |
| **Expanded AQ gate** | [OK] PASS (threshold >= 7.0) |

## Benchmark Results by Category

Evaluated on a 20-prompt diverse benchmark covering 6 task categories (correctness / completion / reasoning hygiene):

| Category | Scores |
|---|---|
| Function calling (3 prompts) | 9.7 / 9.7 / 9.7 |
| Multi-step tool calling (1) | 6.0 / 8.0 / 7.0 |
| Code generation (4) | 9.5 / 8.2 / 9.2 |
| Translation (4) | 8.2 / 9.5 / 8.5 |
| Multi-step reasoning (4) | 7.8 / 7.8 / 8.8 |
| Creative writing (4) | 9.5 / 10.0 / 9.8 |

## Comparison with Other Variants

| Config | Size | RAM | tok/s | Corr. | Comp. | Reas. | Gate |
|---|---|---|---|---|---|---|---|
| **Winzling (this model)** | **2.96 GB** | **2.63 GB** | **51.5** | **8.75** | **8.95** | **9.05** | **PASS** |
| Sproessling (5-bit mixed A) | 3.17 GB | 2.83 GB | 48.6 | 9.25 | 9.15 | 9.35 | PASS |
| Juengling (uniform 5-bit) | 3.86 GB | 3.52 GB | 47.4 | 9.40 | 9.35 | 9.50 | PASS |

## How This Config Was Discovered

This model is the result of a **relative weighting transfer experiment** (Section 7.9 of the research paper). The component sensitivity ordering - attention > MLP > gate > PLE/embed - was discovered at 4-bit base through systematic ablation of 28 component-level configurations. We found that:

1. **Attention is the binding constraint** - it must stay at the highest precision in any valid composition
2. **MLP tolerates 1-bit demotion** from the base precision
3. **PLE embeddings and main embeddings tolerate 2-bit demotion** from base
4. **PLE gate/projection tolerates 1-bit demotion** from base

Transferring these relative offsets from a 4-bit base to a 5-bit base produced this 2.96 GB config - **the smallest model to pass the full expanded benchmark**, displacing uniform 4-bit (3.22 GB) as the practical minimum for diverse-task competence.

### Known Limitations

- Multi-step function calling is the weakest category (6.0 correctness, 8.0 completion, 7.0 reasoning hygiene)
- Multi-step reasoning (7.8 correctness) and translation (8.2 correctness) score lower than other categories

### Research Paper

Full details: [Pushing Below 3 GB: Component-Level Mixed-Precision Quantization for Gemma 4 E2B on Apple Silicon](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Winzling/blob/main/PAPER_GEMMA4_E2B_MLX.md) (Homberg, April 2026). 50+ configurations, 4 experimental phases, 20-prompt diverse benchmark with LLM-judge scoring.

<div align="center">
  <img src=https://github.com/kyr0/waldwicht/raw/main/waldwicht_artwork.png>
</div>

# Base model

For the base model documentation, see [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) - the smallest dense model in the Gemma 4 family, with 2.3B effective parameters and a 128K token context window.

<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>
