---
library_name: mlx
base_model: microsoft/harrier-oss-v1-270m
tags:
- mlx
- embeddings
- mteb
- sentence-transformers
- transformers
language:
- multilingual
- af
- am
- ar
- as
- az
- be
- bg
- bn
- br
- bs
- ca
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- he
- hi
- hr
- hu
- hy
- id
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lo
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- my
- ne
- nl
- 'no'
- om
- or
- pa
- pl
- ps
- pt
- ro
- ru
- sa
- sd
- si
- sk
- sl
- so
- sq
- sr
- su
- sv
- sw
- ta
- te
- th
- tl
- tr
- ug
- uk
- ur
- uz
- vi
- xh
- yi
- zh
license: mit
---

<div align="center">
  <img src=https://github.com/kyr0/waldwicht/raw/main/waldwicht_banner.png>
  <br>
  Wurzler - Embedding Model - MLX Edition - 318.4 MB
</div>

## Overview

**Harrier-Waldwicht-Wurzler-MLX** is the Waldwicht MLX conversion of [`microsoft/harrier-oss-v1-270m`](https://huggingface.co/microsoft/harrier-oss-v1-270m), optimized for Apple Silicon embedding workloads.

This export keeps the original `sentence-transformers` structure intact:

- transformer backbone
- pooling head
- normalize head

The model was converted with the Waldwicht MLX toolchain and quantized for smaller local deployment while preserving the original embedding behavior.

## What Was Done Here

This model directory was produced from the base Hugging Face model with the root Makefile in the Waldwicht repository.

The conversion pipeline does the following:

1. Converts `microsoft/harrier-oss-v1-270m` into MLX format.
2. Quantizes the MLX weights with uniform affine quantization.
3. Writes the converted model files into the target output directory.
4. Copies the scaffold files from `embeddings/model_scaffold/` into the output so the model folder is self-contained.
5. Verifies the result with `make test-embed`.

Default conversion profile used here:

| Setting | Value |
|---|---|
| Quantization | enabled |
| Quantization mode | affine |
| Bits | 8 |
| Group size | 64 |
| Base dtype before quantization | BF16 |

## Quick Start

You currently need to use the [Waldwicht repository's `mlx-embeddings`](https://github.com/kyr0/waldwicht) fork, as I had to implement code changes in the model loading and generation.

You can also validate a local export from the Waldwicht repository with:

```bash
make test-embed \
  EMBEDDING_MLX_PATH=/path/to/Harrier-Waldwicht-Wurzler-MLX \
  EMBED_TEXT="Waldwicht verifies embedding conversion."
```

## MTEB Sanity Check

The Waldwicht Makefile can run a packaged-runtime MTEB evaluation directly against this MLX export:

```bash
make embed-mteb \
  EMBED_MTEB_MODEL=/path/to/Harrier-Waldwicht-Wurzler-MLX \
  EMBED_MTEB_TASKS="STS12 STS13 STS14" \
  EMBED_MTEB_OVERWRITE=1
```

This writes benchmark artifacts to `mteb-results/benchmark_results.json` and `mteb-results/benchmark_results.md`.

Current quick-check result for the quantized 8-bit affine g64 export on the English MTEB v2 STS subset:

| Task | Score |
|---|---:|
| STS12 | 0.605697 |
| STS13 | 0.623238 |
| STS14 | 0.600989 |
| Mean (Task) | 0.609975 |

These numbers are intended as a fast local sanity check for the quantized export, not as a full leaderboard submission.

## Model Details

| Item | Value |
|---|---|
| Base model | `microsoft/harrier-oss-v1-270m` |
| MLX model type | `gemma3_text` |
| Hidden size | 640 |
| Layers | 18 |
| Max position embeddings | 32768 |
| Sentence-transformers modules | Transformer + Pooling + Normalize |
| Quantization | 8-bit affine, group size 64 |
| On-disk size | about 304 MB + tokenizer.json 33.4 MB |

## Waldwicht Workflow

Inside the Waldwicht repository, the intended workflow is:

```bash
make convert-embedding \
  EMBEDDING_MODEL=microsoft/harrier-oss-v1-270m \
  EMBEDDING_MLX_PATH=/path/to/embeddings/Harrier-Waldwicht-Wurzler-MLX
```

The Makefile target installs or reuses the local MLX stack, converts the model, then copies this scaffold into the output directory.

## Included Scaffold Files

This exported directory also includes:

- `README.md` with conversion details and usage notes
- `Makefile` and `scripts/` for Hugging Face upload workflow
- converted MLX weights and tokenizer/config files

## Waldwicht Inference Server

The Waldwicht repository also includes an Apple Silicon inference stack that can serve embedding models through an OpenAI-compatible API via `omlx`.

Repository: [kyr0/waldwicht](https://github.com/kyr0/waldwicht)

### Base Model

For the original base model card and benchmark claims, see [`microsoft/harrier-oss-v1-270m`](https://huggingface.co/microsoft/harrier-oss-v1-270m).

<div align="center">
  <img src=https://github.com/kyr0/waldwicht/raw/main/waldwicht_artwork.png>
</div>
