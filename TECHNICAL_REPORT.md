<div align="center">
  <img src="./waldwicht_banner.png">
</div>

# Waldwicht Quantization and Inference for Apple Silicon / MLX

## Pushing Google Gemma 4 below 3 GB VRAM consumption and above 50 tok/s on consumer Apple devices using Mixed-Precision Quantization for Gemma 4 E2B, preserving output quality.

_Technical Report_

**Author:** Aron Homberg

*Tooling acknowledgement/Transparency: Experiments designed and analyzed with the assistance of Anthropic Claude Opus 4.6 High.*

**Date:** April 2026

**Abstract.** We quantize Gemma 4 E2B (2.3B effective parameters) for Apple Silicon inference via MLX, testing 22 layer-level and 28 component-level configurations across four phases, plus expanded-benchmark evaluations of selected finalists. Layer-level mixed-precision — which helped the larger E4B — hurts E2B; uniform 3-bit g64 wins on perplexity among all 22 layer-level configs. Shifting to component-level allocation (different bits per weight type, not per layer), we find that attention had to remain at 4-bit in all passing compositions while MLP tolerates 3-bit and embeddings tolerate 2-bit, yielding a 2.32 GB config that passes an 8-prompt math gate. However, a broader 20-prompt benchmark covering code, translation, reasoning, and creative writing overturns this result — the 2.32 GB config fails broadly (avg 4.5/5.4/5.0). Transferring the component sensitivity ordering to a 5-bit base produces a **2.96 GB config** (3.2× text-weight compression, from ~9.6 GB text-only BF16) that passes the diverse benchmark at 8.8/9.0/9.1, making it the smallest general-use config tested. The best-performing configurations use no transforms, no rotation, and no calibration data — only standard MLX affine quantization.

**Component-level quantization as a new method.** Unlike uniform or layer-level mixed-precision quantization — both of which existing inference engines support natively — component-level allocation constitutes a distinct quantization paradigm that requires non-trivial inference engine adaptations. Each architectural weight group (attention projections, MLP, embeddings, gating) is quantized at an independently optimized bit-width, meaning a single forward pass touches weights at 3, 4, and 5-bit precision simultaneously. We identify and implement three required changes to the MLX inference stack: (1) per-component config resolution with leaf-name matching, (2) quantized KV-cache reference propagation through Gemma 4's KV-sharing mechanism, and (3) shape-aware attention mask handling for quantized key/value tuples. We release these as a patch to mlx-lm (`patches/mlx-lm-waldwicht.patch`). The resulting method yields both size and throughput gains over uniform quantization at equivalent quality:

| Method | Configuration | Size | tok/s | Quality (C/Cm/R) | Expanded Quality Gate |
|---|---|---|---|---|---|
| Uniform | 5-bit g64 | 3.86 GB | 44.1 | 9.4 / 9.3 / 9.5 | PASS |
| **Component-level** | **attn=5, mlp=4, ple=3, gate=4, embed=3** | **2.96 GB** | **46.6** | **8.8 / 9.0 / 9.1** | **PASS** |

*Component-level allocation achieves 23% smaller size and 6% higher throughput than the nearest uniform config at comparable quality (expanded 20-prompt benchmark, same measurement setup). With KV-cache quantization enabled in the Waldwicht Inference Server, the gap widens to 9% faster throughput and 25% less peak Metal memory (51.5 tok/s at 2.63 GB vs 47.4 tok/s at 3.52 GB).*

**Model releases.** We release the three top-tier models from this ablation study as ready-to-run MLX exports:

| Model | Size | tok/s | Peak RAM | Configuration |
|---|---|---|---|---|
| **Waldwicht-Winzling** | **2.96 GB** | 51.5 | 2.63 GB | attn=5, mlp=4, ple=3, gate=4, embed=3 (recommended) |
| **Waldwicht-Sproessling** | **3.17 GB** | 48.6 | 2.83 GB | attn=5, mlp=5, ple=3, gate=4, embed=3 |
| **Waldwicht-Juengling** | **3.86 GB** | 47.4 | 3.52 GB | uniform 5-bit g64 (near-BF16 quality) |

The above model configurations have been tested while the notebook **was already under 50% CPU load** and **50% memory pressure** _yellow marking_ on the same MacBook Air M4 24 GB machine the ablation tests have been run. This is to ensure that Waldwicht models prove that intelligent models can now run _in background_ on true consumer devices _at speed_.

*Throughput and peak Metal memory measured on MacBook Air M4 24 GB, 256-token generation, 3-run average, greedy decoding.*

**Inference Server release.** Alongside the Waldwicht model family, we release the Waldwicht Inference Server — a high-performance Python API server for Apple Silicon with custom memory management, multi-worker support, custom MLX kernels, TurboQuant, and speculative decoding. It exposes standard OpenAI-compatible endpoints through a user-friendly interface.

**Future Research.** As part of the Waldwicht project, we are developing `autokernel-mlx`, an open-source system for generating, benchmarking, and compiling custom Metal kernels for Apple Silicon inference. It targets three bottlenecks exposed by this study: dynamic precision at inference time, fused mixed-precision dequantization kernels, and KV-cache quantization. The aim is to turn quantization gains into real inference-time wins by reducing memory traffic, improving throughput, and lowering runtime memory cost. `autokernel-mlx` follows an LLM-in-the-loop generate-evaluate-refine workflow for kernel synthesis and optimization. See section 12 for details.

---

## 1. Introduction

In a prior study on Gemma 4 E4B (4.3B effective parameters, 42 decoder layers, 15.0 GB BF16), we demonstrated that precision allocation — assigning different bit widths to different layers — outperforms all algorithmic optimizations (rotation, equalization, permutation) for 3-bit post-training quantization. That study's central finding was a **rotation paradox**: block-Hadamard rotation reduced offline quantization error at 1-bit (+8.5% MSE) but catastrophically degraded generation quality at 3-bit (+390% perplexity), producing multi-language token leaking (Thai, Hindi, Japanese, Tamil characters mid-sentence in English). The E4B optimal configuration uses no rotation — only a three-tier precision allocation scheme (4-bit edge layers, 4-bit near-edge layers, 3-bit middle layers) achieving 4.24 GB (3.5× compression) with coherent generation on all tasks. Additionally, E4B exhibited a group size inversion: at 3-bit, larger groups (g128) outperformed smaller ones (g32) because metadata overhead at small group sizes consumed storage without improving fidelity.

A natural question arises: **does this finding generalize to smaller models in the same family?** Smaller models have less redundancy, different layer sensitivity profiles, and proportionally larger embedding tables relative to decoder weights. These structural differences could change the optimal quantization strategy.

We investigate this question on Gemma 4 E2B, the 2.3-billion-parameter variant in the Gemma 4 family. In Phases 1–3, using the same ROTQ pipeline — with rotation disabled in all primary sweeps except a dedicated Phase 1 sanity check — evaluation methodology, and test prompts as the E4B study, we run a systematic ablation of 22 configurations. The results reveal a **qualitative difference** between the two model sizes: *layer-level* precision allocation that helps the larger model actively *hurts* the smaller one.

In Phase 4, we go deeper — shifting from layer-level to **component-level** mixed-precision, a new quantization paradigm that assigns different bit-widths to different *weight types* (attention projections, MLP matrices, PLE embeddings, PLE gating, and main embeddings/LM-head) rather than to different *layers*. This is not merely a different configuration within existing tooling — it requires non-trivial changes to inference engines that assume uniform or layer-level quantization. Using a quality-gated pipeline with an 8-prompt LLM-judge (Anthropic Claude Opus 4.6 High) evaluation, we evaluate 28 component-level configurations to find per-component bit-width floors and their valid compositions. This approach succeeds where layer-level mixed-precision failed, achieving **2.32 GB** (text-only) — 28% below the uniform 4-bit quality floor (3.22 GB). We identify and implement the three inference engine adaptations required to make component-level quantization work in practice on MLX (Section 9.5), and release them as a patch to mlx-lm.

---

## 2. Gemma 4 E2B Architecture

### 2.1 Model Specifications

| Feature | E2B | E4B (for comparison) |
|---|---|---|
| Decoder layers | 35 | 42 |
| Hidden dimension | 1536 | 2560 |
| MLP intermediate | 6144 | 10240 |
| Attention heads / KV heads | 8 / 1 | 8 / 2 |
| Head dimensions | 256 (sliding) / 512 (global) | 256 (sliding) / 512 (global) |
| Vocabulary | 262,144 | 262,144 |
| Per-Layer Embeddings (PLE) | 35 × Embedding(262K, 256) + gating | 42 × Embedding(262K, 256) + gating |
| KV sharing | Last 20 layers reuse earlier KV caches | Last 18 layers |
| Attention pattern | 4 sliding + 1 global, repeating (×7) | 5 sliding + 1 global, repeating (×7) |
| Tied embeddings | Yes | Yes |
| Total parameters | ~5.1B | ~8.6B |
| Effective parameters | ~2.3B | ~4.3B |
| BF16 model size | 10.25 GB | 15.0 GB |
| Multimodal components | Audio tower + Vision tower | Audio tower + Vision tower |

### 2.2 Structural Differences Relevant to Quantization

**1. Larger embedding-to-decoder ratio.** The vocabulary embedding (262K × 1536 = 768 MB in BF16) is a comparable fraction of E2B's 10.25 GB (7.3%) to E4B's (262K × 2560 = 1280 MB out of 15.0 GB, 8.3%). However, total embedding weight (vocabulary + 35 PLE tables of 128 MB each) constitutes ~50% of E2B's BF16 size versus ~43% for E4B. Embedding quantization therefore has proportionally greater impact on E2B.

**2. Fewer middle layers.** With 35 total layers, edge protection of $e=4$ layers per side leaves only 27 middle layers. Adding near-edge protection of $ne=4$ leaves only 19 middle layers (54% of total). For E4B (42 layers), the same configuration leaves 26 middle layers (62%). E2B has less "middle" to compress.

**3. Single KV head.** E2B uses 1 KV head (vs E4B's 2), making K/V projections smaller: (256, 1536) for sliding layers. These small matrices have dimensions divisible by 128 but represent a smaller fraction of total weights.

### 2.3 Quantization Compatibility

All weight dimensions are divisible by $g=128$ without padding — no shape-related quantization issues arise for any group size tested (32, 64, 128).

**Note:** E2B includes audio tower and vision tower weights that are excluded from quantization (skip rule for `ndim > 2` weights in the ROTQ pipeline).

---

## 3. Experimental Setup

### 3.1 Methodology

We follow the identical evaluation protocol used for E4B:

- **Perplexity (PPL):** Measured on a short calibration paragraph (identical text across all experiments). Values are useful for relative ranking within this study but not for cross-study comparison.
- **Generation quality:** 5 chat-templated prompts at temperature 0, max 60 tokens:
  1. Arithmetic: "What is 2+2? Answer with just the number"
  2. Science: "Explain quantum entanglement in 2 sentences"
  3. Code: "Write a Python function that checks if a string is a palindrome"
  4. Thermodynamics: "What are the three laws of thermodynamics?"
  5. Translation: "Translate 'The cat sat on the mat' into French"
- **Model size:** On-disk size of the exported model directory.

**Size measurement note.** The ROTQ pipeline (Phases 1–3) exports include ~0.63 GB of multimodal tower weights (audio + vision) as unquantized BF16. The AQ-gated pipeline (Phase 4) exports text-only weights. All Phase 1–3 sizes in this paper are reported *including* multimodal overhead to match the actual export artifacts. For text-weight-only comparison, subtract ~0.63 GB from Phase 1–3 sizes (e.g., q3-g64 text-only ≈ 2.64 GB; BF16 text-only ≈ 9.6 GB). Phase 4 sizes are directly comparable to each other and to deployment-relevant model sizes. The PPL values match exactly between pipelines for the same quantization (e.g., uniform 4-bit g64: PPL 32,443 in both), confirming the text weights are identical.
- **Speed:** Average tokens/second across generation prompts (excluding first-token latency).
- **BF16 source:** `google/gemma-4-E2B-it` (HuggingFace).

### 3.2 Pipeline Adaptations for E2B

The original E4B-based ROTQ pipeline required three modifications for E2B:

1. **Layer count patching:** `module_policy.N_LAYERS = 35` (from 42), adjusting edge/near-edge/middle layer boundaries.
2. **Multimodal weight skipping:** Added `ndim > 2` and `audio_tower`/`vision_tower` skip rules to `classify_weight()`, since E2B's HF checkpoint includes non-text modality weights.
3. **Config key normalization:** HF weight keys (`model.language_model.X`) must be mapped to post-sanitize model parameter paths (`language_model.model.X`) in the quantization config, and stacked PLE keys must be expanded to per-layer variants (`.0` through `.34`).

---

## 4. Phase 1: Baselines and Rotation Sanity Check

### 4.1 Configurations

| Model | Configuration | Size |
|---|---|---|
| bf16 | Reference (BF16) | 10.25 GB |
| q4-g64 | Uniform 4-bit, g=64 | 3.85 GB |
| q3-g128 | Uniform 3-bit, g=128 | 2.98 GB |
| q3-g64 | Uniform 3-bit, g=64 | 3.27 GB |
| q3-plain-e4 | 3-bit g128 middle + 4-bit g64 edge (e=4), no transforms | 3.58 GB |
| q3-rotq-sanity | 3-bit g128 middle + 4-bit edge + block-WHT rotation | 3.58 GB |

### 4.2 Results

| Model | PPL | Size (GB) | Speed (tok/s) | Gen Quality |
|---|---|---|---|---|
| **q3-g64** | **5,452** | **3.27** | **40.7** | **5/5 ✅** |
| q3-rotq-sanity | 12,276 | 3.58 | 33.4 | 5/5 ✅ |
| q3-g128 | 14,701 | 2.98 | 46.5 | 5/5 ✅ |
| q3-plain-e4 | 18,542 | 3.58 | 40.9 | 5/5 ✅ |
| q4-g64 | 32,443 | 3.85 | 34.3 | 5/5 ✅ |
| bf16 | 43,302 | 10.25 | 15.2 | 5/5 ✅ |

### 4.3 Observations

Uniform q3-g64 has the lowest PPL (5,452) while remaining smaller than all mixed-precision alternatives (3.27 GB vs 3.44–3.85 GB). All six configs produce coherent output on all 5 prompts — unlike E4B, where rotation and uniform 3-bit caused generation failures. Mixed-precision (e4) and rotation configs are all worse than uniform q3-g64 on PPL, despite being larger. Rotation reduces PPL modestly within mixed configs (12,276 vs 18,542) — the opposite direction from E4B's +390% degradation — but remains irrelevant since uniform wins.

**PPL caveat.** BF16 shows the highest PPL (43,302). This inversion (quantized < BF16) is an artifact of our short calibration text interacting with E2B's chat-tuned distribution. PPL is used only for relative ranking among quantized configs in this study, not as an absolute quality measure. Task-quality evaluation (Phase 4) supersedes PPL for practical decisions.

---

## 5. Phase 2: Configuration Knobs Sweep

### 5.1 Configurations

Eight configs varying edge layer count (2–5), group size (32, 64, 128), and near-edge 4-bit promotion:

### 5.2 Results

| Configuration | Edge | GS | NE→4-bit | 4-bit Layers | 3-bit Layers | Size (GB) | PPL | Speed (tok/s) |
|---|---|---|---|---|---|---|---|---|
| **e4-g32** | 4 | 32 | No | 8 | 27 | 3.85 | **6,788** | 35.9 |
| e2-g128 | 2 | 128 | No | 4 | 31 | 3.54 | 13,899 | 41.7 |
| e4-g64 | 4 | 64 | No | 8 | 27 | 3.67 | 14,378 | 39.1 |
| e4-g128 | 4 | 128 | No | 8 | 27 | 3.58 | 18,542 | 41.1 |
| e5-g128 | 5 | 128 | No | 10 | 25 | 3.60 | 19,526 | 39.4 |
| e3-g128 | 3 | 128 | No | 6 | 29 | 3.56 | 20,347 | 41.1 |
| e3-g128-ne4bit | 3 | 128 | Yes | 14 | 21 | 3.64 | 26,098 | 39.3 |
| e4-g128-ne4bit | 4 | 128 | Yes | 16 | 19 | 3.66 | 34,394 | 39.4 |

All configs: 5/5 generation quality ✅.

### 5.3 Observations

Group size is the dominant knob: g32 → PPL 6,788, g64 → 14,378, g128 → 18,542 (all e4). This reverses E4B's g128 preference, likely because E2B's narrower weight matrices (1536 vs 2560) have more intra-group variance at coarse granularity. Near-edge 4-bit promotion is harmful (+85% PPL for e4-g128), opposite to E4B's 26% benefit. The best mixed config (e4-g32, PPL 6,788 at 3.85 GB) still loses to uniform q3-g64 (5,452 at 3.27 GB) on both metrics.

---

## 6. Phase 3: Near-Edge Depth Sweep

### 6.1 Configurations

With near-edge promotion already shown to hurt in Phase 2 (g128), we still swept depth systematically to confirm. Eight configs vary near-edge layer count (2–6), near-edge group size (g64/g128), and edge count (3–4).

### 6.2 Results

| Configuration | Edge | NE | NE GS | 4-bit Layers | 3-bit Layers | %4-bit | Size (GB) | PPL | Speed (tok/s) |
|---|---|---|---|---|---|---|---|---|---|
| e3-ne6-g128 | 3 | 6 | 128 | 18 | 17 | 51% | 3.44 | **8,437** | 42.1 |
| e4-ne4-g128 | 4 | 4 | 128 | 16 | 19 | 46% | 3.43 | 16,979 | 42.5 |
| e4-ne2-g128 | 4 | 2 | 128 | 12 | 23 | 34% | 3.41 | 17,569 | 43.3 |
| e4-ne3-g128 | 4 | 3 | 128 | 14 | 21 | 40% | 3.42 | 18,123 | 43.3 |
| e4-ne3-g64 | 4 | 3 | 64 | 14 | 21 | 40% | 3.64 | 26,098 | 39.2 |
| e3-ne4-g64 | 3 | 4 | 64 | 14 | 21 | 40% | 3.64 | 26,098 | 39.4 |
| e4-ne4-g64 | 4 | 4 | 64 | 16 | 19 | 46% | 3.66 | 34,394 | 39.9 |
| e4-ne6-g128 | 4 | 6 | 128 | 20 | 15 | 57% | 3.46 | 63,147 | 42.1 |

All configs: 5/5 generation quality ✅.

### 6.3 Observations

The best near-edge config (e3-ne6-g128, PPL 8,437 at 3.44 GB) is still 55% worse than uniform q3-g64. Increasing the 4-bit fraction to 57% (e4-ne6) degrades PPL to 63,147. NE group size at g64 hurts rather than helps. Unlike E4B, configs with the same total 4-bit layer count produce different PPL depending on the partition.

---

## 7. Phase 4: AQ-Gated Component-Level Quantization

Phases 1–3 explored *layer-level* precision allocation (which decoder layers get more bits). The universal failure of mixed-precision at that granularity motivated a shift to *component-level* allocation: assigning different bit-widths to different weight types across *all* layers simultaneously.

**Methodological note.** Component-level quantization is not merely a new configuration within existing quantization frameworks — it is a new quantization method. Existing inference engines — including MLX, llama.cpp, and vLLM — assume either uniform quantization (one bit-width for all quantizable weights) or layer-level mixed-precision (one bit-width per layer). Component-level allocation breaks this assumption: within a single layer, different weight matrices (e.g., `q_proj` at 5-bit, `gate_proj` at 4-bit, PLE embedding at 3-bit) must be dequantized at different precisions. This requires changes to the inference engine's weight loading, quantization config resolution, and — for architectures with KV-cache sharing like Gemma 4 — attention dispatch logic. We detail the required adaptations in Section 9.5.

### 7.1 Methodology

**Component groups.** We partition all quantizable weights into five groups based on their architectural role:

| Group | Weight Keys | Count | Role |
|---|---|---|---|
| **attention** | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 140 matrices | Self-attention projections |
| **mlp** | `gate_proj`, `up_proj`, `down_proj` | 105 matrices | Feed-forward network |
| **ple_embeddings** | `embed_tokens_per_layer.{0..34}` | 35 tables | Per-layer embedding lookups (262K×256 each) |
| **ple_gate_proj** | `per_layer_input_gate`, `per_layer_projection` | 70 matrices | PLE gating (256×1536 and 1536×256) |
| **main_embed_lmhead** | `embed_tokens`, `lm_head` | 2 tables | Main vocabulary embedding + output head (262K×1536) |

**Quality gate.** Perplexity proved unreliable for E2B (Section 4.3: BF16 has higher PPL than 3-bit). We therefore designed an 8-prompt answer-quality (AQ) evaluation:

| ID | Prompt | Expected Answer |
|---|---|---|
| 0 | What is 2+2? Answer with just the number. | 4 |
| 1 | Solve: 17 × 23. Show your work step by step. | 391 |
| 2 | If a train travels 120 km in 2 hours, what is its average speed in m/s? | 50/3 ≈ 16.67 m/s |
| 3 | What is the derivative of x³ + 2x² − 5x + 3? | 3x² + 4x − 5 |
| 4 | A box has 3 red, 5 blue, and 2 green balls. Probability of picking blue? | 1/2 |
| 5 | What is the sum of the first 100 positive integers? | 5050 |
| 6 | Convert 0.375 to a fraction in simplest form. | 3/8 |
| 7 | If f(x) = 2x + 1 and g(x) = x², what is f(g(3))? | 19 |

Each response is scored on three dimensions (0–10 scale): **correctness** (right answer), **completion** (full solution shown), and **reasoning hygiene** (no contradictions, loops, or junk tokens). A variant **passes** the quality gate if all three dimension averages ≥ 9.0/10. BF16 baseline scores 10.0/10.0/10.0 on all prompts.

**Top-down sieve protocol:**
1. **Step 1 (Uniform ladder):** Test uniform quantization at 6→5→4→3 bits. Find the lowest uniform bit-width that passes AQ — this is the *uniform quality floor*.
2. **Step 2 (Component floors):** Starting from the uniform floor, demote one component group at a time to the next lower rung. If it passes, try the rung below, etc. This finds each component's *independent minimum bit-width*.
3. **Combination testing:** Test compositions of component floors, since independent floors may not compose.

### 7.2 Step 1: Uniform Quality Ladder

| Bits | Size (GB) | PPL | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|---|
| 6-bit g64 | 4.50 | 53,964 | 10.0 | 9.6 | 10.0 | **PASS** |
| 5-bit g64 | 3.86 | 42,559 | 10.0 | 10.0 | 10.0 | **PASS** |
| **4-bit g64** | **3.22** | **32,443** | **10.0** | **9.6** | **10.0** | **PASS** |
| 3-bit g64 | 2.59 | 5,452 | 6.9 | 8.6 | 6.4 | **FAIL** |

**Uniform quality floor: 4-bit.** Despite having the best PPL among all configs in Phases 1–3, uniform 3-bit fails the strict AQ gate — it produces incorrect arithmetic (17×23 ≠ 391), incomplete derivations, and reasoning contradictions. This confirms that PPL and answer quality measure different things for this model.

### 7.3 Step 2: Per-Component Quality Floors

Each row demotes one component from the 4-bit floor while keeping all others at 4-bit:

| Component | 3-bit | 2-bit | 1-bit | Floor |
|---|---|---|---|---|
| **ple_embeddings** | PASS (10.0/9.8/10.0) | PASS (10.0/9.8/10.0) | FAIL (8.8/8.8/8.4) | **2-bit** |
| **attention** | PASS (9.6/9.2/9.2) | FAIL (2.2/2.1/2.1) | — | **3-bit** |
| **mlp** | PASS (9.8/9.6/9.4) | FAIL (0.0/0.0/0.0) | — | **3-bit** |
| **ple_gate_proj** | PASS (10.0/9.8/10.0) | FAIL (3.6/5.2/4.0) | — | **3-bit** |
| **main_embed_lmhead** | PASS (10.0/10.0/10.0) | PASS (9.4/9.1/9.8) | FAIL (5.0/6.0/5.0) | **2-bit** |

PLE embeddings tolerate 2-bit (failing only at 1-bit), while attention is the most sensitive core component — 2-bit produces scores ~2/10, and even 3-bit (9.6/9.2/9.2) is the lowest passing score among all components. MLP shows binary behavior: clean pass at 3-bit, total collapse at 2-bit (0.0/0.0/0.0). Main embed+LM-head passes at 2-bit (9.4/9.1/9.8).

**On-disk sizes for single-component demotion variants:**

| Variant | Size (GB) | Δ from 4-bit |
|---|---|---|
| 4-bit uniform (baseline) | 3.22 | — |
| ple_embeddings → 3-bit | 2.93 | −0.29 |
| ple_embeddings → 2-bit | 2.64 | −0.58 |
| attention → 3-bit | 3.18 | −0.04 |
| mlp → 3-bit | 3.02 | −0.20 |
| ple_gate_proj → 3-bit | 3.22 | −0.00 |
| main_embed_lmhead → 3-bit | 3.17 | −0.05 |
| main_embed_lmhead → 2-bit | 3.12 | −0.10 |

The largest size savings come from PLE embeddings (35 large tables) and MLP (105 matrices), while attention and PLE gate projections are small.

### 7.4 The Composition Problem

A critical finding: **component floors do not compose naively.** Applying all five independent floors simultaneously produces severe failure:

| Configuration | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|
| All at independent floors (attn=3, mlp=3, ple=2, gate=3, embed=2) | 2.24 | 4.0 | 4.6 | 4.1 | **FAIL** |

The model degenerates into repetitive loops ("you you you..."), self-contradictions, and wrong answers. The combined quantization error from multiple demoted groups exceeds the model's tolerance, even though each demotion passes individually.

### 7.5 Step 2b: Systematic Combination Testing

We tested 8 combination configs to find which compositions of component demotions are safe.

**Compositions that fail (attention + MLP both at 3-bit):**

| Configuration | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|
| attn=3 + mlp=3 | 2.97 | 7.1 | 8.4 | 7.8 | FAIL |
| attn=3 + mlp=3 + ple=2 | 2.38 | 7.5 | 10.0 | 8.1 | FAIL |
| attn=3 + mlp=3 + ple=2 + gate=3 | 2.38 | 6.2 | 7.6 | 6.6 | FAIL |
| attn=3 + mlp=3 + ple=2 + gate=3 + embed=2 | 2.28 | 5.2 | 7.2 | 6.2 | FAIL |

**Compositions that pass (attention stays at 4-bit):**

| Configuration | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|
| **mlp=3 + ple=2 + gate=3 + embed=2** | **2.32** | **9.9** | **9.4** | **9.6** | **PASS** |
| ple=2 + gate=3 + embed=2 | 2.53 | 9.6 | 9.2 | 9.8 | PASS |
| mlp=3 only | 3.02 | 9.4 | 9.4 | 9.1 | PASS |

**One config that fails (attention at 3-bit without MLP):**

| Configuration | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|
| attn=3 + ple=2 + gate=3 + embed=2 | 2.49 | 8.4 | 9.2 | 8.8 | FAIL |

### 7.6 Phase 4 Summary

Attention is the binding constraint in composition: attention at 3-bit passes in isolation (Section 7.3), but no tested multi-component composition with attention below 4-bit passed. MLP and attention cannot both be at 3-bit. Smaller components (PLE embeddings at 2-bit, PLE gate at 3-bit, embed/LM-head at 2-bit) compose in the tested passing combinations as long as attention stays at 4-bit. The best passing composition is:

| Component | Bits | Group Size |
|---|---|---|
| attention (q, k, v, o projections) | 4-bit | 64 |
| mlp (gate, up, down projections) | 3-bit | 64 |
| ple_embeddings (35 per-layer tables) | 2-bit | 64 |
| ple_gate_proj (gate + projection) | 3-bit | 64 |
| main_embed_lmhead (embed + LM head) | 2-bit | 64 |

**Size: 2.32 GB** — 28% below the 4-bit quality floor (3.22 GB), 4.1× compression from text-only BF16 (~9.6 GB). Independent floors are necessary but not sufficient: the actual floor in combination is higher for attention (must stay 4-bit in any passing combo), because per-component quantization errors interact non-linearly.

### 7.7 Step 5: Group Size Tuning

All Phase 4 experiments up to this point used g64 uniformly. We tested three group size variations on the optimal 2.32 GB config (attn=4, mlp=3, ple=2, gate=3, embed=2) to determine whether per-component group size tuning could further improve size or quality.

| Variant | Group Size Change | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|---|
| Baseline (all g64) | — | 2.32 | 9.9 | 9.4 | 9.6 | **PASS** |
| MLP g32, rest g64 | MLP: 64→32 | 2.53 | 9.0 | 9.0 | 9.0 | **PASS** (borderline) |
| All overrides g128 | All: 64→128 | 2.01 | 4.1 | 4.5 | 4.2 | **FAIL** |
| All overrides g32 | All: 64→32 | 2.96 | 9.9 | 9.5 | 10.0 | **PASS** |

g128 causes degenerate output (scores 4.1–4.5/10) — coarse granularity interacts destructively with aggressive 2–3 bit widths. g32 passes (9.9/9.5/10.0) but adds +0.64 GB (27% overhead) in metadata. No variant improves the size–quality tradeoff over g64.

### 7.8 Expanded Benchmark: Diverse Task Evaluation

The 8-prompt AQ gate consists entirely of math/logic prompts. To assess generalization, we designed a 20-prompt expanded benchmark covering function calling (3), multi-step function calling (1), code generation (4), translation (4), multi-step reasoning (4), and creative writing (4). All prompts use greedy decoding (temp=0, max_tokens=512), scored on correctness, completion, and reasoning hygiene (each 0–10).

#### 7.8.1 Per-Category Results

| Category | 2.32 GB | 2.53 GB | 3.22 GB | 3.86 GB |
|---|---|---|---|---|
| | (optimal) | (ple2+gate3+embed2) | (uniform 4-bit) | (uniform 5-bit) |
| Function calling (3) | 6.3 / 7.0 / 7.3 | 9.7 / 9.7 / 9.7 | 9.7 / 9.7 / 9.7 | 9.7 / 9.7 / 9.7 |
| Multi-step FC (1) | 7.0 / 8.0 / 7.0 | 7.0 / 8.0 / 7.0 | 7.0 / 9.0 / 8.0 | 7.0 / 9.0 / 8.0 |
| Code generation (4) | 1.0 / 1.5 / 2.0 | 4.0 / 5.2 / 4.0 | **9.2 / 8.8 / 9.5** | **9.8 / 9.5 / 9.8** |
| Translation (4) | 4.8 / 7.0 / 4.5 | 6.0 / 7.5 / 6.0 | 8.8 / 9.0 / 8.5 | **9.0 / 9.2 / 9.0** |
| Multi-step reasoning (4) | 2.8 / 2.5 / 4.0 | 5.8 / 5.2 / 6.8 | 7.0 / 6.2 / 8.0 | **9.2 / 8.5 / 9.5** |
| Creative writing (4) | 7.2 / 8.8 / 7.5 | 8.0 / 9.5 / 8.2 | **10.0 / 10.0 / 10.0** | **10.0 / 10.0 / 10.0** |
| **OVERALL (20)** | **4.5 / 5.4 / 5.0** | **6.5 / 7.3 / 6.8** | **8.8 / 8.7 / 9.1** | **9.4 / 9.3 / 9.5** |

Values are correctness / completion / reasoning_hygiene averages per category.

#### 7.8.2 Expanded Quality Gate Results

Using a relaxed threshold of ≥ 7.0/7.0/7.0 (appropriate for a diverse benchmark where some categories are inherently harder):

| Config | Size (GB) | tok/s | Corr. | Comp. | Reas. | Expanded Quality Gate |
|---|---|---|---|---|---|---|
| Optimal (attn=4, mlp=3, ple=2, gate=3, embed=2) | 2.32 | — | 4.45 | 5.40 | 5.05 | **FAIL** |
| ple2+gate3+embed2 (attn=4, mlp=4) | 2.53 | — | 6.55 | 7.35 | 6.80 | **FAIL** |
| **5-bit mixed B (attn=5, mlp=4, ple=3, gate=4, embed=3)** | **2.96** | **46.6** | **8.75** | **8.95** | **9.05** | **PASS** |
| 5-bit mixed A (attn=5, mlp=5, ple=3, gate=4, embed=3) | 3.17 | 46.3 | 9.25 | 9.15 | 9.35 | **PASS** |
| Uniform 4-bit g64 | 3.22 | — | 8.80 | 8.70 | 9.05 | **PASS** |
| Uniform 5-bit g64 | 3.86 | 44.1 | 9.40 | 9.35 | 9.50 | **PASS** |

*Throughput measured on MacBook Air M4 24 GB, 256-token generation benchmark (3-run average, greedy decoding).*

#### 7.8.3 Observations

The 2.32 GB config passes the 8-prompt math gate but fails the expanded benchmark (4.5/5.4/5.0) — its quality is narrow, limited to conversational math. Code generation is the sharpest discriminator: correctness jumps 1.0→4.0→9.2→9.8 across configs. Below ~3-bit effective depth, code output is non-functional (broken recursion, hallucinated methods, degenerate loops). Creative writing recovers earliest (7.2/8.8/7.5 even at 2.32 GB), consistent with fluency being easier to preserve than structured reasoning. Translation at 2.32–2.53 GB contains fabricated vocabulary and mixed-script errors that disappear at 4-bit.

### 7.9 Relative Weighting Transfer Experiment

The 2.32 GB config's component offsets from the 4-bit base (attn=0, mlp=−1, ple=−2, gate=−1, embed=−2) encode a sensitivity ordering. We transfer this pattern to a 5-bit base:

**Variant A:** attn=5, mlp=5, ple=3, gate=4, embed=3 (conservative — only PLE/embed demoted)
**Variant B:** attn=5, mlp=4, ple=3, gate=4, embed=3 (full offset pattern)

#### 7.9.1 5-Bit Mixed Results

| Config | Size | Corr | Comp | RH | Gate |
|---|---|---|---|---|---|
| Variant A (attn=5, mlp=5, ple=3, gate=4, embed=3) | **3.17 GB** | 9.2 | 9.2 | 9.3 | **PASS** |
| Variant B (attn=5, mlp=4, ple=3, gate=4, embed=3) | **2.96 GB** | 8.8 | 9.0 | 9.1 | **PASS** |
| Uniform 4-bit g64 (baseline) | 3.22 GB | 8.8 | 8.7 | 9.1 | PASS |
| Uniform 5-bit g64 (baseline) | 3.86 GB | 9.4 | 9.3 | 9.5 | PASS |

Both 5-bit mixed variants pass. **Variant B (2.96 GB)** passes the Expanded Quality Gate at **2.96 GB — 8% smaller than uniform 4-bit** while scoring comparably, suggesting that 5-bit attention precision is more valuable than uniform 4-bit across all components.

Per-category breakdown for Variant B (2.96 GB):

| Category | Correctness | Completion | Reasoning Hygiene |
|---|---|---|---|
| Function calling (3) | 9.7 | 9.7 | 9.7 |
| Multi-step FC (1) | 6.0 | 8.0 | 7.0 |
| Code generation (4) | 9.5 | 8.2 | 9.2 |
| Translation (4) | 8.2 | 9.5 | 8.5 |
| Multi-step reasoning (4) | 7.8 | 7.8 | 8.8 |
| Creative writing (4) | 9.5 | 10.0 | 9.8 |

#### 7.9.2 Observations

The component sensitivity ordering (attention > MLP > gate > PLE/embed) transfers from 4-bit to 5-bit base. Variant B (2.96 GB) outperforms uniform 4-bit (3.22 GB) at smaller size by allocating precision to the most sensitive components. The transfer is asymmetric: at 4-bit base, the offsets push components to 2–3 bits (below the quality floor for code/reasoning); at 5-bit base, the same offsets yield 3–4 bits, remaining viable.

---

## 8. Cross-Model Comparison: E2B vs E4B

### 8.1 Optimal Configurations

| | E2B (Phase 1–3) | E2B (Phase 4) | E2B (Phase 4 + Transfer) | E4B |
|---|---|---|---|---|
| **Optimal config** | **q3-g64 (uniform)** | **Component-level mixed** | **5-bit mixed component** | **e4-ne4-g64 (3-tier)** |
| Optimal PPL | 5,452 | — | — | 3,490 |
| AQ gate (math) | FAIL | **PASS** | **PASS** | — |
| AQ gate (expanded) | FAIL | FAIL | **PASS** | — |
| Optimal size | 3.27 GB | **2.32 GB** | **2.96 GB** | 4.24 GB |
| Compression ratio (reported export size) | 3.1×* | — | — | 3.5×* |
| Compression ratio (text-only basis) | — | 4.1× | **3.2×** | — |
| Transforms | None | None | None | None |
| Calibration data | None | None | None | None |
| Precision allocation | Uniform 3-bit | attn=4, mlp=3, ple=2, embed=2 | 5-bit mixed component | 3-tier layer-level |
| Group size | 64 | 64 | 64 | 128 (3-bit) / 64 (4-bit) |

*\* Phase 1–3 and E4B ratios use reported export sizes (including multimodal overhead): 10.25 / 3.27 and 15.0 / 4.24. Phase 4 ratios use text-only basis (~9.6 GB / size) since Phase 4 exports exclude multimodal weights.*

### 8.2 Why Mixed-Precision Helps E4B but Hurts E2B (at Layer Level)

Three likely factors: (1) **Redundancy ratio** — E4B has 42 layers so protecting 16 edge layers still leaves 62% middle; E2B has 35 layers leaving only 54%. (2) **Hidden dimension** — E2B's 1536-wide rows produce only 12 groups at g128 (vs E4B's 20), increasing per-group quantization noise. (3) **Embedding dominance** — total embedding weight (main + PLE) is ~50% of E2B vs ~43% of E4B, so uniform treatment works better than mixed configs that trade coarser middle-layer quantization for finer embeddings.

### 8.3 The Group Size Reversal

| | E2B Optimal | E4B Optimal |
|---|---|---|
| 3-bit group size | **g64** | **g128** |
| Effective bpw (3-bit) | 3.50 | 3.25 |

E2B's narrower weights (1536 vs 2560) have more intra-group variance at g128, and the quality gain from finer granularity (g64 → PPL 5,452 vs g128 → PPL 14,701) overwhelms the +0.29 GB size increase.

### 8.4 The Rotation Divergence

| | E2B | E4B |
|---|---|---|
| Rotation effect on PPL | −34% (18,542 → 12,276) | **+390%** (4,749 → 23,305) |
| Rotation verdict | Slight benefit, but irrelevant | Catastrophic |

Rotation moderately helps E2B but is irrelevant because uniform q3-g64 outperforms all rotation configs. The divergence suggests the rotation paradox is model-size-dependent.

---

## 9. Discussion

### 9.1 Model Size and Precision Axis

| Aspect | E2B (2.3B) | E4B (4.3B) |
|---|---|---|
| Layer-level precision | Uniform optimal | Three-tier mixed optimal |
| Component-level precision | Mixed (attn=4, mlp=3, ple=2) | Not tested |
| Optimal group size | g64 | g128 (3-bit) / g64 (4-bit) |
| Edge layer protection | Harmful | Essential |
| Rotation effect | Slight benefit | Catastrophic harm |

Quantization recipes do not transfer across model sizes — each size class needs its own validation. For E2B, layer sensitivity is approximately flat but component sensitivity varies enormously (attention requires 4-bit; PLE embeddings tolerate 2-bit). The right axis for precision allocation in smaller models is weight type, not layer index.

Independent component floors do not compose: applying all floors simultaneously produces scores of 4.0–4.6/10. The practical heuristic is to keep the most sensitive component (attention) at the uniform floor and demote everything else.

### 9.2 Robustness of Smaller Models

A striking finding is that all 22 Phase 1–3 configs produce coherent output (5/5 prompt quality), whereas E4B had multiple severe failure modes. Phase 4's stricter AQ gate reveals that this apparent robustness masks subtle quality degradation — uniform 3-bit produces coherent but *incorrect* math, only detectable with targeted evaluation.

### 9.3 PPL vs Answer Quality

PPL ranking is inversely correlated with answer quality on this model (uniform 3-bit: best PPL, fails AQ; BF16: worst PPL, perfect AQ). This likely reflects that our calibration text is raw prose, not instruction-formatted, and quantization may reduce hedging/formatting tokens. For instruction-tuned models, perplexity should be supplemented with task-specific quality evaluation.

### 9.4 Limitations

1. **Two model sizes.** Our size-dependence conclusion rests on two data points (E2B and E4B). Intermediate sizes (Gemma 4 E3B, if it existed) would strengthen the finding.
2. **AQ gate calibration.** The original 8-prompt AQ benchmark is math-heavy. An expanded 20-prompt benchmark (Section 7.8) confirms that the 2.32 GB optimal config fails on diverse tasks (code, translation, reasoning), while uniform 4-bit g64 (3.22 GB) passes. The expanded benchmark uses manual LLM-judge scoring, which may have evaluator variance.
3. **PPL anomaly.** E2B's BF16 reference has higher PPL than all quantized variants on our calibration text, indicating this PPL metric does not reflect absolute quality. All comparisons are relative within-study.
4. **No calibration-based comparison.** GPTQ/AWQ might further improve the component-level configs.
5. **Group size tuning was inconclusive on mixed strategies.** We tested uniform g32, g64, and g128 on the optimal config, confirming g64 as the best tested tradeoff (Section 7.7). Per-component mixed group sizes (e.g., MLP at g32 with attention at g128) remain unexplored.
6. **Composition search is incomplete.** We tested 8 of the many possible component combinations. An exhaustive search might find additional valid configs.

### 9.5 Inference Engine Adaptation for Component-Level Quantization

Component-level mixed-precision is not a tuning knob exposed by existing inference engines. Deploying the configurations developed in Phase 4 required three modifications to the MLX inference stack (mlx-lm), which we contribute as a patch (`patches/mlx-lm-waldwicht.patch`, ~50 lines across two files).

**1. Per-component quantization config resolution.** Standard mlx-lm resolves quantization parameters by exact weight path matching: a weight at path `language_model.model.layers.0.self_attn.q_proj` is looked up directly in the quantization config dictionary. Component-level configs use short architectural names (e.g., `embed_tokens`, `per_layer_input_gate`) that must match weight paths differing in prefix conventions between HuggingFace checkpoints and MLX's post-sanitization model tree. We add leaf-name matching to the quantization class predicate: after checking the full dotted path, it extracts the last path component (e.g., `embed_tokens` from `language_model.model.embed_tokens`) and checks whether it appears as a component-level override in the config. This enables a single config dictionary to specify per-component bit-widths without enumerating every full weight path.

**2. Quantized KV-cache reference propagation.** Gemma 4's KV-sharing mechanism (last 20 of 35 decoder layers reuse KV caches from earlier "originating" layers) interacts non-trivially with quantized KV caches. When KV-cache quantization is enabled (e.g., `--kv-bits 8`), the originating layer's `QuantizedKVCache.update_and_fetch()` returns quantized tuples `((data, scales, biases), (data, scales, biases))` rather than plain `mx.array` tensors. Sharing layers receive these tuples as `shared_kv` but pass `cache=None` to the scaled dot-product attention (SDPA) dispatcher, which checks `hasattr(cache, 'bits')` to decide between quantized and standard SDPA. With `cache=None`, the dispatcher routes to `mx.fast.scaled_dot_product_attention`, which cannot handle quantized tuples — producing a `TypeError`. We fix this by propagating the originating layer's cache object (`shared_cache`) through the sharing path, so the SDPA dispatcher receives a cache with the correct `.bits` attribute and routes to `quantized_scaled_dot_product_attention`.

**3. Shape-aware attention mask handling.** Attention mask dimensions are validated against the key sequence length. With quantized KV caches, keys are tuples `(data, scales, biases)` rather than plain arrays, so `keys.shape[-2]` fails. We add a type check: `keys[0].shape[-2]` for quantized tuples, `keys.shape[-2]` otherwise.

These changes are specific to Gemma 4's KV-sharing architecture but illustrate a general principle: **component-level quantization introduces a new class of inference-time compatibility requirements** that do not arise with uniform or layer-level quantization. Other architectures with weight sharing, grouped-query attention, or cross-layer parameter tying will likely require analogous adaptations when component-level precision is applied.

---

## 10. Conclusion

We evaluated 50 post-training quantization configurations for Gemma 4 E2B in four phases (22 layer-level, 28 component-level), plus expanded-benchmark evaluations of finalists. 

Key findings:

1. **Uniform beats layer-level mixed-precision for E2B.** Uniform 3-bit g64 (3.27 GB, PPL 5,452) outperforms all 22 layer-level mixed-precision configs. Group size g64 outperforms g128 (PPL 5,452 vs 14,701). Both reverse E4B findings.

2. **Component-level mixed-precision is a new quantization method, not a configuration.** Assigning different bit-widths to architectural weight groups (not layers) yields a 2.96 GB general-use config that is 23% smaller and 6% faster than uniform 5-bit at comparable quality — a Pareto improvement that no uniform configuration achieves. This is not a tuning knob within existing frameworks: it requires inference engine changes to support per-component config resolution, quantized KV-cache propagation through weight-sharing paths, and shape-aware attention mask handling (Section 9.5). We release the required changes as a patch to mlx-lm. In all tested passing compositions, attention had to remain at the base bit-width; other components tolerate 1–2 bit demotions.

3. **Independent component floors do not compose.** Applying all independent minimums simultaneously fails (scores 4.0–4.6/10). The heuristic: keep attention at the quality floor, demote everything else.

4. **PPL is inversely correlated with answer quality on this model.** Uniform 3-bit has the best PPL but fails the quality gate; BF16 has the worst PPL but scores 10/10. For chat-instruct models, task-specific evaluation is essential.

5. **Domain-specific gates are misleading.** The 2.32 GB config passes math but fails a diverse 20-prompt benchmark (code, translation, reasoning). A 5-bit mixed variant at **2.96 GB** (3.2× text-weight compression) passes the expanded benchmark, displacing uniform 4-bit (3.22 GB) as the smallest general-use config.

6. **Quantization strategy is model-size-dependent.** The optimal approach differs qualitatively between E2B and E4B across every dimension tested (layer allocation, group size, edge protection, rotation). Recipes do not transfer without validation.

7. **Inference engine adaptation is a prerequisite for component-level quantization.** Three changes to the MLX inference stack — per-component config resolution with leaf-name matching, quantized KV-cache reference propagation through Gemma 4's sharing mechanism, and shape-aware attention mask handling — are required to deploy the configs developed in this study. These changes are architecture-specific but the pattern generalizes: any model with weight sharing or cross-layer parameter tying will need analogous adaptations for component-level precision.

Three practical optima (text-only sizes): **2.32 GB** (math/conversational only), **2.96 GB** (general use, recommended), **3.86 GB** (near-BF16 quality). All achieved with standard MLX `mx.quantize` — the best-performing configurations use no transforms, no rotation, and no calibration data.

---

## 11. Reproduction

All experiments are fully reproducible from the repository. Relevant scripts:

| Phase | Script | Description |
|---|---|---|
| Setup | `pip install -U ./mlx transformers safetensors huggingface_hub` | Install dependencies |
| Phase 1 | `ablation_scripts/ablation_baselines.py` | Baselines and rotation sanity check (6 configs) |
| Phase 2 | `ablation_scripts/ablation_q3_knobs.py` | Configuration knobs sweep (8 configs) |
| Phase 3 | `ablation_scripts/ablation_ne_depth.py` | Near-edge depth sweep (8 configs) |
| Phase 4 Step 1 | `ablation_scripts/step1_uniform_ladder.py` | Uniform quality ladder |
| Phase 4 Step 2 | `ablation_scripts/step2_component_floors.py` | Per-component quality floors |
| Phase 4 Step 3 | `ablation_scripts/step3_ple_sensitivity.py` | PLE sensitivity analysis |
| Phase 4 Step 4 | `ablation_scripts/step4_awq_ple.py` | AWQ-style PLE experiments |
| Phase 4 Step 5 | `ablation_scripts/step5_projection_gs.py` | Projection group size tuning |
| Phase 4 Step 6 | `ablation_scripts/step6_combine.py` | Combination testing |
| Group sizes | `ablation_scripts/group_sizes.py` | Group size tuning on optimal config |
| Expanded AQ | `ablation_scripts/expanded_aq.py` | 20-prompt expanded benchmark |
| 5-bit mixed | `ablation_scripts/expanded_aq_5bit_mixed.py` | 5-bit mixed-precision transfer experiment |
| Export | `ablation_scripts/export_e2b.py` | Export final quantized models |

**Note:** ROTQ pipeline code is included in the repository but not documented in the report since it was ultimately irrelevant to the optimal config.

## 12. Future Research

As part of the Waldwicht project, we are developing `autokernel-mlx`, an open-source tool to auto-regressively generate, optimize, evaluate, and compile custom Metal kernels for Apple Silicon inference. The goal is to close the gap between quantization-time decisions and inference-time execution by targeting three areas identified in this study:

1. **Dynamic precision at inference time.** Our three-tier allocation is static: each layer's bit-width is fixed at export. A lightweight dispatch layer could select between pre-compiled 3-bit and 4-bit Metal kernels per layer based on input complexity (e.g., code/math tokens trigger 4-bit, conversational text uses 3-bit). This requires fused dequantization kernels that can switch precision mid-sequence without reloading weights — achievable with Metal's argument buffers. The potential payoff is 4-bit quality at average 3-bit memory cost.

2. **Fused mixed-precision dequantization kernels.** MLX currently dequantizes all weights through a single kernel path regardless of bit-width. Component-level mixed-precision (as validated in the E2B study) means a single forward pass touches 2-bit, 3-bit, 4-bit, and 5-bit weights. Custom Metal kernels fusing dequantization with the GEMM for each bit-width — rather than dequantizing to BF16 first — would reduce memory bandwidth and intermediate storage, directly improving tok/s.

3. **KV cache quantization kernels.** Gemma 4's KV cache sharing (last 20 layers reuse earlier caches for E2B, 18 for E4B) means cached KV pairs are read far more often than written. A 4-bit or 8-bit KV cache with custom quantize-on-write / dequantize-on-read kernels would substantially reduce inference-time memory without touching model weights — orthogonal to weight quantization and composable with all configurations in this study.

`autokernel-mlx` uses an LLM-in-the-loop pipeline: given a kernel specification and target hardware profile, it generates candidate Metal Shading Language (MSL) implementations, compiles them, benchmarks against a reference (baseline) kernel, and iterates — applying the same generate-evaluate-refine loop used in this study's quality-gated quantization pipeline, but targeting throughput rather than answer quality.

---

## Appendix A: All 22 Layer-Level Configurations Ranked by PPL

*18 rows; rows 11, 14, and 16 merge cross-phase duplicates that produced identical results.*

| Rank | Config | Source | Size (GB) | PPL | Speed (tok/s) |
|---|---|---|---|---|---|
| 1 | **q3-g64** | **Phase 1** | **3.27** | **5,452** | **40.7** |
| 2 | e4-g32 | Phase 2 | 3.85 | 6,788 | 35.9 |
| 3 | e3-ne6-g128 | Phase 3 | 3.44 | 8,437 | 42.1 |
| 4 | q3-rotq-sanity | Phase 1 | 3.58 | 12,276 | 33.4 |
| 5 | e2-g128 | Phase 2 | 3.54 | 13,899 | 41.7 |
| 6 | e4-g64 | Phase 2 | 3.67 | 14,378 | 39.1 |
| 7 | q3-g128 | Phase 1 | 2.98 | 14,701 | 46.5 |
| 8 | e4-ne4-g128 | Phase 3 | 3.43 | 16,979 | 42.5 |
| 9 | e4-ne2-g128 | Phase 3 | 3.41 | 17,569 | 43.3 |
| 10 | e4-ne3-g128 | Phase 3 | 3.42 | 18,123 | 43.3 |
| 11 | q3-plain-e4 / e4-g128 | Phase 1/2 | 3.58 | 18,542 | 41.0 |
| 12 | e5-g128 | Phase 2 | 3.60 | 19,526 | 39.4 |
| 13 | e3-g128 | Phase 2 | 3.56 | 20,347 | 41.1 |
| 14 | e3-g128-ne4bit / e4-ne3-g64 / e3-ne4-g64 | Phase 2/3 | 3.64 | 26,098 | 39.3 |
| 15 | q4-g64 | Phase 1 | 3.85 | 32,443 | 34.3 |
| 16 | e4-g128-ne4bit / e4-ne4-g64 | Phase 2/3 | 3.66 | 34,394 | 39.6 |
| 17 | bf16 | Phase 1 | 10.25 | 43,302 | 15.2 |
| 18 | e4-ne6-g128 | Phase 3 | 3.46 | 63,147 | 42.1 |

---

## Appendix B: Phase 4 Complete Results

### B.1 Step 1: Uniform Quality Ladder (4 configs)

| Config | Size (GB) | PPL | Correctness | Completion | Reasoning | AQ Gate |
|---|---|---|---|---|---|---|
| uniform-6bit-g64 | 4.50 | 53,964 | 10.00 | 9.62 | 10.00 | PASS |
| uniform-5bit-g64 | 3.86 | 42,559 | 10.00 | 10.00 | 10.00 | PASS |
| **uniform-4bit-g64** | **3.22** | **32,443** | **10.00** | **9.62** | **10.00** | **PASS** |
| uniform-3bit-g64 | 2.59 | 5,452 | 6.88 | 8.62 | 6.38 | FAIL |

### B.2 Step 2: Individual Component Demotion (12 configs)

Each variant demotes a single component below the 4-bit floor, keeping all others at 4-bit g64.

| Component | Test Bits | Size (GB) | Correctness | Completion | Reasoning | AQ Gate |
|---|---|---|---|---|---|---|
| ple_embeddings | 3-bit | 2.93 | 10.00 | 9.75 | 10.00 | PASS |
| ple_embeddings | 2-bit | 2.64 | 10.00 | 9.75 | 10.00 | PASS |
| ple_embeddings | 1-bit | 2.34 | 8.75 | 8.75 | 8.38 | FAIL |
| attention | 3-bit | 3.18 | 9.62 | 9.25 | 9.25 | PASS |
| attention | 2-bit | 3.13 | 2.25 | 2.12 | 2.12 | FAIL |
| mlp | 3-bit | 3.02 | 9.75 | 9.62 | 9.38 | PASS |
| mlp | 2-bit | 2.81 | 0.00 | 0.00 | 0.00 | FAIL |
| ple_gate_proj | 3-bit | 3.22 | 10.00 | 9.75 | 10.00 | PASS |
| ple_gate_proj | 2-bit | 3.22 | 3.62 | 5.25 | 4.00 | FAIL |
| main_embed_lmhead | 3-bit | 3.17 | 10.00 | 10.00 | 10.00 | PASS |
| main_embed_lmhead | 2-bit | 3.12 | 9.38 | 9.12 | 9.75 | PASS |
| main_embed_lmhead | 1-bit | 3.07 | 5.00 | 6.00 | 5.00 | FAIL |

### B.3 Step 2b: Combination Testing (9 configs)

Compositions of component demotions. All use g64. Default bits = 4 unless overridden.

| Configuration | attn | mlp | ple | gate | embed | Size (GB) | Corr. | Comp. | Reas. | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| All independent floors | 3 | 3 | 2 | 3 | 2 | 2.24 | 4.00 | 4.62 | 4.12 | FAIL |
| attn3 + mlp3 | 3 | 3 | 4 | 4 | 4 | 2.97 | 7.12 | 8.38 | 7.75 | FAIL |
| attn3 + mlp3 + ple2 | 3 | 3 | 2 | 4 | 4 | 2.38 | 7.50 | 10.00 | 8.12 | FAIL |
| attn3 + mlp3 + ple2 + gate3 | 3 | 3 | 2 | 3 | 4 | 2.38 | 6.25 | 7.62 | 6.62 | FAIL |
| attn3 + mlp3 + ple2 + gate3 + embed2 | 3 | 3 | 2 | 3 | 2 | 2.28 | 5.25 | 7.25 | 6.25 | FAIL |
| attn3 + ple2 + gate3 + embed2 | 3 | 4 | 2 | 3 | 2 | 2.49 | 8.38 | 9.25 | 8.75 | FAIL |
| **mlp3 + ple2 + gate3 + embed2** | **4** | **3** | **2** | **3** | **2** | **2.32** | **9.88** | **9.38** | **9.62** | **PASS** |
| ple2 + gate3 + embed2 | 4 | 4 | 2 | 3 | 2 | 2.53 | 9.62 | 9.25 | 9.75 | PASS |
| mlp3 only | 4 | 3 | 4 | 4 | 4 | 3.02 | 9.38 | 9.38 | 9.12 | PASS |

### B.4 All 28 Phase 4 Configs Ranked by Size (passing only)

| Rank | Config | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|---|
| 1 | **mlp=3, ple=2, gate=3, embed=2 (all g64)** | **2.32** | **9.88** | **9.38** | **9.62** | **PASS** |
| 2 | ple=2, gate=3, embed=2 | 2.53 | 9.62 | 9.25 | 9.75 | PASS |
| 3 | mlp=3, ple=2, gate=3, embed=2 (MLP g32) | 2.53 | 9.00 | 9.00 | 9.00 | PASS |
| 4 | ple_embeddings → 2-bit only | 2.64 | 10.00 | 9.75 | 10.00 | PASS |
| 5 | ple_embeddings → 3-bit only | 2.93 | 10.00 | 9.75 | 10.00 | PASS |
| 6 | mlp=3, ple=2, gate=3, embed=2 (all g32) | 2.96 | 9.88 | 9.50 | 10.00 | PASS |
| 7 | mlp → 3-bit only | 3.02 | 9.75 | 9.62 | 9.38 | PASS |
| 8 | mlp=3 only (combo) | 3.02 | 9.38 | 9.38 | 9.12 | PASS |
| 9 | main_embed_lmhead → 2-bit only | 3.12 | 9.38 | 9.12 | 9.75 | PASS |
| 10 | main_embed_lmhead → 3-bit only | 3.17 | 10.00 | 10.00 | 10.00 | PASS |
| 11 | attention → 3-bit only | 3.18 | 9.62 | 9.25 | 9.25 | PASS |
| 12 | uniform 4-bit g64 | 3.22 | 10.00 | 9.62 | 10.00 | PASS |
| 13 | ple_gate_proj → 3-bit only | 3.22 | 10.00 | 9.75 | 10.00 | PASS |
| 14 | uniform 5-bit g64 | 3.86 | 10.00 | 10.00 | 10.00 | PASS |
| 15 | uniform 6-bit g64 | 4.50 | 10.00 | 9.62 | 10.00 | PASS |

### B.5 Step 5: Group Size Tuning (3 configs)

All variants use the optimal component bit-widths (attn=4, mlp=3, ple=2, gate=3, embed=2). Only group size varies.

| Variant | Default GS | Override GS | Size (GB) | Correctness | Completion | Reasoning | Gate |
|---|---|---|---|---|---|---|---|
| MLP g32, rest g64 | 64 | MLP:32 | 2.53 | 9.00 | 9.00 | 9.00 | PASS |
| All overrides g128 | 128 | All:128 | 2.01 | 4.12 | 4.50 | 4.25 | FAIL |
| All overrides g32 | 32 | All:32 | 2.96 | 9.88 | 9.50 | 10.00 | PASS |

### B.6 Phase 4 Failure Modes

| Config | Size (GB) | Primary Failure Mode |
|---|---|---|
| uniform 3-bit | 2.59 | Wrong arithmetic (17×23), incomplete derivations, contradictions |
| mlp → 2-bit | 2.81 | Total model collapse — outputs pure junk (all scores 0.0) |
| attention → 2-bit | 3.13 | Garbled computation steps, wrong answers (scores ~2/10) |
| ple_gate_proj → 2-bit | 3.22 | Treats g(x)=x² as g(x)=x, nonsensical fraction conversion |
| ple_embeddings → 1-bit | 2.34 | Self-contradictions (50×101=550), confused multiplication methods |
| main_embed_lmhead → 1-bit | 3.07 | Repetitive token loops ("you you you..."), wrong unit conversions |
| All independent floors | 2.24 | Repetitive loops, self-contradictions, wrong answers across board |
| attn=3 + mlp=3 | 2.97 | Combined noise exceeds budget; wrong arithmetic, truncated reasoning |
| All overrides g128 | 2.01 | Degenerate repetition ("a a a..."), garbled loops ("step-故step-故..."), single-token non-sequiturs ("oneself") |

<br>

<div align="center">
  <img src="./waldwicht_artwork.png">
</div>