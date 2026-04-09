<div align="center">
  <img src="./waldwicht_banner.png">
</div>


# Waldwicht High Performance MLX Inference Server

This is an OpenAI-compatible LLM inference server for running the [**Waldwicht model family**](https://huggingface.co/kyr0/Waldwicht-8B-mlx-1bit) (Mixed-Precision Quantization models of Google Gemma 4 E2B, preserving output quality, based on an extensive ablation study by Aron Homberg).

The Waldwicht inference server supports 8-bit KV cache quantization, alongside TurboQuant and Speculative Decoding on any Apple Silicon machines via [MLX](https://github.com/ml-explore/mlx) and a fork of [MLX-LM](https://github.com/ml-explore/mlx-lm) in which I implemented several optimiztations.

> Using Waldwicht models, everyone with a Macbook Air M-series processor can now run intelligent AI models with good performance (see below), handling a wide range of tasks, including tool calling and long-context retrieval (I heard you wanted to run OpenClaw locally for free?):

<img src="docs/waldwicht_demo_aichat.gif" alt="Waldwicht demo in aichat terminal UI" width="600"/>

## Requirements / Setup

- macOS on Apple Silicon / M-series processor _(tested on Macbook Air M4 24GB, macOS 15.7.3)_
- [Homebrew](https://brew.sh/) — required for installing packaging tools (`pipx`). Install with:
  ```sh
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

Open a Terminal and run:

```sh
make setup   # install uv, venv, deps, download model
```

**Note**: The setup process includes downloading the `Waldwicht-Winzling` model from HuggingFace, which is around 3 GB in size. Make sure you have a stable internet connection and enough disk space. **The first-time installation process may take around 10-15 minutes, especially on the first run when it compiles the MLX extensions.**

## Available Models

The Waldwicht model family was developed through a systematic ablation study of 22 layer-level and 28 component-level quantization configurations on Google Gemma 4 E2B (2.3B effective parameters). See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for full details.

| Model | Size | tok/s | Peak RAM | Configuration |
|---|---|---|---|---|
| **[Waldwicht-Winzling](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Winzling)** | **2.96 GB** | ~51.5 | 2.63 GB | attn=5, mlp=4, ple=3, gate=4, embed=3 (recommended) |
| **[Waldwicht-Sproessling](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Sproessling)** | **3.17 GB** | ~48.6 | 2.83 GB | attn=5, mlp=5, ple=3, gate=4, embed=3 |
| **[Waldwicht-Juengling](https://huggingface.co/kyr0/Gemma-4-Waldwicht-Juengling)** | **3.86 GB** | ~47.4 | 3.52 GB | uniform 5-bit g64 (near-BF16 quality) |

*Throughput and peak Metal memory measured on MacBook Air M4 24 GB, 256-token generation, 3-run average, greedy decoding — under 50% CPU load and 50% memory pressure to reflect real consumer conditions.*

## Special features

This server diverges from `mlx-lm` baseline quite a lot:

- Default KV quantization is **8-bit standard** (`--kv-bits 8 --quantized-kv-start 128`), which keeps the first 128 tokens in full precision and quantizes the rest. For more aggressive compression, you can enable TurboQuant instead (see comment in Makefile): `--turbo-kv-bits 3 --turbo-fp16-layers 2` uses PolarQuant with randomized Hadamard rotation + Lloyd-Max codebook.
- **Memory diagnostics**: the patched server exposes `GET /debug/memory` (MLX active/peak/cache/prompt-cache stats) and `POST /debug/clear_cache` (frees MLX buffer cache, returns before/after). The proxy watchdog polls these every tick and clears the buffer cache automatically when a backend goes idle.
- Multi-worker processing using an internal  **reverse proxy** (`proxy.py`) - sits in front of the `mlx-lm` backends and provides:
  - **Connection-aware routing**: tracks active requests per backend, routes to the least-busy one
  - **Auto-scale**: when all backends are busy and a new request arrives, a new backend is spawned on the next port. If memory allows. The `--max-mem-util` setting (default 80%) is a hard ceiling: after spawning, at least 20% of unified memory (including GPU) must remain free. This overrides `--max-backends` if the machine is memory-constrained.
  - **Auto-unload**: after `--idle-timeout` (default 300s) with zero active requests, all backends are killed and memory is freed. The proxy stays alive and accepts new connections; the next request triggers a cold-start (~2-3s), which is a great compromise for consumer workloads.
  - **Memory watchdog**: a background task samples baseline memory footprint per backend (using macOS `footprint` which includes Metal/GPU unified memory). When pressure is detected and the backend is idle >=30s, it first tries clearing the MLX buffer cache; if that's insufficient, it restarts the backend.
  - **SSE streaming relay**: raw chunk pass-through via `httpx` async streaming.

## Quick start

```sh
make setup   # install uv, venv, deps, download model
make start   # launch server on localhost:8430
make test    # run example queries
make stop    # stop the server
```

## Accessing the model

### ChatGPT-like Web Interface

Open a Terminal, and run:

```bash
docker run -d -p 3000:3000 \
  -e OPENAI_API_KEY= \
  -e BASE_URL=http://localhost:8432/v1 \
  -e CUSTOM_MODELS="-all" \
  -e CODE=demo \
  yidadaa/chatgpt-next-web
```

### Any other software

Whatever client software you might use - you will be prompted for providing a config. 

Answer it like this:

```yaml
API Provider: 
  openai-compatible

Provider Name: 
  waldwicht

Model Name (default: "kyr0/Gemma-4-Waldwicht-Winzling"):
  kyr0/Gemma-4-Waldwicht-Winzling

API Base URL: 
  http://localhost:8432/v1

API Key (default: blank, not required): 
  (leave blank)
```

## cURL / API

Sometimes, a cURL says more than a thousand words:

```sh
curl http://localhost:8432/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128, "temperature": 0.7, "top_p": 0.85, "seed": 42, "stream": false, "model": "kyr0/Gemma-4-Waldwicht-Winzling"}'
```

Endpoints:
- `GET  /v1/models` - list available models
- `POST /v1/chat/completions` - chat completion (supports `"stream": true`)
- `GET  /health` - proxy health (backends alive, active connections)


## Makefile Commands

Open any terminal, run:

| Command | Description |
|---------|-------------|
| `make setup` | Install everything + download model |
| `make start` | Start the OpenAI-compatible server |
| `make stop` | Stop the server |
| `make status` | Check if the server is running |
| `make log` | Tail server logs |
| `make test` | Run example queries (health, chat, streaming) |
| `make bench` | Run 50 varied prompts and report tok/s |
| `make models` | List downloaded models with size and location |
| `make clean` | Remove venv, logs, and caches |


## Per-request Hyperparameters

Just send them as part of the JSON body in the `POST /v1/chat/completions` request.

For example (max predictability):

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128,
  "temperature": 0.01,
  "seed": 42
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | `512` | Maximum number of tokens to generate |
| `temperature` | `0.5` | Sampling temperature (`0.0` = greedy/deterministic, higher = more "creative") |
| `top_p` | `0.85` | Nucleus sampling cutoff (lower = narrower vocabulary) |
| `seed` | - | Random seed for reproducible output |
| `stream` | `false` | Stream response as SSE events |

## Thinking / Reasoning Mode

Waldwicht-family models support a reasoning mode where the model produces a `<think>...</think>` block before the final answer. This is **enabled by default** in the server via `--chat-template-args '{"enable_thinking":true}'`.

> **Note**: Reasoning mode requires `temperature > 0` — greedy decoding (`temperature: 0`) suppresses the thinking token.

### Disabling thinking per-request

Send `chat_template_kwargs` in the request body:

```sh
curl http://localhost:8432/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kyr0/Gemma-4-Waldwicht-Winzling",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

With the OpenAI Python SDK:

```python
client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

### Disabling thinking globally

Remove or change the `--chat-template-args` line in the Makefile's `start` target, or override on the command line:

### Using a local model

```sh
make start MODEL=/path/to/model  # path to a directory containing the model files (config.json, tokenizer.model, etc.)
```

## Server Configuration

Edit variables at the top of the `Makefile`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8432` | Proxy listen port |
| `MODEL` | `kyr0/Gemma-4-Waldwicht-Winzling` | HuggingFace model ID |
| `DRAFT_MODEL` | _(empty)_ | Draft model for speculative decoding (see below) |
| `MAX_BACKENDS` | `2` | Maximum backend instances |
| `MAX_MEM_UTIL` | `80` | Max memory utilisation % (must keep rest free) |

### Speculative Decoding

You can optionally enable speculative decoding with a draft model (in case you're planning to use a larger Waldwicht model in the future, or just want to experiment with the feature). For example, using the smaller Waldwicht-Winzling as a draft model for the Waldwicht-Juengling:

```sh
make start DRAFT_MODEL=kyr0/Gemma-4-Waldwicht-Winzling
```

> **Note**: On most Apple Silicon machines, speculative decoding with MLX actually _decreases_ generation speed rather than improving it. The overhead of running the draft model, verifying tokens, and rolling back on misses outweighs the savings from accepted tokens - especially on memory-bandwidth-bound hardware where both models compete for the same unified memory bus. The feature is disabled by default for this reason. It may help on machines with very high memory bandwidth (e.g. M2 Ultra / M4 Max with high-bandwidth unified memory), but benchmark with `make bench` before committing to it.

## About the Waldwicht Models

The Waldwicht family is based on Google Gemma 4 E2B (~5.1B total / 2.3B effective parameters, 35 decoder layers) with mixed-precision quantization. The key finding from the ablation study: **component-level** mixed-precision (different bits per weight type) succeeds where layer-level mixed-precision fails on this model size.

| Property | Value |
|----------|-------|
| Base model | Google Gemma 4 E2B (2.3B effective params) |
| Architecture | 35 decoder layers, 1536 hidden dim, 8 attn heads / 1 KV head |
| Quantization | Mixed-precision component-level (no rotation, no calibration data) |
| Context window | 64k tokens |
| Multimodal | Audio + Vision towers (excluded from quantization) |
| Tool Calling | Supported (single and parallel function calls via OpenAI API) |

### What the models do well

The recommended Waldwicht-Winzling (2.96 GB) passes a diverse 20-prompt benchmark covering code, translation, reasoning, and creative writing at 8.8/9.0/9.1 (avg scores):

- **Concept explanations** — clear, structured answers (e.g. quantum computing with proper use of bold, headings, and analogies)
- **Factual Q&A** — short, accurate responses to direct questions
- **Code generation** — understands programming topics (palindromes, Fibonacci, Python)
- **Creative writing** — haiku, limericks, and freeform poetry with reasonable quality
- **Translation** — handles English↔French and other language pairs
- **Math and reasoning** — arithmetic, thermodynamics, science questions
- **Tool calling** — single and parallel function calls via the OpenAI API; correctly emits `finish_reason: tool_calls` and valid JSON arguments
- **Streaming** — proper SSE streaming with `[DONE]` sentinel
- **Sampling controls** — `temperature`, `top_p`, and `seed` all work as expected; `seed` + low temp produces deterministic output across runs
- **Long context** — needle-in-a-haystack retrieval and coherent summarization of long multi-topic transcripts

## License

MIT (for my code, for 3rd party code in `./mlx` see their respective licenses)

<br>

<div align="center">
  <img src="./waldwicht_artwork.png">
</div>