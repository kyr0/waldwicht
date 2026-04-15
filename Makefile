# waldwicht - Makefile
# Usage: make              # build DMG with quiet console output + full file logs
#        make VERBOSE=1    # stream verbose preflight/build logs
#        make VERBOSE=2    # add shell trace logging to verbose output
#        make help         # show supported public commands
#        make setup        # set up the local Waldwicht development environment
#        make start        # launch the inference server

.DEFAULT_GOAL := dmg
SHELL       := /bin/zsh
VENV        := .venv
UV          := uv
override VENV_PYTHON_VERSION := 3.12.8
PYTHON      := $(VENV)/bin/python
BUILD_PYTHON = $(abspath $(PYTHON))
PID_FILE    := .waldwicht.pid
LOG_FILE    := waldwicht-server.log
MODELS_DIR  ?= $(CURDIR)/models
HOST        := 127.0.0.1
PORT        := 8432
MODEL_REPO  ?= kyr0/Gemma-4-Waldwicht-Winzling
MODEL       ?= $(MODELS_DIR)/$(notdir $(MODEL_REPO))
DRAFT_MODEL_REPO ?=
DRAFT_MODEL ?= $(if $(strip $(DRAFT_MODEL_REPO)),$(MODELS_DIR)/$(notdir $(DRAFT_MODEL_REPO)),)
EMBEDDING_MODEL ?= kyr0/Harrier-Waldwicht-Wurzler-MLX
EMBEDDING_MLX_PATH ?= $(notdir $(EMBEDDING_MODEL))
EMBEDDING_DTYPE ?= bfloat16
EMBEDDING_QUANTIZE ?= 1
EMBEDDING_Q_GROUP_SIZE ?= 64
EMBEDDING_Q_BITS ?= 8
EMBEDDING_Q_MODE ?= affine
EMBED_TEXT ?= The quick brown fox jumps over the lazy dog.
EMBED_MAX_LENGTH ?= 512
EMBED_PREVIEW_DIMS ?= 8
OMLX_EMBED_MODEL_SOURCE ?= $(EMBEDDING_TEST_MODEL)
OMLX_EMBED_TEXT ?= $(EMBED_TEXT)
OMLX_EMBED_HOST ?= 127.0.0.1
OMLX_EMBED_PORT ?= 8436
OMLX_EMBED_TIMEOUT ?= 180
OMLX_EMBED_LOG_LEVEL ?= warning
EMBED_MTEB_MODEL ?= $(EMBEDDING_TEST_MODEL)
EMBED_MTEB_BENCHMARK ?= MTEB(eng, v2)
EMBED_MTEB_TASKS ?=
EMBED_MTEB_PROMPT_FILE ?=
EMBED_MTEB_MAX_LENGTH ?= $(EMBED_MAX_LENGTH)
EMBED_MTEB_BATCH_SIZE ?= 32
EMBED_MTEB_OVERWRITE ?= 0
MLX_EMBEDDINGS_DIR := mlx-embeddings
EMBEDDING_SCAFFOLD_DIR := embeddings/model_scaffold
HF_HOME     ?= $(HOME)/.cache/huggingface
HUGGINGFACE_TOKEN ?=
export HUGGINGFACE_TOKEN
MACOS_MAJOR := $(shell /usr/bin/sw_vers -productVersion 2>/dev/null | cut -d. -f1)
PACKAGE_BUILD_ARGS ?= $(if $(filter 26,$(MACOS_MAJOR)),--macos-target 26.0,)
VERBOSE ?= 0
LOG_TAIL_LINES ?= 60
CHECK_LOG_FILE ?= $(CURDIR)/waldwicht-check.log
PACKAGE_LOG_FILE ?= $(CURDIR)/waldwicht-package.log
PACKAGED_OMLX_APP ?= omlx/packaging/dist/oMLX.app
PACKAGED_OMLX_RUNTIME := $(PACKAGED_OMLX_APP)/Contents/MacOS/python3
PACKAGED_RUNTIME_SOURCES := omlx/packaging/build.py omlx/packaging/venvstacks.toml omlx/pyproject.toml mlx-embeddings/pyproject.toml mlx-embeddings/mlx_embeddings/models/gemma3_text.py

EMBEDDING_CONVERT_ARGS = $(if $(filter 1 true yes on,$(EMBEDDING_QUANTIZE)),--quantize --q-group-size $(EMBEDDING_Q_GROUP_SIZE) --q-bits $(EMBEDDING_Q_BITS) --q-mode $(EMBEDDING_Q_MODE),--dtype $(EMBEDDING_DTYPE))
EMBEDDING_TEST_MODEL ?= $(if $(wildcard $(EMBEDDING_MLX_PATH)/config.json),$(EMBEDDING_MLX_PATH),$(EMBEDDING_MODEL))
EMBED_MTEB_OUTPUT_DIR ?= $(if $(wildcard $(EMBED_MTEB_MODEL)/config.json),$(EMBED_MTEB_MODEL)/mteb-results,mteb-results/$(notdir $(EMBED_MTEB_MODEL)))
EMBED_MTEB_CACHE_DIR ?= $(EMBED_MTEB_OUTPUT_DIR)/cache
EMBED_MTEB_RESULTS_JSON ?= $(EMBED_MTEB_OUTPUT_DIR)/benchmark_results.json
EMBED_MTEB_RESULTS_MD ?= $(EMBED_MTEB_OUTPUT_DIR)/benchmark_results.md

.PHONY: help setup start restart stop status log test test-embed test-embed-omlx embed-mteb bench download patch unpatch check dmg clean models export-model convert-embedding _ensure_packaged_runtime _workspace_deps

help:
	@echo "Supported public commands:"
	@echo ""
	@echo "  Preflight"
	@echo "    check             Verify the macOS DMG packaging environment"
	@echo ""
	@echo "  Proxy serving"
	@echo "    start             Launch the Waldwicht server"
	@echo "    restart           Restart the Waldwicht server"
	@echo "    stop              Stop the Waldwicht server"
	@echo "    status            Show the Waldwicht server status"
	@echo "    log               Tail the Waldwicht server log"
	@echo ""
	@echo "  Proxy testing & benchmarking"
	@echo "    test              Run inference-server integration tests (requires local MODEL)"
	@echo "    bench             Run inference-server benchmarks (requires local MODEL)"
	@echo "    test-embed        Smoke test local embedding inference"
	@echo ""
	@echo "  Models & artifact preparation"
	@echo "    download          Download remote model weights into ./models (use HUGGINGFACE_TOKEN=...)"
	@echo "    models            List downloaded models"
	@echo "    export-model      Export a quantized Waldwicht model variant"
	@echo "    convert-embedding Convert an embedding model to MLX"
	@echo ""
	@echo "  oMLX-specific workflows"
	@echo "    test-embed-omlx   Smoke test the oMLX /v1/embeddings endpoint"
	@echo "    embed-mteb        Run MTEB with the packaged oMLX runtime"
	@echo ""
	@echo "  Building & packaging"
	@echo "    make / dmg        Build the Waldwicht DMG (default)"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean             Remove local build and runtime state"

_install_uv:
	@if ! command -v $(UV) &>/dev/null; then \
		echo "=> Installing uv ..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	elif [ "$(VERBOSE)" -ge 1 ]; then \
		echo "=> uv already installed: $$($(UV) --version)"; \
	fi

_ensure_metal_toolchain:
	@if ! xcrun metal -v &>/dev/null 2>&1; then \
		echo "=> Metal Toolchain missing - downloading (may need sudo) ..."; \
		xcodebuild -downloadComponent MetalToolchain; \
	elif [ "$(VERBOSE)" -ge 1 ]; then \
		echo "=> Metal Toolchain present."; \
	fi

_venv:
	@recreate=0; \
	needs_pip=0; \
	if [ -x "$(PYTHON)" ]; then \
		current_version=$$($(PYTHON) -c 'import platform; print(platform.python_version())'); \
		if [ "$$current_version" != "$(VENV_PYTHON_VERSION)" ]; then \
			recreate=1; \
			echo "=> Virtual environment uses Python $$current_version, expected $(VENV_PYTHON_VERSION) - recreating ..."; \
			rm -rf "$(VENV)"; \
		elif ! "$(PYTHON)" -m pip --version >/dev/null 2>&1; then \
			needs_pip=1; \
		fi; \
	else \
		recreate=1; \
	fi; \
	if [ $$recreate -eq 1 ]; then \
		echo "=> Creating virtual environment with Python $(VENV_PYTHON_VERSION) ..."; \
		$(UV) venv --python $(VENV_PYTHON_VERSION) $(VENV); \
		needs_pip=1; \
	elif [ "$(VERBOSE)" -ge 1 ]; then \
		echo "=> Virtual environment exists (Python $(VENV_PYTHON_VERSION))."; \
	fi; \
	if [ $$needs_pip -eq 1 ]; then \
		echo "=> Bootstrapping pip into $(VENV) ..."; \
		"$(PYTHON)" -m ensurepip --upgrade >/dev/null; \
	fi

_workspace_deps: _install_uv _venv
	@echo "=> Syncing version-locked workspace dependencies into $(VENV) ..."
	@$(UV) sync --quiet --locked --inexact --no-install-project --python $(PYTHON)

_deps: _workspace_deps
	@echo "=> Xcode command line tools ..."
	@xcode-select -p &>/dev/null || (echo "=> Installing Xcode command line tools ..."; xcode-select --install)
	@echo "=> Installing Waldwicht MLX fork (editable, from source) ..."
	PYPI_RELEASE=1 $(UV) pip install --quiet -e ./mlx --no-build-isolation
	@echo "=> Installing mlx-lm fork (editable, Gemma4 support + local MLX) ..."
	$(UV) pip install --quiet -e ./mlx-lm --no-build-isolation
	@echo "=> Installing omlx (editable, against local mlx-lm) ..."
	$(UV) pip install --quiet -e ./omlx --no-deps --no-build-isolation
	@echo "=> Installing omlx remaining dependencies (skipping mlx/mlx-lm) ..."
	$(UV) pip install --quiet -e "./omlx[dev,grammar]" --no-build-isolation 2>&1 | grep -v "already satisfied" || true
	@echo "=> Re-linking local forks (in case omlx deps overwrote them) ..."
	PYPI_RELEASE=1 $(UV) pip install --quiet -e ./mlx --no-build-isolation --no-deps
	$(UV) pip install --quiet -e ./mlx-lm --no-build-isolation --no-deps
	@if [ -f "$(MLX_EMBEDDINGS_DIR)/pyproject.toml" ]; then \
		echo "=> Re-linking local mlx-embeddings fork (in case omlx deps overwrote it) ..."; \
		$(UV) pip install --quiet -e ./$(MLX_EMBEDDINGS_DIR) --no-build-isolation --no-deps; \
	else \
		echo "=> mlx-embeddings submodule missing - keeping installed version."; \
	fi

_embedding_deps: _workspace_deps _ensure_metal_toolchain
	@if [ ! -f "$(MLX_EMBEDDINGS_DIR)/pyproject.toml" ]; then \
		echo "=> Missing mlx-embeddings submodule. Run 'git submodule update --init --recursive $(MLX_EMBEDDINGS_DIR)'."; \
		exit 1; \
	fi
	@if $(PYTHON) tools/check_embedding_deps.py >/dev/null 2>&1; then \
		echo "=> Embedding conversion dependencies already available in $(VENV); skipping install."; \
	else \
		echo "=> Installing embedding conversion dependencies into $(VENV) ..."; \
		PYPI_RELEASE=1 $(UV) pip install --quiet --python $(PYTHON) -e ./mlx --no-build-isolation; \
		$(UV) pip install --quiet --python $(PYTHON) -e ./mlx-lm --no-build-isolation; \
		$(UV) pip install --quiet --python $(PYTHON) mlx-vlm --no-deps; \
		$(UV) pip install --quiet --python $(PYTHON) -e ./$(MLX_EMBEDDINGS_DIR) --no-build-isolation --no-deps; \
	fi

_omlx_embedding_deps: _install_uv _venv _ensure_metal_toolchain
	@if [ ! -f "$(MLX_EMBEDDINGS_DIR)/pyproject.toml" ]; then \
		echo "=> Missing mlx-embeddings submodule. Run 'git submodule update --init --recursive $(MLX_EMBEDDINGS_DIR)'."; \
		exit 1; \
	fi
	@if $(PYTHON) tools/check_omlx_embedding_deps.py >/dev/null 2>&1; then \
		echo "=> oMLX embedding HTTP dependencies already available in $(VENV); skipping install."; \
	else \
		echo "=> Installing oMLX + local embedding dependencies into $(VENV) ..."; \
		$(MAKE) --no-print-directory _deps; \
		$(MAKE) --no-print-directory _embedding_deps; \
	fi

MLX_LM_DIR = $$($(PYTHON) -c "import mlx_lm; print(mlx_lm.__path__[0])")

patch:
	@if echo $(MLX_LM_DIR) | grep -q 'site-packages'; then \
		echo "=> Applying rotation patch to mlx_lm ..."; \
		cd $(MLX_LM_DIR) && patch -p2 --forward < $(CURDIR)/patches/1_rotation.patch; \
		echo "=> Applying turbo quant patch to mlx_lm ..."; \
		cd $(MLX_LM_DIR) && patch -p2 --forward < $(CURDIR)/patches/2_turbo_quant.patch; \
		echo "=> Applying Gemma4 token buffer patch to mlx_lm ..."; \
		cd $(MLX_LM_DIR) && patch -p2 --forward < $(CURDIR)/patches/4_gemma4_token_buffer.patch; \
		echo "=> Applying Waldwicht quantization patch to mlx_lm ..."; \
		cd $(MLX_LM_DIR) && patch -p2 --forward < $(CURDIR)/patches/3_mlx-lm-waldwicht.patch; \
		find $(MLX_LM_DIR) -name '*.rej' -delete 2>/dev/null || true; \
		find $(MLX_LM_DIR) -name '*.orig' -delete 2>/dev/null || true; \
	else \
		echo "=> Using local mlx-lm fork - patches already included, skipping."; \
	fi

_require_huggingface_token:
	@needs_token=0; \
	if [ ! -d "$(MODEL)" ] || [ ! -f "$(MODEL)/config.json" ]; then \
		if [ -n "$(MODEL_REPO)" ]; then \
			needs_token=1; \
		else \
			echo "ERROR: MODEL $(MODEL) does not exist locally and MODEL_REPO is empty."; \
			echo "       Set MODEL=/path/to/local/model or MODEL_REPO=owner/name."; \
			exit 1; \
		fi; \
	fi; \
	if [ -n "$(DRAFT_MODEL)" ] && { [ ! -d "$(DRAFT_MODEL)" ] || [ ! -f "$(DRAFT_MODEL)/config.json" ]; }; then \
		if [ -n "$(DRAFT_MODEL_REPO)" ]; then \
			needs_token=1; \
		else \
			echo "ERROR: DRAFT_MODEL $(DRAFT_MODEL) does not exist locally and DRAFT_MODEL_REPO is empty."; \
			echo "       Set DRAFT_MODEL=/path/to/local/model or DRAFT_MODEL_REPO=owner/name."; \
			exit 1; \
		fi; \
	fi; \
	if [ $$needs_token -eq 1 ] && [ -z "$$HUGGINGFACE_TOKEN" ]; then \
		echo "ERROR: HUGGINGFACE_TOKEN is required for remote Hugging Face downloads."; \
		echo "       Re-run with: HUGGINGFACE_TOKEN=hf_xxx make download"; \
		exit 1; \
	fi

_require_cached_model:
	@if [ ! -d "$(MODEL)" ] || [ ! -f "$(MODEL)/config.json" ]; then \
		echo "ERROR: MODEL $(MODEL) is not available locally."; \
		echo "       Run 'HUGGINGFACE_TOKEN=... make download MODEL_REPO=$(MODEL_REPO)' first or set MODEL=/path/to/local/model."; \
		exit 1; \
	fi
	@if [ -n "$(DRAFT_MODEL)" ] && { [ ! -d "$(DRAFT_MODEL)" ] || [ ! -f "$(DRAFT_MODEL)/config.json" ]; }; then \
		echo "ERROR: DRAFT_MODEL $(DRAFT_MODEL) is not available locally."; \
		echo "       Run 'HUGGINGFACE_TOKEN=... make download DRAFT_MODEL_REPO=$(DRAFT_MODEL_REPO)' first or set DRAFT_MODEL=/path/to/local/model."; \
		exit 1; \
	fi

download: _workspace_deps _require_huggingface_token
	@mkdir -p "$(MODELS_DIR)"
	@prune_model_dir() { \
		model_dir="$$1"; \
		index_path="$$model_dir/model.safetensors.index.json"; \
		if [ ! -d "$$model_dir" ] || [ ! -f "$$index_path" ]; then \
			return 0; \
		fi; \
		MODEL_DIR="$$model_dir" $(PYTHON) -c "import json, os, pathlib; model_dir = pathlib.Path(os.environ['MODEL_DIR']); index = json.loads((model_dir / 'model.safetensors.index.json').read_text()); referenced = {pathlib.Path(v).name for v in index.get('weight_map', {}).values()}; removed = []; [removed.append(p.name) or p.unlink() for p in model_dir.glob('model-*.safetensors') if p.name not in referenced]; print(f'=> Pruned {len(removed)} unreferenced safetensor shard(s) from {model_dir}' if removed else f'=> No unreferenced safetensor shards found in {model_dir}')"; \
	}; \
	if [ -d "$(MODEL)" ] && [ -f "$(MODEL)/config.json" ]; then \
		echo "=> MODEL points to local model directory $(MODEL); skipping Hugging Face download."; \
	else \
		echo "=> Downloading model $(MODEL_REPO) to $(MODEL) ..."; \
		rm -rf "$(MODEL)"; \
		HF_HOME="$(HF_HOME)" $(PYTHON) -c "import os; from huggingface_hub import snapshot_download; snapshot_download('$(MODEL_REPO)', local_dir='$(MODEL)', token=os.environ.get('HUGGINGFACE_TOKEN') or None)"; \
	fi; \
	prune_model_dir "$(MODEL)"
	@if [ -n "$(DRAFT_MODEL)" ]; then \
		prune_model_dir() { \
			model_dir="$$1"; \
			index_path="$$model_dir/model.safetensors.index.json"; \
			if [ ! -d "$$model_dir" ] || [ ! -f "$$index_path" ]; then \
				return 0; \
			fi; \
			MODEL_DIR="$$model_dir" $(PYTHON) -c "import json, os, pathlib; model_dir = pathlib.Path(os.environ['MODEL_DIR']); index = json.loads((model_dir / 'model.safetensors.index.json').read_text()); referenced = {pathlib.Path(v).name for v in index.get('weight_map', {}).values()}; removed = []; [removed.append(p.name) or p.unlink() for p in model_dir.glob('model-*.safetensors') if p.name not in referenced]; print(f'=> Pruned {len(removed)} unreferenced safetensor shard(s) from {model_dir}' if removed else f'=> No unreferenced safetensor shards found in {model_dir}')"; \
		}; \
		if [ -d "$(DRAFT_MODEL)" ] && [ -f "$(DRAFT_MODEL)/config.json" ]; then \
			echo "=> DRAFT_MODEL points to local model directory $(DRAFT_MODEL); skipping Hugging Face download."; \
		else \
			echo "=> Downloading draft model $(DRAFT_MODEL_REPO) to $(DRAFT_MODEL) ..."; \
			rm -rf "$(DRAFT_MODEL)"; \
			HF_HOME="$(HF_HOME)" $(PYTHON) -c "import os; from huggingface_hub import snapshot_download; snapshot_download('$(DRAFT_MODEL_REPO)', local_dir='$(DRAFT_MODEL)', token=os.environ.get('HUGGINGFACE_TOKEN') or None)"; \
		fi; \
		prune_model_dir "$(DRAFT_MODEL)"; \
	fi
	@echo "=> Models downloaded into $(MODELS_DIR)."


# To enable TurboQuant KV cache, add after --max-tokens:
#   --turbo-kv-bits 8 --turbo-fp16-layers 2
start:
	@if [ -f "$(PID_FILE)" ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Proxy already running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "=> Starting waldwicht-proxy on $(HOST):$(PORT) ..."; \
		$(PYTHON) proxy.py \
			--host $(HOST) --port $(PORT) \
			--model $(MODEL) \
			$(if $(DRAFT_MODEL),--draft-model $(DRAFT_MODEL)) \
			--max-backends $(MAX_BACKENDS) \
			--max-mem-util $(MAX_MEM_UTIL) \
			--temp 0.5 --top-p 0.85 \
			--kv-bits 8 \
			--quantized-kv-start 128 \
			--max-tokens 65536 \
			--chat-template-args '{"enable_thinking":true}' \
			>> $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "=> Proxy PID: $$(cat $(PID_FILE))  (log: $(LOG_FILE))"; \
		echo "=> Waiting for proxy + backend to be ready ..."; \
		for i in {1..60}; do \
			if curl -sf http://$(HOST):$(PORT)/v1/models >/dev/null 2>&1; then \
				echo "=> Ready."; \
				break; \
			fi; \
			sleep 2; \
		done; \
	fi
	@$(MAKE) --no-print-directory status

stop:
	@if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "=> Stopping proxy (PID $$PID) ..."; \
			kill $$PID; \
			for i in {1..20}; do \
				kill -0 $$PID 2>/dev/null || break; \
				sleep 0.25; \
			done; \
			if kill -0 $$PID 2>/dev/null; then \
				echo "=> Proxy didn't exit, sending SIGKILL ..."; \
				kill -9 $$PID 2>/dev/null; \
			fi; \
		else \
			echo "=> Proxy already dead (stale PID file)."; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "=> No PID file found."; \
	fi
	@# Kill any orphaned mlx_lm.server backends
	@ORPHANS=$$(pgrep -f 'mlx_lm\.server.*--host 127\.0\.0\.1' 2>/dev/null || true); \
	if [ -n "$$ORPHANS" ]; then \
		echo "=> Killing orphaned backends: $$ORPHANS"; \
		echo "$$ORPHANS" | xargs kill 2>/dev/null; \
		sleep 1; \
		echo "$$ORPHANS" | xargs kill -9 2>/dev/null || true; \
	fi
	@echo "=> Stopped."

# -- status -----------------------------------------------------------
status:
	@if [ -f "$(PID_FILE)" ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Proxy: running (PID $$(cat $(PID_FILE)))"; \
		echo "  URL: http://$(HOST):$(PORT)/v1"; \
		curl -sf http://$(HOST):$(PORT)/health 2>/dev/null | python3 -m json.tool || true; \
		echo "  Models:"; \
		curl -sf http://$(HOST):$(PORT)/v1/models 2>/dev/null | python3 -c "import sys,json; [print(f'    - {m[\"id\"]}') for m in json.load(sys.stdin).get('data',[])]" || true; \
	else \
		echo "Proxy: not running"; \
	fi

log:
	@if [ -f "$(LOG_FILE)" ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file yet."; \
	fi

test: _require_cached_model start
	@echo "=> Running integration tests ..."
	MODEL=$(MODEL) $(VENV)/bin/python test.py
	@echo "\n=> Running tool calling tests ..."
	MODEL=$(MODEL) $(VENV)/bin/python test_tools.py
	@echo "\n=> Running test calibration ..."
	MODEL=$(MODEL) $(VENV)/bin/python test_calibration.py

# -- bench ------------------------------------------------------------
bench: _require_cached_model start
	@$(VENV)/bin/python bench.py

# -- models -----------------------------------------------------------
models: _workspace_deps
	@$(PYTHON) -c "\
	from huggingface_hub import scan_cache_dir; \
	info = scan_cache_dir(); \
	models = sorted([ \
	    (r.repo_id, r.size_on_disk, str(r.repo_path)) \
	    for r in info.repos if r.repo_type == 'model' \
	]); \
	wn = max((len(r) for r, _, _ in models), default=20); \
	wp = max((len(p) for _, _, p in models), default=20); \
	print(f'{\"Model\":<{wn}}  {\"Size\":>8}  Location'); \
	print('-' * (wn + wp + 14)); \
	[print(f'{rid:<{wn}}  {sz/1e9:>7.2f}G  {path}') for rid, sz, path in models]; \
	print(f'\n{len(models)} model(s), {sum(sz for _,sz,_ in models)/1e9:.2f} GB total') \
	"

# -- dmg -----------------------------------------------------------
_ensure_pipx:
	@if ! command -v pipx &>/dev/null; then \
		echo "=> Installing pipx via Homebrew ..."; \
		brew install pipx && pipx ensurepath; \
	elif [ "$(VERBOSE)" -ge 1 ]; then \
		echo "=> pipx already installed: $$(pipx --version)"; \
	fi

check: _ensure_pipx _install_uv _venv _ensure_metal_toolchain _check_package_env

_check_package_env:
	@echo "check"
	@rm -f "$(CHECK_LOG_FILE)"
	@verbose="$(VERBOSE)"; \
	log_file="$(CHECK_LOG_FILE)"; \
	if [ "$$verbose" -ge 1 ]; then \
		echo "Checking macOS DMG packaging environment ..."; \
		echo "Verbosity: $$verbose (0=quiet, 1=verbose, 2=trace)"; \
		echo "Preflight log: $$log_file"; \
	fi; \
	( \
		if [ "$$verbose" -ge 2 ]; then set -x; fi; \
		echo "=> Host: os=$$(uname -s) arch=$$(uname -m)"; \
		if [ "$$(uname -s)" != "Darwin" ]; then \
			echo "ERROR: DMG packaging requires macOS (Darwin); current OS: $$(uname -s)."; \
			exit 1; \
		fi; \
		if [ "$$(uname -m)" != "arm64" ]; then \
			echo "ERROR: oMLX packaging currently requires Apple Silicon (arm64); current arch: $$(uname -m)."; \
			exit 1; \
		fi; \
		macos_version=$$(/usr/bin/sw_vers -productVersion 2>/dev/null || true); \
		if [ -z "$$macos_version" ]; then \
			echo "ERROR: Unable to determine macOS version with sw_vers."; \
			exit 1; \
		fi; \
		macos_major=$${macos_version%%.*}; \
		if [ "$$macos_major" -lt 15 ]; then \
			echo "ERROR: macOS 15+ required for oMLX packaging; detected $$macos_version."; \
			exit 1; \
		fi; \
		echo "=> macOS version: $$macos_version"; \
		if [ ! -x "$(PYTHON)" ]; then \
			echo "ERROR: Missing repo Python at $(PYTHON). Run 'make setup' or plain 'make' first."; \
			exit 1; \
		fi; \
		python_version=$$($(PYTHON) -c 'import platform; print(platform.python_version())'); \
		if [ "$$python_version" != "$(VENV_PYTHON_VERSION)" ]; then \
			echo "ERROR: Expected $(PYTHON) to be Python $(VENV_PYTHON_VERSION), found $$python_version."; \
			exit 1; \
		fi; \
		echo "=> Repo Python: $(PYTHON) ($$python_version)"; \
		if ! $(PYTHON) -m pip --version >/dev/null 2>&1; then \
			echo "ERROR: Repo Python at $(PYTHON) does not have pip available. Recreate the virtualenv or run '$(PYTHON) -m ensurepip --upgrade'."; \
			exit 1; \
		fi; \
		echo "=> Repo pip: available"; \
		if [ ! -d "omlx" ] || [ ! -f "omlx/packaging/build.py" ] || [ ! -f "omlx/packaging/venvstacks.toml" ] || [ ! -f "omlx/pyproject.toml" ] || [ ! -f "omlx/omlx/_version.py" ]; then \
			echo "ERROR: oMLX packaging sources are missing or incomplete under ./omlx."; \
			echo "       Expected: omlx/packaging/build.py, omlx/packaging/venvstacks.toml, omlx/pyproject.toml, omlx/omlx/_version.py"; \
			exit 1; \
		fi; \
		echo "=> oMLX sources: present"; \
		omlx_version=$$($(PYTHON) -c 'from pathlib import Path; import re; content=Path("omlx/omlx/_version.py").read_text(); m=re.search(r"__version__\s*=\s*\"([^\"]+)\"", content); print(m.group(1) if m else "unknown")'); \
		echo "=> oMLX version: $$omlx_version"; \
		for tool in $(UV) pipx hdiutil codesign sips iconutil sw_vers xcrun xcodebuild cc; do \
			if ! command -v $$tool &>/dev/null; then \
				echo "ERROR: Required build tool '$$tool' is missing."; \
				exit 1; \
			fi; \
			echo "=> Tool $$tool: $$(command -v $$tool)"; \
		done; \
		echo "=> uv version: $$($(UV) --version)"; \
		echo "=> pipx version: $$(pipx --version)"; \
		echo "=> hdiutil: $$(hdiutil version 2>/dev/null | sed -n '1p')"; \
		echo "=> codesign: $$(codesign --version 2>/dev/null | sed -n '1p')"; \
		echo "=> cc: $$(cc --version 2>/dev/null | sed -n '1p')"; \
		if ! xcode-select -p &>/dev/null; then \
			echo "ERROR: Xcode command line tools are not configured. Run 'xcode-select --install'."; \
			exit 1; \
		fi; \
		echo "=> xcode-select path: $$(xcode-select -p)"; \
		echo "=> xcodebuild version:"; \
		xcodebuild -version 2>/dev/null | sed 's/^/   /'; \
		if ! xcrun --sdk macosx --show-sdk-path >/dev/null 2>&1; then \
			echo "ERROR: macOS SDK is unavailable via xcrun."; \
			exit 1; \
		fi; \
		echo "=> macOS SDK: $$(xcrun --sdk macosx --show-sdk-path)"; \
		if ! xcrun metal -v &>/dev/null 2>&1; then \
			echo "ERROR: Apple Metal toolchain is unavailable. Install/download it and retry."; \
			exit 1; \
		fi; \
		echo "=> Metal compiler: $$(xcrun --find metal)"; \
		echo "=> metallib: $$(xcrun --find metallib)"; \
		if ! $(PYTHON) omlx/packaging/build.py --help >/dev/null 2>&1; then \
			echo "ERROR: Packaging entrypoint smoke test failed: $(PYTHON) omlx/packaging/build.py --help"; \
			exit 1; \
		fi; \
		echo "=> Packaging CLI smoke test: ok"; \
		echo "=> Summary: macOS=$$(sw_vers -productVersion) sdk=$$(xcrun --sdk macosx --show-sdk-path) python=$$($(PYTHON) -c 'import platform; print(platform.python_version())') package_args='$(if $(PACKAGE_BUILD_ARGS),$(PACKAGE_BUILD_ARGS),<none>)'"; \
	) >> "$$log_file" 2>&1; \
	check_status=$$?; \
	if [ $$check_status -ne 0 ]; then \
		echo "ERROR: Packaging environment preflight failed. See $$log_file."; \
		echo "Last $(LOG_TAIL_LINES) preflight log lines:"; \
		tail -n "$(LOG_TAIL_LINES)" "$$log_file" 2>/dev/null || true; \
		exit $$check_status; \
	fi; \
	if [ "$$verbose" -ge 1 ]; then \
		cat "$$log_file"; \
		echo "[OK] macOS DMG packaging environment looks ready."; \
	fi

dmg: check
	@echo "dmg"
	@rm -f "$(PACKAGE_LOG_FILE)"
	@verbose="$(VERBOSE)"; \
	if [ "$$verbose" -ge 1 ]; then \
		echo "Starting oMLX macOS app + DMG build ..."; \
		echo "Working directory: $(CURDIR)"; \
		echo "Build log: $(PACKAGE_LOG_FILE)"; \
		echo "Verbosity: $$verbose (0=quiet, 1=verbose, 2=trace)"; \
		echo "Final DMG name format: waldwicht-\$$version.dmg"; \
		echo "macOS $(MACOS_MAJOR) packaging args: $(if $(PACKAGE_BUILD_ARGS),$(PACKAGE_BUILD_ARGS),<none>)"; \
	else \
		echo "Building DMG..."; \
	fi; \
	build_started_at=$$(date '+%Y-%m-%d %H:%M:%S'); \
	echo "Build started at: $$build_started_at" >> "$(PACKAGE_LOG_FILE)"; \
	if [ "$$verbose" -eq 0 ]; then \
		echo "Console output hidden by default. Use VERBOSE=1 to stream logs or VERBOSE=2 for shell trace." >> "$(PACKAGE_LOG_FILE)"; \
	else \
		echo "Streaming packaging output to console ..." | tee -a "$(PACKAGE_LOG_FILE)"; \
	fi; \
	echo "Running packaging build.py ..." >> "$(PACKAGE_LOG_FILE)"; \
	if [ "$$verbose" -ge 1 ]; then \
		set -o pipefail; \
		( \
			if [ "$$verbose" -ge 2 ]; then set -x; fi; \
			cd omlx/packaging; \
			"$(BUILD_PYTHON)" build.py $(PACKAGE_BUILD_ARGS) \
		) 2>&1 | tee -a "$(PACKAGE_LOG_FILE)"; \
		build_status=$${pipestatus[1]}; \
	else \
		( \
			cd omlx/packaging; \
			"$(BUILD_PYTHON)" build.py $(PACKAGE_BUILD_ARGS) \
		) >> "$(PACKAGE_LOG_FILE)" 2>&1; \
		build_status=$$?; \
	fi; \
	if [ $$build_status -ne 0 ]; then \
		echo "ERROR: Packaging build failed. See $(PACKAGE_LOG_FILE)." | tee -a "$(PACKAGE_LOG_FILE)"; \
		echo "Last $(LOG_TAIL_LINES) build log lines:"; \
		tail -n "$(LOG_TAIL_LINES)" "$(PACKAGE_LOG_FILE)" 2>/dev/null || true; \
		exit $$build_status; \
	fi; \
	dmg_src=$$(ls -1t omlx/packaging/dist/*.dmg 2>/dev/null | head -n 1); \
	if [ -z "$$dmg_src" ]; then \
		echo "ERROR: Build completed but no DMG was found in omlx/packaging/dist/." | tee -a "$(PACKAGE_LOG_FILE)"; \
		exit 1; \
	fi; \
	dmg_name=$$(basename "$$dmg_src"); \
	dmg_version=$${dmg_name#*-}; \
	final_dmg_name="waldwicht-$$dmg_version"; \
	dmg_dst="$(CURDIR)/$$final_dmg_name"; \
	mv -f "$$dmg_src" "$$dmg_dst"; \
	build_finished_at=$$(date '+%Y-%m-%d %H:%M:%S'); \
	echo "Build finished at: $$build_finished_at" >> "$(PACKAGE_LOG_FILE)"; \
	echo "DMG: $$dmg_dst" | tee -a "$(PACKAGE_LOG_FILE)"; \
	echo "Log: $(PACKAGE_LOG_FILE)" | tee -a "$(PACKAGE_LOG_FILE)"

_ensure_packaged_runtime:
	@needs_build=0; \
	if [ ! -x "$(PACKAGED_OMLX_RUNTIME)" ]; then \
		needs_build=1; \
		echo "=> Packaged oMLX runtime missing - building it first ..."; \
	else \
		for src in $(PACKAGED_RUNTIME_SOURCES); do \
			if [ "$$src" -nt "$(PACKAGED_OMLX_RUNTIME)" ]; then \
				needs_build=1; \
				echo "=> Packaged oMLX runtime is stale (newer source: $$src) - rebuilding ..."; \
				break; \
			fi; \
		done; \
	fi; \
	if [ $$needs_build -eq 1 ]; then \
		$(MAKE) --no-print-directory dmg; \
	else \
		echo "=> Using packaged oMLX runtime at $(PACKAGED_OMLX_APP)"; \
	fi

# -- legacy unpatch maintenance ---------------------------------------
unpatch:
	@if echo $(MLX_LM_DIR) | grep -q 'site-packages'; then \
		echo "=> Reinstalling mlx-lm (clean, unpatched) ..."; \
		$(UV) pip install --quiet --force-reinstall --no-deps mlx-lm; \
		echo "=> Removing leftover patch artifacts ..."; \
		find $(MLX_LM_DIR)/models -name 'turboquant_*.py' -delete 2>/dev/null || true; \
		rm -f $(MLX_LM_DIR)/test_turboquant.py 2>/dev/null || true; \
		find $(MLX_LM_DIR) -name '*.rej' -delete 2>/dev/null || true; \
		find $(MLX_LM_DIR) -name '*.orig' -delete 2>/dev/null || true; \
		echo "=> mlx_lm restored to clean state."; \
	else \
		echo "=> Using local mlx-lm fork - nothing to unpatch."; \
	fi

# -- export-model -----------------------------------------------------
EXPORT_MODEL ?= Gemma-4-Waldwicht-Winzling
OUTPUT_DIR   ?= $(abspath $(HF_HOME)/../mlx)

export-model:
	@echo "=> Exporting $(EXPORT_MODEL) to $(OUTPUT_DIR) ..."
	$(PYTHON) ablation_scripts/export.py --model $(EXPORT_MODEL) --output-dir $(OUTPUT_DIR)

convert-embedding: _embedding_deps
	@echo "=> Converting embedding model $(EMBEDDING_MODEL) to $(EMBEDDING_MLX_PATH) ..."
	@echo "=> Embedding quantization: $(if $(filter 1 true yes on,$(EMBEDDING_QUANTIZE)),enabled ($(EMBEDDING_Q_MODE), $(EMBEDDING_Q_BITS)-bit g$(EMBEDDING_Q_GROUP_SIZE)),disabled (dtype $(EMBEDDING_DTYPE)))"
	HF_HOME="$(HF_HOME)" HF_TOKEN="$(HUGGINGFACE_TOKEN)" HUGGINGFACE_HUB_TOKEN="$(HUGGINGFACE_TOKEN)" $(PYTHON) -m mlx_embeddings.convert \
		--hf-path $(EMBEDDING_MODEL) \
		--mlx-path $(EMBEDDING_MLX_PATH) \
		$(EMBEDDING_CONVERT_ARGS)
	@echo "=> Copying embedding scaffold from $(EMBEDDING_SCAFFOLD_DIR) to $(EMBEDDING_MLX_PATH) ..."
	@mkdir -p "$(EMBEDDING_MLX_PATH)"
	rsync -a --exclude '.DS_Store' "$(EMBEDDING_SCAFFOLD_DIR)/" "$(EMBEDDING_MLX_PATH)/"

test-embed: _embedding_deps
	@echo "=> Embedding test text with $(EMBEDDING_TEST_MODEL) ..."
	HF_HOME="$(HF_HOME)" HF_TOKEN="$(HUGGINGFACE_TOKEN)" HUGGINGFACE_HUB_TOKEN="$(HUGGINGFACE_TOKEN)" $(PYTHON) test_embed.py \
		--model "$(EMBEDDING_TEST_MODEL)" \
		--text "$(EMBED_TEXT)" \
		--max-length $(EMBED_MAX_LENGTH) \
		--preview-dims $(EMBED_PREVIEW_DIMS)

test-embed-omlx: _omlx_embedding_deps
	@echo "=> Smoke testing oMLX /v1/embeddings with $(OMLX_EMBED_MODEL_SOURCE) ..."
	HF_HOME="$(HF_HOME)" HF_TOKEN="$(HUGGINGFACE_TOKEN)" HUGGINGFACE_HUB_TOKEN="$(HUGGINGFACE_TOKEN)" $(PYTHON) test_embed_omlx.py \
		--model-source "$(OMLX_EMBED_MODEL_SOURCE)" \
		--text "$(OMLX_EMBED_TEXT)" \
		--host "$(OMLX_EMBED_HOST)" \
		--port $(OMLX_EMBED_PORT) \
		--timeout $(OMLX_EMBED_TIMEOUT) \
		--log-level "$(OMLX_EMBED_LOG_LEVEL)"

embed-mteb: _ensure_packaged_runtime
	@echo "=> Running $(EMBED_MTEB_BENCHMARK) for $(EMBED_MTEB_MODEL) with the packaged oMLX runtime ..."
	HF_HOME="$(HF_HOME)" python3 tools/run_packaged_python.py \
		--app "$(PACKAGED_OMLX_APP)" \
		tools/embed_mteb.py \
		--model "$(EMBED_MTEB_MODEL)" \
		--benchmark "$(EMBED_MTEB_BENCHMARK)" \
		--output-dir "$(EMBED_MTEB_OUTPUT_DIR)" \
		--cache-dir "$(EMBED_MTEB_CACHE_DIR)" \
		--results-json "$(EMBED_MTEB_RESULTS_JSON)" \
		--results-md "$(EMBED_MTEB_RESULTS_MD)" \
		--max-length $(EMBED_MTEB_MAX_LENGTH) \
		--batch-size $(EMBED_MTEB_BATCH_SIZE) \
		$(if $(strip $(EMBED_MTEB_PROMPT_FILE)),--prompt-file "$(EMBED_MTEB_PROMPT_FILE)") \
		$(if $(strip $(EMBED_MTEB_TASKS)),--tasks $(EMBED_MTEB_TASKS)) \
		$(if $(filter 1 true yes on,$(EMBED_MTEB_OVERWRITE)),--overwrite)

# -- clean ------------------------------------------------------------
clean:
	@echo "=> Removing virtual environment and cached state ..."
	rm -rf $(VENV) $(PID_FILE) $(LOG_FILE)
	@echo "=> Clearing uv git cache (PrismML fork) ..."
	rm -rf $$(python3 -c "import pathlib; p=pathlib.Path.home()/'.cache'/'uv'/'git-v0'; print(p)" 2>/dev/null)
	@echo "=> Clean complete. Run 'make setup' to reinstall."
