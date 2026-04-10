# waldwicht - MLX-powered LLM server
# Usage: make setup && make start

SHELL       := /bin/zsh
VENV        := .venv
UV          := uv
PYTHON      := $(VENV)/bin/python
PID_FILE    := .waldwicht.pid
LOG_FILE    := waldwicht-proxy.log
HOST        := 127.0.0.1
PORT        := 8432
MODEL       ?= kyr0/Gemma-4-Waldwicht-Winzling
DRAFT_MODEL ?=
MAX_BACKENDS := 2
MAX_MEM_UTIL := 80
HF_HOME     ?= $(HOME)/.cache/huggingface

.PHONY: setup start stop status log test test-tools bench download patch unpatch package clean models export-model

setup: _clean _install_uv _venv _deps _ensure_metal_toolchain
	@echo "\n[OK] Setup complete. Run 'make start' to launch the server."
	@echo "    For local models:  make start MODEL=/path/to/model"
	@echo "    For HF models:     make download && make start"

_clean:
	@echo "=> Cleaning previous setup (if any) ..."
	rm -rf $(VENV)
	rm -rf mlx/build

_install_uv:
	@if ! command -v $(UV) &>/dev/null; then \
		echo "=> Installing uv ..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "=> uv already installed: $$($(UV) --version)"; \
	fi

_ensure_metal_toolchain:
	@if ! xcrun metal -v &>/dev/null 2>&1; then \
		echo "=> Metal Toolchain missing - downloading (may need sudo) ..."; \
		xcodebuild -downloadComponent MetalToolchain; \
	else \
		echo "=> Metal Toolchain present."; \
	fi

_venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "=> Creating virtual environment ..."; \
		$(UV) venv $(VENV); \
	else \
		echo "=> Virtual environment exists."; \
	fi

_deps:
	@echo "=> Xcode command line tools ..."
	@xcode-select -p &>/dev/null || (echo "=> Installing Xcode command line tools ..."; xcode-select --install)
	@echo "=> Installing Python dependencies ..."
	$(UV) pip install --quiet pip setuptools openai python-dotenv httpx uvicorn starlette pydantic
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
		echo "=> Using local mlx-lm fork — patches already included, skipping."; \
	fi

download:
	@echo "=> Pre-downloading model $(MODEL) ..."
	$(VENV)/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)')"
	@if [ -n "$(DRAFT_MODEL)" ]; then \
		echo "=> Pre-downloading draft model $(DRAFT_MODEL) ..."; \
		$(VENV)/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(DRAFT_MODEL)')"; \
	fi
	@echo "=> Models cached."

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

test:
	@echo "=> Running integration tests ..."
	MODEL=$(MODEL) $(VENV)/bin/python test.py
	@echo "\n=> Running tool calling tests ..."
	MODEL=$(MODEL) $(VENV)/bin/python test_tools.py
	@echo "\n=> Running test calibration ..."
	MODEL=$(MODEL) $(VENV)/bin/python test_calibration.py

# -- bench ------------------------------------------------------------
bench:
	@$(VENV)/bin/python bench.py

# -- models -----------------------------------------------------------
models:
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

# -- package -----------------------------------------------------------
_ensure_pipx:
	@if ! command -v pipx &>/dev/null; then \
		echo "=> Installing pipx via Homebrew ..."; \
		brew install pipx && pipx ensurepath; \
	else \
		echo "=> pipx already installed: $$(pipx --version)"; \
	fi

package: _ensure_pipx
	@echo "=> Building oMLX macOS app + DMG ..."
	cd omlx/packaging && $(CURDIR)/$(PYTHON) build.py
	@echo "=> Done. Output in omlx/packaging/dist/"

# -- unpatch ----------------------------------------------------------
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
		echo "=> Using local mlx-lm fork — nothing to unpatch."; \
	fi

# -- export-model -----------------------------------------------------
EXPORT_MODEL ?= Gemma-4-Waldwicht-Winzling
OUTPUT_DIR   ?= $(abspath $(HF_HOME)/../mlx)

export-model:
	@echo "=> Exporting $(EXPORT_MODEL) to $(OUTPUT_DIR) ..."
	$(PYTHON) ablation_scripts/export.py --model $(EXPORT_MODEL) --output-dir $(OUTPUT_DIR)

# -- clean ------------------------------------------------------------
clean:
	@echo "=> Removing virtual environment and cached state ..."
	rm -rf $(VENV) $(PID_FILE) $(LOG_FILE) waldwicht-backend-*.log
	@echo "=> Clearing uv git cache (PrismML fork) ..."
	rm -rf $$(python3 -c "import pathlib; p=pathlib.Path.home()/'.cache'/'uv'/'git-v0'; print(p)" 2>/dev/null)
	@echo "=> Clean complete. Run 'make setup' to reinstall."
