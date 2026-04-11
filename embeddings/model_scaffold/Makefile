SHELL := /usr/bin/env bash

NAME ?= kyr0/Harrier-Waldwicht-Wurzler-MLX
REVISION ?= main
COMMIT_MESSAGE ?= update from local model directory

SCRIPT := ./scripts/hf-upload.sh

.PHONY: upload update check

check:
	@test -n "$(NAME)" || (echo "ERROR: NAME is required, e.g. make upload NAME=your-user/your-model" >&2; exit 1)
	@test -n "$$HF_TOKEN" || (echo "ERROR: HF_TOKEN must be set in env" >&2; exit 1)
	@command -v hf >/dev/null 2>&1 || (echo "ERROR: hf CLI not found in PATH" >&2; exit 1)

upload: check
	@$(SCRIPT) "$(NAME)" "$(REVISION)" "$(COMMIT_MESSAGE)" ""

update: check
	@$(SCRIPT) "$(NAME)" "$(REVISION)" "$(COMMIT_MESSAGE)" "*"