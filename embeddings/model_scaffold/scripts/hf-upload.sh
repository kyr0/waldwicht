#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:?repo id required}"
REVISION="${2:-main}"
COMMIT_MESSAGE="${3:-update from local model directory}"
DELETE_PATTERN="${4:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be set in env" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found in PATH" >&2
  exit 1
fi

# Upload current directory to repo root.
# If the repo does not exist yet, hf upload creates it automatically.
if [[ -n "${DELETE_PATTERN}" ]]; then
  exec hf upload "${REPO_ID}" . . \
    --revision "${REVISION}" \
    --commit-message "${COMMIT_MESSAGE}" \
    --delete "${DELETE_PATTERN}" \
    --token "${HF_TOKEN}"
else
  exec hf upload "${REPO_ID}" . . \
    --revision "${REVISION}" \
    --commit-message "${COMMIT_MESSAGE}" \
    --token "${HF_TOKEN}"
fi