#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

make -C "${REPO_ROOT}" convert-embedding \
  EMBEDDING_MODEL="${EMBEDDING_MODEL:-microsoft/harrier-oss-v1-270m}" \
  EMBEDDING_MLX_PATH="${EMBEDDING_MLX_PATH:-embeddings/harrier-oss-v1-270m-mlx}" \
  EMBEDDING_DTYPE="${EMBEDDING_DTYPE:-bfloat16}" \
  EMBEDDING_QUANTIZE="${EMBEDDING_QUANTIZE:-1}" \
  EMBEDDING_Q_GROUP_SIZE="${EMBEDDING_Q_GROUP_SIZE:-64}" \
  EMBEDDING_Q_BITS="${EMBEDDING_Q_BITS:-8}" \
  EMBEDDING_Q_MODE="${EMBEDDING_Q_MODE:-affine}"