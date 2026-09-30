#!/usr/bin/env bash
set -euo pipefail

# Optional: allow overriding config path via MODEL_PATH env var or keep defaults
if [ -n "${MODEL_PATH:-}" ]; then
  echo "MODEL_PATH is set to $MODEL_PATH"
fi

exec "$@"
