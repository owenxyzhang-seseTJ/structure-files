#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -x "${DSH_STRUCTURE_PYTHON:-}" ]]; then
  exec "${DSH_STRUCTURE_PYTHON}" -m structure_files "$@"
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  exec "${PROJECT_ROOT}/.venv/bin/python" -m structure_files "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --no-sync --project "${PROJECT_ROOT}" python -m structure_files "$@"
fi

exec python3 -m structure_files "$@"
