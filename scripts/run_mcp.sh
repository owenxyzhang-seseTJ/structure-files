#!/usr/bin/env bash
set -euo pipefail

# Portable stdio entrypoint for DeepSeek Harness, Codex and Claude Code.
# Resolve the project from this script rather than from the caller's cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DSH_STRUCTURE_ROOTS="${DSH_STRUCTURE_ROOTS:-${PWD}}"

if [[ -x "${DSH_STRUCTURE_PYTHON:-}" ]]; then
  exec "${DSH_STRUCTURE_PYTHON}" -m structure_files.mcp_server
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  exec "${PROJECT_ROOT}/.venv/bin/python" -m structure_files.mcp_server
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --no-sync --project "${PROJECT_ROOT}" python -m structure_files.mcp_server
fi

exec python3 -m structure_files.mcp_server
