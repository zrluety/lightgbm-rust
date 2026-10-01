#!/usr/bin/env bash
# Run a command inside the Linux dev environment (used on Windows hosts where
# freshly compiled binaries are blocked; see docs/TESTING.md).
#
#   wsl -d Ubuntu -- bash scripts/wsl-dev.sh cargo test -p lgbm-core
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.local/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/lgbm-target}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/lgbm-venv}"
export UV_LINK_MODE=copy
exec "$@"
