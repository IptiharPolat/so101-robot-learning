#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec conda run --no-capture-output -n so101-ordering \
    python scripts/run_layout_reach_check.py --layout U3 --execute "$@"
fi

exec conda run --no-capture-output -n so101-ordering \
  python scripts/run_layout_reach_check.py --layout U3 "$@"
