#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec python scripts/run_act_eval.py --execute "$@"
fi

exec python scripts/run_act_eval.py --allow-missing-model "$@"
