#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

config="configs/act_eval_temporal_diagnostic_002.yaml"

if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec conda run --no-capture-output -n so101-ordering \
    python scripts/run_act_eval.py --config "$config" --execute "$@"
fi

exec conda run --no-capture-output -n so101-ordering \
  python scripts/run_act_eval.py --config "$config" --allow-missing-model "$@"
