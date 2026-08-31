#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

config="configs/act_eval_temporal_diagnostic.yaml"

if [[ "${1:-}" == "--latency" ]]; then
  shift
  exec conda run --no-capture-output -n so101-ordering \
    python scripts/benchmark_act_temporal_ensemble.py --config "$config" "$@"
fi

if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec conda run --no-capture-output -n so101-ordering \
    python scripts/run_act_eval.py --config "$config" --execute "$@"
fi

exec conda run --no-capture-output -n so101-ordering \
  python scripts/run_act_eval.py --config "$config" --allow-missing-model "$@"
