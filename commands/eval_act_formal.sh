#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 ACT30K-FORMAL-NNN [--execute]" >&2
  exit 2
fi

evaluation_id="$1"
shift
exec conda run --no-capture-output -n so101-ordering \
  python scripts/run_act_formal_eval.py "$evaluation_id" "$@"
