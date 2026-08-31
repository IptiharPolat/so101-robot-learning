#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "${SO101_TIMING_APPROVED:-}" != "YES" ]]; then
  echo "FAIL: SO101_TIMING_APPROVED=YES is required" >&2
  exit 2
fi

exec conda run --no-capture-output -n so101-ordering \
  python scripts/check_act_end_to_end_timing.py "$@"
