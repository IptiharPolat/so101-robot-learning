#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec conda run --no-capture-output -n so101-ordering \
  python scripts/compare_act_temporal_gripper.py "$@"
