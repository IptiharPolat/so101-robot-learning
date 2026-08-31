#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Example dry-run:
#   commands/record_vla_episode.sh --episode-id vla_000
# Hardware execution additionally requires --execute and two matching env gates.
exec python scripts/run_vla_record.py \
  --config configs/rig.local.yaml \
  --experiment configs/smolvla_experiment.yaml \
  "$@"
