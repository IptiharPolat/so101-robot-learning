#!/usr/bin/env bash
set -uo pipefail

env_name="${1:-lerobot}"

run() {
  printf '\n$ %s\n' "$*"
  "$@"
}

run conda env list
run which conda
run which python
run python --version
run python -m pip show lerobot
run python -m pip show torch
run python -m pip show transformers
run nvidia-smi

run conda run -n "$env_name" which python
run conda run -n "$env_name" python --version
run conda run -n "$env_name" python -c   'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
run conda run -n "$env_name" python -c   'import lerobot; print(lerobot.__file__)'
run conda run -n "$env_name" python -c   'from lerobot.policies.act.modeling_act import ACTPolicy; print("ACT_IMPORT_OK")'
run conda run -n "$env_name" python -c   'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print("SMOLVLA_IMPORT_OK")'

for cli in lerobot-train lerobot-record lerobot-rollout lerobot-edit-dataset; do
  run conda run -n "$env_name" bash -lc "command -v $cli && $cli --help"
done

run conda run -n "$env_name" python -m pip check
run conda env export -n "$env_name" --from-history
run conda run -n "$env_name" python -m pip freeze

printf '\nAuthentication checks (no tokens are printed)\n'
env -u ALL_PROXY -u all_proxy hf auth whoami || true
conda run -n "$env_name" python -c   'import os,pathlib; print("WANDB_CREDENTIAL_PRESENT="+str(bool(os.getenv("WANDB_API_KEY")) or pathlib.Path.home().joinpath(".netrc").is_file()).lower())'
