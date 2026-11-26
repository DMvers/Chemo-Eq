#!/usr/bin/env bash
# Install for the LLM search: AlmaLinux 9 or 10 (x86_64) with one NVIDIA A100 80 GB. Two passes, with a reboot between:
#   sudo bash llm/scripts/install.sh      as root: system packages and the NVIDIA driver; then reboot
#   bash llm/scripts/install.sh           as your user: Python, vLLM 0.31.0 (~/venvs/vllm), llm/.venv, and the
#                                         Qwen3.8-27B weights (~55 GB, in $HF_HOME, default ~/models)
# Rerunning is safe: what is already there is skipped.
set -euo pipefail
VLLM_VERSION=0.31.0; UV_VERSION=0.12.23
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
export HF_HOME="${HF_HOME:-$HOME/models}" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm}"
here="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $EUID -eq 0 ]]; then
  dnf -y install dnf-plugins-core epel-release
  dnf config-manager --set-enabled crb
  dnf -y install git curl gcc gcc-c++ make kernel-devel-matched kernel-headers pciutils python3-pip
  lspci | grep -qi nvidia || { echo "no NVIDIA GPU visible"; exit 1; }
  dnf -y install almalinux-release-nvidia-driver
  dnf -y install nvidia-open-kmod nvidia-driver nvidia-driver-cuda
  echo "done; reboot, then as your user: bash $here/scripts/install.sh"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: run the root pass and reboot first"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || python3 -m pip install --user "uv==$UV_VERSION"
uv python install 3.12
if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  uv venv "$VLLM_ENV" --python 3.12
  uv pip install --python "$VLLM_ENV/bin/python" "vllm==$VLLM_VERSION" --torch-backend=auto
fi
"$VLLM_ENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'"
[[ -d "$here/.venv" ]] || uv venv "$here/.venv" --python 3.12
uv pip install --python "$here/.venv/bin/python" -r "$here/requirements.txt"
"$VLLM_ENV/bin/hf" download "$MODEL"
echo "done; next: bash $here/scripts/prelaunch.sh"
