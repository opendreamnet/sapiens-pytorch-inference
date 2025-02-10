#!/usr/bin/env bash

export NONINTERACTIVE=1
export DEBIAN_FRONTEND=noninteractive

#
# Packages Update
#
apt-get update

#
# Recommended
#
apt-get install -y \
  wget \
  curl \
  libgl1 \
  libglib2.0-0

#
# Python - UV
#
export UV_INSTALL_DIR="/usr/local/bin"
curl -LsSf https://astral.sh/uv/0.5.24/install.sh | sh
#echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc