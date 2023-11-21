#!/bin/sh

set -e
# set -x

export DEBIAN_FRONTEND=noninteractive

# Install dependencies
apt-get update
apt-get install -y \
  bash \
  curl \
  jq \
  git \
  unzip \
  time \
  gettext \
  ca-certificates \
  gnupg \
  lsb-release \
  python3-pip \
  groff less # required by awscli

# Install git-lfs
curl -fsSL https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

# Install docker
# from https://docs.docker.com/engine/install/debian/
#shellcheck disable=SC2174
mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list >/dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# docker --version

time rm -rf /var/lib/apt/lists/*

# Install just
curl --proto '=https' -sSf https://just.systems/install.sh | bash -s -- --to /usr/bin
just --version

time rm -rf /var/lib/apt/lists/*

# Install brew
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"

# Install awscli v2
# brew install awscli
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
aws --version
