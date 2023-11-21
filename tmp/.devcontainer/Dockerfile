# syntax=docker/dockerfile:1.4
FROM public.ecr.aws/docker/library/python:3.10.9 as devcontainer

# Versions
ARG HELM_VERSION=3.11.1
ARG KUBECTL_VERSION=1.24.0
ARG FORCE_REBUILD=4
ARG PIPX_VERSION=0.16.4
ARG POETRY_VERSION=1.4.0

ENV PATH=/opt/pipx/bin:/app/.venv/bin:$PATH \
  PIPX_BIN_DIR=/opt/pipx/bin \
  PIPX_HOME=/opt/pipx/home \
  POETRY_VIRTUALENVS_IN_PROJECT=true
RUN python -m pip install --no-cache-dir --upgrade pip "pipx==$PIPX_VERSION" && \
  pipx install "poetry==$POETRY_VERSION"

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY devcontainer-setup.sh /devcontainer-setup.sh
RUN /devcontainer-setup.sh

# from https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions
# - Docker actions must be run by the default Docker user (root)

ENTRYPOINT ["bash"]
