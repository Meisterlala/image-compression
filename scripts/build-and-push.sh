#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-latest}"
IMAGE="${IMAGE:-registry.meisterlala.dev/image-compression:${TAG}}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER="${BUILDER:-image-compression-multiarch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for command in docker npm git rustup wasm-pack; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "${command}" >&2
    exit 1
  fi
done
docker buildx version >/dev/null
docker info >/dev/null

# The browser/WASM assets are architecture-independent. Build them once on
# the host; Docker selects the matching nginx base for each target platform.
npm ci
npm run models
npm run build

if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER}" --driver docker-container >/dev/null
fi
docker buildx inspect "${BUILDER}" --bootstrap >/dev/null

# No binfmt/QEMU setup is needed: the runtime Dockerfile has no RUN steps.
docker buildx build \
  --builder "${BUILDER}" \
  --platform "${PLATFORMS}" \
  --tag "${IMAGE}" \
  --push \
  "${REPO_ROOT}"

printf 'Pushed %s for %s\n' "${IMAGE}" "${PLATFORMS}"
