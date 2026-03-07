#!/usr/bin/env bash

compute_codex_smoke_image_hash() {
  cat .github/docker/codex-smoke.Dockerfile lib/crewai/pyproject.toml \
    | shasum -a 256 \
    | awk '{print substr($1, 1, 16)}'
}

codex_smoke_image_tag() {
  local image_repo image_hash

  image_repo="${IMAGE_REPO:-crewai-codex-smoke-base}"
  image_hash="${IMAGE_HASH:-$(compute_codex_smoke_image_hash)}"

  printf '%s:%s\n' "${image_repo}" "${image_hash}"
}

ensure_codex_smoke_image() {
  local image_tag

  image_tag="${1:-$(codex_smoke_image_tag)}"

  if docker image inspect "${image_tag}" >/dev/null 2>&1; then
    echo "docker_image_cache=hit"
  else
    echo "docker_image_cache=miss"
    DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
      --tag "${image_tag}" \
      --file .github/docker/codex-smoke.Dockerfile \
      .
  fi

  echo "docker_image_tag=${image_tag}"
}
