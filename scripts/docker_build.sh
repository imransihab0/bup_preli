#!/usr/bin/env bash
# Build, smoke-test, and optionally push the fallback image.
#
#   ./scripts/docker_build.sh                          # build + test locally
#   ./scripts/docker_build.sh yourname/gridwise-api    # build + test + push
#
# The image contains no credentials; OPENAI_API_KEY is passed at run time.
set -euo pipefail

IMAGE="${1:-gridwise-api}"
TAG="${TAG:-1.0.0}"
REF="${IMAGE}:${TAG}"
PORT="${PORT:-8099}"

echo "==> Building ${REF}"
docker build -t "${REF}" .

echo "==> Verifying no secrets were baked into the image"
if docker run --rm --entrypoint sh "${REF}" -c 'ls -a /app' | grep -qx '.env'; then
  echo "FAIL: .env is present inside the image"; exit 1
fi
if docker run --rm --entrypoint sh "${REF}" -c 'env' | grep -qE 'sk-(proj|ant)-'; then
  echo "FAIL: a credential is baked into the image environment"; exit 1
fi
echo "    no .env, no credential-shaped env vars"

echo "==> Starting container on port ${PORT}"
CID=$(docker run -d -p "${PORT}:8000" \
  ${OPENAI_API_KEY:+-e OPENAI_API_KEY="${OPENAI_API_KEY}"} \
  "${REF}")
trap 'docker rm -f "${CID}" >/dev/null 2>&1 || true' EXIT

echo "==> Waiting for /health"
curl -fsS --retry 45 --retry-delay 1 --retry-connrefused --max-time 90 \
  "http://127.0.0.1:${PORT}/health"
echo

echo "==> Running the smoke test against the container"
./scripts/smoke_test.sh "http://127.0.0.1:${PORT}"

echo "==> Container healthcheck status"
docker inspect --format='{{.State.Health.Status}}' "${CID}" || true

if [[ "${IMAGE}" == */* ]]; then
  echo "==> Pushing ${REF}"
  docker push "${REF}"
  echo
  echo "Submit this exact reference:"
  docker inspect --format='{{index .RepoDigests 0}}' "${REF}"
else
  echo
  echo "Local build only. To publish, re-run with a registry path:"
  echo "    ./scripts/docker_build.sh <dockerhub-user>/gridwise-api"
fi
