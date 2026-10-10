#!/usr/bin/env bash
# CI: backend is reused by worker/migrate. Upload this directory as one artifact.
set -euo pipefail
: "${GITHUB_SHA:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}"
out="${IMAGE_OUTPUT_DIR:-${RUNNER_TEMP:?}/supportchat-images}"
mkdir -p "$out"
backend="supportchat-ci-backend:${GITHUB_SHA}"
frontend="supportchat-ci-frontend:${GITHUB_SHA}"
# Bypass the network wait entrypoint, exercising the actual shipped import graph.
docker run --rm --network none --entrypoint python \
  -e APP_ENV=test -e DATABASE_URL=postgresql+asyncpg://chat:chat@postgres:5432/chat_test \
  -e REDIS_URL=redis://redis:6379/15 \
  -e JWT_SECRET=tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt \
  -e WIDGET_TOKEN_SECRET=wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww \
  -e RATE_KEY_SECRET=rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr \
  "$backend" -c "import app.main, app.workers.entrypoint"
# Any packaging/import failure marks this optional step failed, suppressing upload.
if ! docker save "$backend" "$frontend" | zstd -T0 -3 -o "$out/images.tar.zst"; then
  echo "WARN: image packaging failed; deployment will build on host" >&2
  rm -f "$out/images.tar.zst"
  exit 1
fi
(cd "$out" && sha256sum images.tar.zst > images.tar.zst.sha256)
python3 - "$out/manifest.json" "$backend" "$frontend" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
images = {name: {"ref": ref, "layers": json.loads(subprocess.check_output(
    ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", ref], text=True))}
    for name, ref in zip(("backend", "frontend"), sys.argv[2:])}
# The frontend bakes these public values in at build time; the host re-resolves them
# from the new checkout and refuses an artifact built with different ones.
build_arg_keys = ("API_ORIGIN", "NEXT_PUBLIC_API_ORIGIN", "NEXT_PUBLIC_WIDGET_ORIGIN",
    "NEXT_PUBLIC_STAFF_APP_ORIGIN", "NEXT_PUBLIC_MARKETING_HOST_ORIGIN",
    "NEXT_PUBLIC_DEMO_SITE_KEY", "NEXT_PUBLIC_DEMO_PUBLIC_KEY")
Path(sys.argv[1]).write_text(json.dumps({"sha": os.environ["GITHUB_SHA"],
    "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    "build_args": {key: os.environ[key] for key in build_arg_keys},
    "images": images}))
PY
