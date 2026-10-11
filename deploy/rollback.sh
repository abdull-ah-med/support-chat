#!/usr/bin/env bash
# App images only. Never retag/re-run migrate against a newer Alembic head.
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source deploy/host-common.sh
read -r -a APP_SERVICES <<< "${COMPOSE_APP_SERVICES:-backend worker frontend}"
images=()
recoverable=()
missing=0
for service in "${APP_SERVICES[@]}"; do
  case "$service" in backend|worker|frontend) ;; *) echo "Rollback accepts app services only" >&2; exit 1 ;; esac
  cid="$("${COMPOSE[@]}" ps -aq "$service" | head -n 1)"
  # Resolve from Compose if a failed `up` removed the container.
  if [ -n "$cid" ]; then
    image="$(docker inspect --format '{{.Config.Image}}' "$cid")"
  else
    image="$(python3 deploy/compose-config.py images "$service")"
  fi
  repo="$image"
  case "${image##*/}" in *:*) repo="${image%:*}" ;; esac
  if ! docker image inspect "${repo}:previous" >/dev/null 2>&1; then
    echo "WARN: No ${repo}:previous image; skipping $service" >&2
    missing=1
    continue
  fi
  images+=("$image")
  recoverable+=("$service")
done
# Restore each available image, even if another service has no previous generation.
[ "${#recoverable[@]}" -gt 0 ] || { echo "No app images available for rollback" >&2; exit 1; }
for image in "${images[@]}"; do
  repo="$image"
  case "${image##*/}" in *:*) repo="${image%:*}" ;; esac
  docker tag "${repo}:previous" "$image"
done
# --no-deps is critical: Compose must not start migrate or recreate Postgres/Redis.
"${COMPOSE[@]}" up -d --no-build --no-deps --wait --wait-timeout 180 "${recoverable[@]}"
echo "Restored app services: ${recoverable[*]}; database schema and volumes unchanged."
exit "$missing"
