#!/usr/bin/env bash
# Configuration/resource checks and isolated nginx staging; safe to run manually on the host. Never print Compose/env secrets.
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source deploy/host-common.sh
python3 deploy/compose-config.py validate
python3 deploy/compose-config.py origins >/dev/null
backup_kb="$(backup_required_kb)"
# Hard requirement: room for the DB backup (size estimate + 1 GiB margin). A failed backup
# aborts the deploy before any change, so insufficient space here must stop it.
check_disk "$BACKUP_DIR" "$backup_kb"
check_disk "${DOCKER_DATA_ROOT:-/var/lib/docker}" "$backup_kb"
# Advisory only: extra room for archives, extraction, loaded images and a host-build fallback.
# load-ci-images.sh enforces its own reserve and falls back to a host build when short.
image_kb="${DEPLOY_IMAGE_RESERVE_KB:-12582912}"
[[ "$image_kb" =~ ^[0-9]+$ ]] || { echo "Invalid DEPLOY_IMAGE_RESERVE_KB" >&2; exit 1; }
for image_path in "$IMAGE_ARCHIVE_DIR" "${DOCKER_DATA_ROOT:-/var/lib/docker}"; do
  check_disk "$image_path" "$((backup_kb + image_kb))" 2>/dev/null ||
    echo "WARN: low free disk on ${image_path} for CI images/host build; deploy may fall back or fail" >&2
done

redis_id="$("${COMPOSE[@]}" ps -q --status running redis)"
if [ -n "$redis_id" ]; then
  # Password stays inside the container; INFO is an admin-only command under our ACL.
  info="$(docker exec "$redis_id" sh -ec '
    export REDISCLI_AUTH="$REDIS_ADMIN_PASSWORD"
    exec redis-cli --tls --cacert /run/tls/ca.crt -h redis --user admin --no-auth-warning INFO memory
  ')" || { echo "Cannot read Redis memory; check TLS/admin credentials" >&2; exit 1; }
  used="$(printf '%s\n' "$info" | tr -d '\r' | awk -F: '$1=="used_memory" {print $2}')"
  limit="$(python3 deploy/compose-config.py redis-limit)"
  if ! [[ "$used" =~ ^[0-9]+$ ]]; then
    echo "Cannot parse Redis used_memory" >&2; exit 1
  fi
  if [ "$((used * 100))" -gt "$((limit * 80))" ]; then
    echo "Redis used_memory exceeds 80% of the proposed maxmemory/container limit; aborting" >&2
    exit 1
  fi
fi
if ! bash deploy/update-nginx.sh --prepare; then
  echo "WARN: nginx candidate render/validation failed; keeping current config" >&2
fi
echo "Preflight passed (configuration, Redis memory, disk; nginx candidate checked)."
