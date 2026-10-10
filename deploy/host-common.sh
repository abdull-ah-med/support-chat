#!/usr/bin/env bash
# Sourced by host scripts after cd to the checkout.
COMPOSE=(docker compose -f "${COMPOSE_FILE:-docker-compose.yml}" -f "${COMPOSE_PROD_FILE:-docker-compose.prod.yml}")
BACKUP_DIR="${BACKUP_DIR:-/var/backups/support-chat}"
IMAGE_ARCHIVE_DIR="${IMAGE_ARCHIVE_DIR:-/var/cache/support-chat/images}"

check_disk() {
  local path="$1" required_kb="$2" free_kb
  while [ ! -d "$path" ]; do path="$(dirname "$path")"; done
  free_kb="$(df -Pk "$path" | awk 'NR==2 {print $4}')"
  if ! [[ "$free_kb" =~ ^[0-9]+$ ]] || [ "$free_kb" -lt "$required_kb" ]; then
    echo "Insufficient free disk for backup and image archives (need ${required_kb} KiB)" >&2
    return 1
  fi
}

# A conservative uncompressed DB size estimate plus 1 GiB reserve. Read the live
# container's own credentials, never the new checkout's possibly rotated values.
backup_required_kb() {
  local size
  # shellcheck disable=SC2016 # expand credentials inside the running container only
  size="$("${COMPOSE[@]}" exec -T postgres sh -ec '
    export PGPASSWORD="$POSTGRES_PASSWORD"
    exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT pg_database_size(current_database())"
  ' 2>/dev/null)" || { echo "Cannot estimate database backup size" >&2; return 1; }
  [[ "$size" =~ ^[0-9]+$ ]] || { echo "Invalid database size response" >&2; return 1; }
  echo "$((size / 1024 + 1048576))"
}
