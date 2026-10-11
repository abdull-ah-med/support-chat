#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source deploy/host-common.sh
check_disk "$BACKUP_DIR" "$(backup_required_kb)"
umask 077
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
backup="$(mktemp "$BACKUP_DIR/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.dump.partial")"
trap 'rm -f "$backup"' EXIT
# shellcheck disable=SC2016 # expand credentials inside the running container only
if ! "${COMPOSE[@]}" exec -T postgres sh -ec '
  export PGPASSWORD="$POSTGRES_PASSWORD"
  exec pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"
' > "$backup" || [ ! -s "$backup" ]; then
  echo "Database backup failed; aborting before migration. Old app remains running." >&2
  exit 1
fi
# pg_dump's exit status and a nonempty custom-format header must both be good.
[ "$(head -c 5 "$backup")" = PGDMP ] || { echo "Invalid database backup" >&2; exit 1; }
mv "$backup" "${backup%.partial}"
python3 - "$BACKUP_DIR" <<'PY'
import sys
from pathlib import Path
files = sorted(Path(sys.argv[1]).glob("*.dump"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
for path in files[5:]:
    path.unlink()
PY
echo "Database backup saved outside checkout (last five retained)."
