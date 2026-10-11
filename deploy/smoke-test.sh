#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
hosts="$(python3 deploy/compose-config.py origins)"
origins=()
while IFS= read -r host; do origins+=("$host"); done <<< "$hosts"
# /health belongs to the staff Host in HEAD's backend/app/security/surfaces.py.
for endpoint in "https://${origins[0]}/login" "https://${origins[1]}/widget" "https://${origins[2]}/demo" "https://${origins[0]}/health"; do
  host="${endpoint#https://}"
  host="${host%%/*}"
  # TLS/SNI and the public Host pass through host nginx; no loopback backend shortcuts.
  if ! status="$(curl --fail --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --resolve "$host:443:127.0.0.1" --noproxy "*" --connect-timeout 5 --max-time 20 --retry 5 --retry-delay 2 --retry-all-errors \
    --retry-max-time 60 "$endpoint")" || [ "$status" != 200 ]; then
    echo "Public nginx smoke test failed: $endpoint" >&2
    exit 1
  fi
done
echo "Staff, widget and API health smoke tests passed."
