#!/bin/bash
set -euo pipefail

# SEC-9: host nginx reaches the API through a published loopback port, so the API sees the compose
# network's gateway as its peer and must trust X-Forwarded-For from exactly that address.
# TRUSTED_PROXY_CIDRS=auto (or unset in production) resolves it here from the container's default
# route, so the value survives the network being recreated. Any explicit value is left untouched.
# Parsed from /proc/net/route (no `ip` binary in the image): the gateway of the 0.0.0.0 route is
# stored as little-endian hex, e.g. 010012AC -> 172.18.0.1. PROC_NET_ROUTE exists for tests.
default_gateway() {
  local _iface dest gw _rest
  while read -r _iface dest gw _rest; do
    if [ "$dest" = "00000000" ] && [ "$gw" != "00000000" ] && [ "${#gw}" -eq 8 ]; then
      printf '%d.%d.%d.%d\n' "0x${gw:6:2}" "0x${gw:4:2}" "0x${gw:2:2}" "0x${gw:0:2}"
      return 0
    fi
  done < "${PROC_NET_ROUTE:-/proc/net/route}"
  return 1
}

if [ "${TRUSTED_PROXY_CIDRS-unset}" = "auto" ] \
  || { [ "${TRUSTED_PROXY_CIDRS-unset}" = "unset" ] && [ "${APP_ENV:-}" = "production" ]; }; then
  if gateway="$(default_gateway)" && [ -n "$gateway" ]; then
    export TRUSTED_PROXY_CIDRS="${gateway}/32"
    echo "TRUSTED_PROXY_CIDRS resolved to ${TRUSTED_PROXY_CIDRS}"
  else
    echo "TRUSTED_PROXY_CIDRS=auto but no default gateway was found in ${PROC_NET_ROUTE:-/proc/net/route}" >&2
    exit 1
  fi
fi

echo "Waiting for Postgres..."
i=0
pg_errors="$(mktemp)"
trap 'rm -f "$pg_errors"' EXIT
while [ "$i" -lt 60 ]; do
  if python -c "
from sqlalchemy import create_engine, text
from app.settings import get_settings
engine = create_engine(get_settings().sync_database_url, pool_pre_ping=True)
with engine.connect() as conn:
    conn.execute(text('SELECT 1'))
" 2>"$pg_errors"; then
    break
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$i" -eq 60 ]; then
  echo "Postgres readiness/settings check failed; detailed values suppressed" >&2
  echo "Postgres did not become ready in time" >&2
  exit 1
fi

rm -f "$pg_errors"
trap - EXIT

# Alembic migrations do not use Redis. Skip the wait when this container is
# only applying schema (prod Redis is TLS-only; a missing CA must not block migrate).
case " $* " in
  *" migrate.sh "*|*" alembic "*) ;;
  *)
    echo "Waiting for Redis..."
    i=0
    while [ "$i" -lt 60 ]; do
      if python -c "
import redis
from app.settings import get_settings
client = redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=1)
client.ping()
" 2>/dev/null; then
        break
      fi
      i=$((i + 1))
      sleep 1
    done
    if [ "$i" -eq 60 ]; then
      echo "Redis did not become ready in time" >&2
      exit 1
    fi
    ;;
esac

exec "$@"
