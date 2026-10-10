#!/bin/sh
# Production Redis: TLS-only, ACL identities, bounded memory.
#
# Runs as root only long enough to (a) copy the TLS key to a 0600 file owned by the
# redis uid, (b) write the ACL file, (c) chown the data dir, then drops to the redis
# user with setpriv. The container needs CHOWN, SETUID, SETGID and DAC_READ_SEARCH
# (docker-compose.prod.yml adds exactly those after cap_drop ALL).
set -eu

: "${REDIS_APP_PASSWORD:?REDIS_APP_PASSWORD is required}"
: "${REDIS_ADMIN_PASSWORD:?REDIS_ADMIN_PASSWORD is required}"
# SEC-2: cap memory so cached vectors can never OOM the box. volatile-lru evicts only keys that
# carry a TTL, selecting the least recently used (including query/document caches); the ingest queue has none and is kept.
REDIS_MAXMEMORY="${REDIS_MAXMEMORY:-512mb}"

app_hash="$(printf '%s' "$REDIS_APP_PASSWORD" | sha256sum | cut -d ' ' -f 1)"
admin_hash="$(printf '%s' "$REDIS_ADMIN_PASSWORD" | sha256sum | cut -d ' ' -f 1)"

umask 077
# A fresh private directory on every start: after the first start these files belong to the redis
# user, which root (without DAC_OVERRIDE) cannot overwrite when the container is merely restarted.
RUN_DIR="$(mktemp -d /run/redis-XXXXXX)"

# INF-10: the app identity gets only the commands the backend issues.
#   +client|setinfo/setname/id  redis-py handshake (CLIENT SETINFO) and connection naming
#   +eval                       rate_limit.py / rate_ceiling.py run Lua (INCR+EXPIRE, login reservations)
#   +get +set +del +incr +decr +expire +ttl   keys and counters inside those scripts and the vector caches
#   +rpush +blpop               KB ingest queue
#   +publish +subscribe ...     chat wake-up channel
{
  printf 'user default off\n'
  printf 'user app on #%s ~rate:* ~ratekey:* ~kb:* ~bot:* ~worker:* &chat:wakeup +ping +auth +hello +select +client|setinfo +client|setname +client|id +get +set +del +incr +decr +expire +ttl +eval +rpush +blpop +publish +subscribe +unsubscribe +psubscribe +punsubscribe\n' "$app_hash"
  printf 'user admin on #%s ~* &* +@all\n' "$admin_hash"
} > "$RUN_DIR/users.acl"

# The host-side key can be 0600 and owned by the deploy user: root reads it (DAC_READ_SEARCH)
# and hands the server its own private copy.
cp /run/tls/redis.key "$RUN_DIR/redis.key"

if [ "$(id -u)" = "0" ]; then
  chown redis:redis "$RUN_DIR" "$RUN_DIR/users.acl" "$RUN_DIR/redis.key"
  find /data \! -user redis -exec chown redis '{}' +
  drop="setpriv --reuid redis --regid redis --clear-groups --"
else
  drop=""
fi

# shellcheck disable=SC2086 # $drop is intentionally word-split into the setpriv command.
exec $drop redis-server \
  --port 0 \
  --tls-port 6379 \
  --tls-cert-file /run/tls/redis.crt \
  --tls-key-file "$RUN_DIR/redis.key" \
  --tls-ca-cert-file /run/tls/ca.crt \
  --tls-auth-clients no \
  --aclfile "$RUN_DIR/users.acl" \
  --appendonly yes \
  --maxmemory "$REDIS_MAXMEMORY" \
  --maxmemory-policy volatile-lru
