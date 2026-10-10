#!/bin/sh
set -eu

TLS_DIR="${REDIS_TLS_DIR:-./deploy/redis-tls}"

if [ -f "$TLS_DIR/ca.crt" ] && [ -f "$TLS_DIR/redis.crt" ] && [ -f "$TLS_DIR/redis.key" ]; then
  # The Redis entrypoint copies the key as root, so the host copy no longer needs to be world-readable.
  chmod 600 "$TLS_DIR/redis.key"
  if [ -e "$TLS_DIR/ca.key" ]; then
    echo "WARNING: $TLS_DIR/ca.key is still on this host. Move it to offline storage: anyone holding it can mint certs the backend trusts." >&2
  fi
  echo "Redis TLS material already exists in $TLS_DIR"
  exit 0
fi

if [ -e "$TLS_DIR/ca.crt" ] || [ -e "$TLS_DIR/redis.crt" ] || [ -e "$TLS_DIR/redis.key" ]; then
  echo "Redis TLS material is incomplete in $TLS_DIR; inspect it before retrying." >&2
  exit 1
fi

umask 077
mkdir -p "$TLS_DIR"

openssl req -x509 -newkey rsa:3072 -sha256 -nodes \
  -keyout "$TLS_DIR/ca.key" \
  -out "$TLS_DIR/ca.crt" \
  -days 3650 \
  -subj "/CN=SupportChat Redis CA"

openssl req -new -newkey rsa:3072 -sha256 -nodes \
  -keyout "$TLS_DIR/redis.key" \
  -out "$TLS_DIR/redis.csr" \
  -subj "/CN=redis" \
  -addext "subjectAltName=DNS:redis"

openssl x509 -req -sha256 \
  -in "$TLS_DIR/redis.csr" \
  -CA "$TLS_DIR/ca.crt" \
  -CAkey "$TLS_DIR/ca.key" \
  -CAcreateserial \
  -copy_extensions copy \
  -out "$TLS_DIR/redis.crt" \
  -days 825

# The CA key has done its only job (signing this one server cert). Do not leave it on the host.
# Rotating the certificate means regenerating the whole directory, which this script supports.
rm -f "$TLS_DIR/ca.key" "$TLS_DIR/ca.srl" "$TLS_DIR/redis.csr"

# The directory is mode 700 from umask, so only the deployment owner can read it on the host.
# The Redis container's entrypoint runs as root, copies redis.key to a 0600 file owned by the
# redis uid, and drops privileges; the key itself therefore stays 0600 here. Certificates are public.
chmod 700 "$TLS_DIR"
chmod 644 "$TLS_DIR/ca.crt" "$TLS_DIR/redis.crt"
chmod 600 "$TLS_DIR/redis.key"
echo "Generated Redis TLS material in $TLS_DIR"
