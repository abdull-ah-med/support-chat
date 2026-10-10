# Deployment, rollback and database recovery

CI builds two images with plain `docker build --pull` on the native linux/amd64 runner;
an arm64 host skips the CI build and builds on
the host. Worker and migrate reuse the backend image. Public frontend build arguments
are resolved from the host root `.env` through SSM before building (including
`FRONTEND_API_ORIGIN` → `API_ORIGIN`) and passed by name from the environment. The
backend image must import `app.main` and `app.workers.entrypoint` before upload.
Images are saved with zstd, checksummed, and uploaded for one day. The manifest also
records the frontend build arguments; the host re-resolves them from the new checkout
and falls back to a host build when they differ, or when the loaded image architecture
differs from the host. The SSM sender embeds the run ID and the build job's attempt
(not the deploy job's, so re-running only the deploy job still finds the artifact)
only when upload succeeded. GitHub token transport and authenticated clone/fetch are
unchanged.
No registry, GitHub settings, or AWS configuration changes are needed.

Public build-env lookup runs on a separate runner. Build/upload actions run in
a separate job with no AWS credentials or OIDC permission, including their post
cleanup hooks. Fresh AWS credentials are acquired immediately before deployment. Any optional
lookup/build/package/upload failure skips the artifact and deploys via host build.
Every third-party action in both workflows is pinned to a full commit SHA.

The host requires its existing Docker/Compose, Python 3.8+, curl, nginx, systemctl,
flock and GNU timeout. Do not copy deploy scripts onto the host by hand. Deploy
from a reviewed commit through the pinned SSM path so scripts and images agree.
If zstd is missing, artifacts expire, upload/download fails, integrity checks fail,
or `docker load` fails, the script logs a warning and builds on the host. No artifact
failure silently substitutes an image from another commit/run/attempt. The last two
successful compressed image archives remain in `/var/cache/support-chat/images`.
Each archive has its manifest and checksum beside it. Identity checks compare
RootFS layer diff IDs, which remain stable across classic and containerd image stores.
Downloads use a 600-second per-attempt limit and fail if throughput stays below
1 MiB/s for 60 seconds. A whole-transfer/load timeout also bounds retries/unpacking.

The host lock waits at most 1,800 seconds before checkout. Optional artifact/load
work gets at most 900 seconds (normally about 800); host fallback builds get at most
1,200 seconds. Both are shortened by elapsed lock/preflight/transfer time to reserve
1,200 seconds for backup/migration/rollout inside the 3,300-second SSM executionTimeout.
Worst case after the build: Redis wait 180 + app wait 180 + smoke 4 x 80 (60-second
retry window plus one 20-second attempt) + rollback wait 180 = 860 seconds, leaving
340 seconds for backup, role provisioning and migration.
The sender polls for 3,500 seconds. The deploy job allows 65 minutes, and the
separate optional build job allows 60 minutes. Fresh default one-hour AWS credentials cover sending and polling.
A slow build times out before backup or migration. Highly contended deploys may
need a retry after the previous deployment finishes.

## First deployment on the existing host

1. Prefer the exact `support-chat_postgres_data` volume (override with
   `POSTGRES_VOLUME_NAME`), retaining the legacy `postgres_data` filter fallback.
   Preflight checks required root `.env` variables, `APP_ENV=production`, an explicit backend proxy gateway
   `/32`, Redis memory, and free disk for the backup (database size + 1 GiB, on the backup
   directory and the Docker data root; failing aborts). The extra image reserve only warns:
   the artifact loader enforces its own reserve and falls back to a host build. A proposed Redis limit below current usage
   must be resolved before deploying. It also renders nginx and validates an isolated
   copy of the full host config (`nginx -t -c` on the staged copy) before backup/migration;
   dangling symlinks are skipped in the copy.
2. Tag healthy running backend/worker/frontend images `:previous`, then load CI
   images or build locally. No running containers change during this stage.
3. Save a custom-format `pg_dump -Fc` using the running Postgres container's own
   credentials. Backups are mode 600 under `/var/backups/support-chat` (directory 700);
   retain the last five. A failed/empty dump aborts before provisioning/migration.
4. Run a disposable `db-provision` container with `--no-deps`, recreate Redis for
   its new digest/limits/ACL, and run one disposable migrate container with
   `--no-deps`. PostgreSQL is **never recreated or restarted**: its base and overlay
   service definitions match HEAD, and every `run`/`up` explicitly excludes dependencies.
5. Recreate backend, worker and frontend using `up --no-build --no-deps --wait`.
   The old app serves until this stage; the Redis restart briefly interrupts Redis
   connectivity. A migration failure leaves app images running and exits nonzero;
   Redis may already have restarted and partially applied schema is not reverted.
6. Atomically switch the nginx candidate already validated by preflight and reload.
   Check staff `/login`, widget `/widget`, marketing `/demo`, and staff `/health` with
   bounded retries and `--resolve host:443:127.0.0.1` (normal TLS and SNI verification).
   `/health` checks both DB and Redis and is restricted to the staff Host by the backend.

The nginx template includes HTTPS and the normal `/etc/letsencrypt/live/<lineage>/`
`fullchain.pem` / `privkey.pem` paths. Existing certbot lineage names are reused;
bootstrap's combined certificate defaults to the marketing hostname. Missing TLS
files, rendering errors, or invalid candidates cause a warning and leave nginx unchanged.
After switching, `nginx -t` on the installed config runs before the reload; a test or
reload failure restores the previous nginx file and reloads it.
Nginx update failure does not roll back healthy app images. A separate smoke/startup
failure after migration rolls back app images; it leaves a validated nginx config active.
The saved config is `/etc/nginx/conf.d/support-chat.conf.previous`.

## App rollback (no schema downgrade)

In the EC2 SSM shell:

```bash
cd /home/ssm-user/support-chat
sudo bash deploy/rollback.sh
```

This checks each `:previous` image, reports missing images, and restores every
available backend/worker/frontend image and runs `up -d --no-build --no-deps --wait`. It never retags or starts migrate,
never traverses dependencies, and never modifies volumes. Automatic rollback
uses the same script and exits nonzero so Actions reports failure. Partial recovery
also returns nonzero after restarting all recoverable services.

**Schema migrations are not rolled back.** The older app must remain compatible
with the migrated schema; otherwise fix forward or deliberately restore a backup.
Only one image rollback generation exists. The checked-out repo and Compose
configuration remain at the deployed commit. Do not use plain `compose up` to
rollback: it can rerun migrations. App rollback leaves nginx unchanged.

## Manual Redis recovery

If a Redis configuration/image change caused a failure, select the previous reviewed
commit in the existing repo checkout and recover Redis independently. Set
`PREVIOUS_COMMIT` to that commit before running this block from the repo root:

```bash
git -c safe.directory="$PWD" checkout --detach "${PREVIOUS_COMMIT:?Select the previous reviewed commit}"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --no-build --wait redis
```

Do not restart PostgreSQL or remove volumes. App/schema recovery remains separate.

## Deliberate DB restore

Restoring a backup replaces database contents and loses writes made after that
backup. This is a separate operator recovery action, never part of automatic
rollback. Choose and validate the backup first; retain an additional current dump
when possible. The running Postgres container and its named volume remain intact.
Run these commands in a root EC2 SSM shell, selecting an actual backup filename:

```bash
set -euo pipefail
cd /home/ssm-user/support-chat
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
BACKUP=/var/backups/support-chat/REPLACE_WITH_SELECTED_BACKUP.dump
[ -s "$BACKUP" ]
[ "$(head -c 5 "$BACKUP")" = PGDMP ]
PG_CONTAINER="$("${COMPOSE[@]}" ps -q postgres)"
[ -n "$PG_CONTAINER" ]
# Validate that pg_restore can read the archive before interrupting traffic.
docker exec -i "$PG_CONTAINER" pg_restore --list < "$BACKUP" >/dev/null
"${COMPOSE[@]}" stop backend worker frontend
# Close remaining connections to the app DB; credentials stay inside the container.
docker exec "$PG_CONTAINER" sh -ec '
  export PGPASSWORD="$POSTGRES_PASSWORD"
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()"
'
# --create restores the archived database name, including its Alembic version.
# --clean drops that database before creating it; this replaces DATA, never volumes.
docker exec -i "$PG_CONTAINER" sh -ec '
  export PGPASSWORD="$POSTGRES_PASSWORD"
  exec pg_restore -U "$POSTGRES_USER" -d postgres --clean --if-exists --create --exit-on-error
' < "$BACKUP"
# Run the app generation compatible with the restored schema. Do not run migrate.
bash deploy/rollback.sh
bash deploy/smoke-test.sh
```

`pg_restore --create` uses the database name in the archive. Backups here are taken
from the container's `POSTGRES_DB`; confirm it is the intended DB before recovery.
Cluster-level roles are not in `pg_dump`; the existing provisioned roles must still
exist. On any restore error leave apps stopped, preserve the archive, and investigate.

## Owner checks before pushing (read-only)

In the existing EC2 SSM shell, from the repo root:

```bash
cd /home/ssm-user/support-chat
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
# Presence only: no secret values are printed. Check the variables the new overlay requires.
python3 - <<'PY'
from pathlib import Path
required = '''APP_ENV APP_DB_USER APP_DB_PASSWORD MIGRATION_DB_USER MIGRATION_DB_PASSWORD
POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB REDIS_APP_PASSWORD REDIS_ADMIN_PASSWORD
JWT_SECRET WIDGET_TOKEN_SECRET RATE_KEY_SECRET WIDGET_CSP_SERVICE_SECRET
NEXT_PUBLIC_WIDGET_ORIGIN NEXT_PUBLIC_STAFF_APP_ORIGIN NEXT_PUBLIC_MARKETING_HOST_ORIGIN'''.split()
values = {}
for line in Path('.env').read_text().splitlines():
    key, sep, value = line.partition('=')
    if sep:
        values[key.strip()] = value.strip().strip('\"\'')
missing = [key for key in required if not values.get(key)]
print('Missing variable names: ' + ', '.join(missing) if missing else 'Required root variables present')
print('APP_ENV is production: ' + str(values.get('APP_ENV') == 'production'))
PY
"${COMPOSE[@]}" config -q
REDIS_CONTAINER="$("${COMPOSE[@]}" ps -q redis)"
docker exec "$REDIS_CONTAINER" sh -ec '
  export REDISCLI_AUTH="$REDIS_ADMIN_PASSWORD"
  exec redis-cli --tls --cacert /run/tls/ca.crt -h redis --user admin --no-auth-warning INFO memory
'
# used_memory should be <= 80% of the smaller proposed Redis limits:
# defaults: maxmemory 512 MiB, container 1 GiB -> threshold 429496729 bytes.
df -Pk /var/lib/docker /var/backups /var/cache
# Preflight reserves DB size + 1 GiB, plus 12 GiB for images/build/extraction.
"${COMPOSE[@]}" exec -T postgres sh -ec '
  export PGPASSWORD="$POSTGRES_PASSWORD"
  exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT pg_database_size(current_database())"
'
docker network inspect support-chat_default --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}'
# Compare the gateway with backend/.env.prod TRUSTED_PROXY_CIDRS (currently 172.18.0.1/32).
sed -n '/^TRUSTED_PROXY_CIDRS=/p' backend/.env.prod
# Discover the actual certificate lineage paths without printing key material.
sudo nginx -T 2>/dev/null | sed -n '/ssl_certificate[ _]/p'
sudo find /etc/letsencrypt/live -maxdepth 2 \( -name fullchain.pem -o -name privkey.pem \) -exec test -e '{}' \; -print
command -v python3 curl zstd
```

If the new scripts are already present, `sudo bash deploy/preflight.sh` performs
these gates without creating containers or writing backups (nginx staging uses
a temporary directory). Inspect `nginx -t`
and the active certificate paths if TLS configuration differs from bootstrap.
Postgres hardening, including `stop_grace_period: 60s`, is deferred to a separate deploy.
