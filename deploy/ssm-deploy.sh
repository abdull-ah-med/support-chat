#!/usr/bin/env bash
# Runs on the EC2 instance via `aws ssm send-command` (AWS-RunShellScript).
# SSM executes as root — use safe.directory instead of git config --global.
#
# Required env (exported by the workflow / github-actions-ssm-deploy.sh):
#   DEPLOY_SHA, GITHUB_TOKEN, REPO_DIR, GITHUB_REPO, BRANCH
# Optional:
#   COMPOSE_FILE, COMPOSE_PROD_FILE, COMPOSE_APP_SERVICES, POSTGRES_VOLUME_FILTER
#
# Rollback: every deploy first tags the images that are running (and healthy) as `:previous`.
# See deploy/ROLLBACK.md and deploy/rollback.sh.
set -euo pipefail

deploy_started=$SECONDS
exec 9>"${DEPLOY_LOCK_FILE:-/var/lock/support-chat-deploy.lock}"
flock -w 1800 9 || { echo "another deploy is running" >&2; exit 1; }

REPO_DIR="${REPO_DIR:?REPO_DIR required}"
GITHUB_REPO="${GITHUB_REPO:?GITHUB_REPO required}"
BRANCH="${BRANCH:?BRANCH required}"
SHA="${DEPLOY_SHA:?DEPLOY_SHA required}"
GITHUB_TOKEN="${GITHUB_TOKEN:?GITHUB_TOKEN required}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
COMPOSE_PROD_FILE="${COMPOSE_PROD_FILE:-docker-compose.prod.yml}"
COMPOSE_APP_SERVICES="${COMPOSE_APP_SERVICES:-backend worker frontend}"
POSTGRES_VOLUME_FILTER="${POSTGRES_VOLUME_FILTER:-postgres_data}"

AUTH_FETCH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"

# First deploy: Actions token clones the private repo so operators need not.
if [ ! -d "${REPO_DIR}/.git" ]; then
  if [ -e "${REPO_DIR}" ]; then
    echo "REPO_DIR exists without .git — refusing to overwrite: ${REPO_DIR}" >&2
    exit 1
  fi
  echo "REPO_DIR missing — cloning ${GITHUB_REPO} (${BRANCH}) into ${REPO_DIR}"
  mkdir -p "$(dirname "${REPO_DIR}")"
  git -c "safe.directory=${REPO_DIR}" clone --branch "${BRANCH}" "${AUTH_FETCH_URL}" "${REPO_DIR}"
fi

GIT=(git -c "safe.directory=${REPO_DIR}")
cd "$REPO_DIR"

"${GIT[@]}" fetch "$AUTH_FETCH_URL" "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
if ! "${GIT[@]}" merge-base --is-ancestor "$SHA" "origin/$BRANCH"; then
  echo "DEPLOY_SHA $SHA is not an ancestor of origin/$BRANCH" >&2
  exit 1
fi
"${GIT[@]}" checkout --detach "$SHA"

COMPOSE=(docker compose -f "$COMPOSE_FILE" -f "$COMPOSE_PROD_FILE")

# Data safety: never wipe Postgres/Redis. Rebuild app containers only.
if ! docker volume inspect "${POSTGRES_VOLUME_NAME:-support-chat_postgres_data}" >/dev/null 2>&1 \
  && ! docker volume ls -q --filter "name=${POSTGRES_VOLUME_FILTER}" | grep -q .; then
  echo "${POSTGRES_VOLUME_FILTER} volume not found — aborting (refusing fresh database)" >&2
  exit 1
fi

# Validate before building, loading images, provisioning roles or recreating services.
NGINX_CANDIDATE_DIR="$(mktemp -d)"
export NGINX_CANDIDATE_DIR
trap 'rm -rf "$NGINX_CANDIDATE_DIR"' EXIT
bash deploy/preflight.sh
read -r -a APP_SERVICES <<< "$COMPOSE_APP_SERVICES"
for service in "${APP_SERVICES[@]}"; do
  case "$service" in backend|worker|frontend) ;; *) echo "Only app services may be deployed" >&2; exit 1 ;; esac
done
BUILT_SERVICES=("${APP_SERVICES[@]}" migrate)

# Snapshot only app images. The migrate image must always remain at the new schema head.
for service in "${APP_SERVICES[@]}"; do
  cid="$("${COMPOSE[@]}" ps -aq "$service" 2>/dev/null | head -n 1 || true)"
  [ -n "$cid" ] || continue
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid")"
  image_ref="$(docker inspect --format '{{.Config.Image}}' "$cid")"
  image_id="$(docker inspect --format '{{.Image}}' "$cid")"
  repo="$image_ref"
  case "${image_ref##*/}" in *:*) repo="${image_ref%:*}" ;; esac
  case "$health" in
    healthy|running)
      docker tag "$image_id" "${repo}:previous"
      echo "Tagged ${repo}:previous -> ${image_id}"
      ;;
    *) echo "WARN: $service is $health; keeping earlier :previous image" >&2 ;;
  esac
done

# CI injects these non-secret coordinates into the pinned script before the
# unchanged SSM sender embeds it. An unavailable artifact falls back to host builds.
CI_ARTIFACT_RUN_ID="${CI_ARTIFACT_RUN_ID:-}"
CI_ARTIFACT_RUN_ATTEMPT="${CI_ARTIFACT_RUN_ATTEMPT:-1}"
export CI_ARTIFACT_RUN_ID CI_ARTIFACT_RUN_ATTEMPT
# Reserve 20 minutes for backup/migration/rollout. Include lock wait in the
# 55-minute SSM deadline; contention shortens optional artifact/build attempts.
# Worst case after this point: redis wait 180 + app up wait 180 + smoke 4 x 80 (60 s retry
# window plus one 20 s attempt) + rollback up wait 180 = 860 s, leaving 340 s for the
# backup, role provisioning and migration inside the 1200 s reserve.
remaining=$((3300 - (SECONDS - deploy_started) - 1200))
[ "$remaining" -gt 0 ] || { echo "Deploy time budget exhausted after lock wait" >&2; exit 1; }
artifact_budget=$((remaining / 3))
[ "$artifact_budget" -le 900 ] || artifact_budget=900
[ "$artifact_budget" -ge 1 ] || artifact_budget=1
if ! timeout --kill-after=10 "$artifact_budget" bash deploy/load-ci-images.sh; then
  echo "WARN: CI artifact unavailable or invalid; building on host" >&2
  build_budget=$((3300 - (SECONDS - deploy_started) - 1200))
  [ "$build_budget" -le 1200 ] || build_budget=1200
  [ "$build_budget" -gt 0 ] || { echo "No time left for host build" >&2; exit 1; }
  timeout --kill-after=10 "$build_budget" "${COMPOSE[@]}" build "${BUILT_SERVICES[@]}"
fi

# Backup failure exits here; the running apps and dependencies are still untouched.
bash deploy/backup-db.sh

# Provision roles without traversing the postgres dependency. Its definition is
# identical to HEAD and it must not be recreated by this first hardening deploy.
"${COMPOSE[@]}" run --rm -T --no-deps db-provision
# Redis has new limits/ACLs; apply them explicitly without starting app dependencies.
"${COMPOSE[@]}" up -d --no-build --no-deps --wait --wait-timeout 180 redis
"${COMPOSE[@]}" run --rm -T --no-deps migrate

# After a successful migration, app startup/smoke failure restores app images.
# Never downgrade the schema or execute the previous migrate image.
on_failure() {
  local status=$?
  trap - ERR
  bash deploy/rollback.sh || echo "ERROR: app rollback needs operator attention" >&2
  exit "$status"
}
trap on_failure ERR
# --no-deps prevents a second migration or any Postgres recreation.
"${COMPOSE[@]}" up -d --no-build --no-deps --wait --wait-timeout 180 "${APP_SERVICES[@]}"
if ! bash deploy/update-nginx.sh; then
  echo "WARN: nginx update failed; keeping/restoring current config; app images remain deployed" >&2
fi
bash deploy/smoke-test.sh
trap - ERR

# Keep :previous tags and all volumes. Only dangling layers and old build cache go.
docker image prune -f || true
docker builder prune -af --filter "until=48h" || true
