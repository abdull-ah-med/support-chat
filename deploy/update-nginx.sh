#!/usr/bin/env bash
# Stage/validate during preflight, then switch only that prepared candidate after app startup.
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
NGINX_ROOT="${NGINX_ROOT:-/etc/nginx}"
dest="${NGINX_CONF_DEST:-$NGINX_ROOT/conf.d/support-chat.conf}"
previous="${dest}.previous"
restore() {
  [ -f "$previous" ] || { echo "Previous nginx config unavailable" >&2; return 1; }
  local copy
  copy="$(mktemp "${dest}.restore-XXXXXX")"
  cp -p "$previous" "$copy"
  mv -f "$copy" "$dest"
  nginx -t && systemctl reload nginx
}
if [ "${1:-}" = --restore ]; then restore; exit; fi
# Manual invocations prepare in a temporary directory. The deploy supplies a
# private directory shared by preflight and the later switch, cleaned at exit.
owned_work=""
if [ -z "${NGINX_CANDIDATE_DIR:-}" ]; then
  owned_work="$(mktemp -d)"
  NGINX_CANDIDATE_DIR="$owned_work"
fi
candidate=""
trap '[ -z "$owned_work" ] || rm -rf "$owned_work"; [ -z "$candidate" ] || rm -f "$candidate"' EXIT
prepare() {
  rm -f "$NGINX_CANDIDATE_DIR/ready.conf"
  [ -f "$dest" ] || { echo "WARN: No active nginx config; skipping replacement" >&2; return 0; }
  case "$dest" in "$NGINX_ROOT"/*) ;; *) echo "nginx destination outside NGINX_ROOT" >&2; return 1 ;; esac
  local status=0
  python3 deploy/render-nginx.py > "$NGINX_CANDIDATE_DIR/candidate.conf" || status=$?
  [ "$status" -ne 3 ] || return 0
  [ "$status" -eq 0 ] || return "$status"
  # Expand valid links into isolated files. nginx itself fails on an include glob that matches
  # a dangling link, so staging skips those; an installed config broken that way is caught by
  # the post-switch `nginx -t` below, which restores the previous file.
  # Every absolute path under NGINX_ROOT is rewritten into the staged tree, and relative
  # includes resolve against the directory of the -c file, so `-c` alone tests the candidate
  # as it will be installed. No `-p`: the compiled-in prefix keeps relative `load_module
  # modules/*.so` working on Debian/Ubuntu.
  python3 - "$NGINX_ROOT" "$NGINX_CANDIDATE_DIR/nginx" "$dest" "$NGINX_CANDIDATE_DIR/candidate.conf" <<'PY'
import shutil, sys
from pathlib import Path
root, stage, dest, candidate = map(Path, sys.argv[1:])
def broken_links(directory, names):
    # Check relative targets against their source directory, not the process cwd.
    return [name for name in names
            if (Path(directory) / name).is_symlink() and not (Path(directory) / name).exists()]
shutil.copytree(root, stage, symlinks=False, ignore=broken_links)
shutil.copyfile(candidate, stage / dest.relative_to(root))
for path in stage.rglob("*"):
    if path.is_file():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        path.write_text(text.replace(str(root) + "/", str(stage) + "/"))
PY
  nginx -t -c "$NGINX_CANDIDATE_DIR/nginx/nginx.conf"
  mv "$NGINX_CANDIDATE_DIR/candidate.conf" "$NGINX_CANDIDATE_DIR/ready.conf"
}
# Called directly (not under `if`): bash must stop prepare() on validation/copy errors.
if [ "${1:-}" = --prepare ]; then prepare; exit; fi
if [ -n "$owned_work" ]; then prepare; fi
[ -f "$NGINX_CANDIDATE_DIR/ready.conf" ] || { echo "WARN: No validated nginx candidate; keeping current config" >&2; exit 0; }
# Same filesystem as destination, so rename is atomic. Previous stays outside *.conf glob.
cp -p "$dest" "$previous"
candidate="$(mktemp "${dest}.candidate-XXXXXX")"
cp "$NGINX_CANDIDATE_DIR/ready.conf" "$candidate"
chmod 644 "$candidate"
mv -f "$candidate" "$dest"
candidate=""
# Test the installed config itself before reloading; restore the previous file on any failure.
if ! nginx -t || ! systemctl reload nginx; then
  echo "WARN: nginx test or reload failed; restoring previous config" >&2
  restore || echo "ERROR: nginx restore/reload needs operator attention" >&2
  exit 1
fi
[ -z "${NGINX_SWITCH_MARKER:-}" ] || touch "$NGINX_SWITCH_MARKER"
echo "Prepared nginx config replaced and reloaded."
