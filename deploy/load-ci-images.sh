#!/usr/bin/env bash
# Host: bounded, verified artifact transfer. Any failure is handled by host-build fallback.
set -euo pipefail
cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source deploy/host-common.sh
: "${CI_ARTIFACT_RUN_ID:?No pinned CI artifact run; use host build}"
[[ "$CI_ARTIFACT_RUN_ID" =~ ^[0-9]+$ ]] || exit 1
[[ "${CI_ARTIFACT_RUN_ATTEMPT:-1}" =~ ^[0-9]+$ ]] || exit 1
command -v zstd >/dev/null || { echo "zstd unavailable on host; use host build" >&2; exit 1; }
check_disk "$IMAGE_ARCHIVE_DIR" "${DEPLOY_IMAGE_RESERVE_KB:-12582912}"
umask 077
mkdir -p "$IMAGE_ARCHIVE_DIR"
work="$(mktemp -d "$IMAGE_ARCHIVE_DIR/.download-XXXXXX")"
cleanup() {
  rm -rf "$work"
  # Remove transport tags after Compose aliases exist (or a partial load failed).
  # Docker retains layers referenced by app/previous tags and running containers.
  docker image rm "supportchat-ci-backend:${DEPLOY_SHA}" "supportchat-ci-frontend:${DEPLOY_SHA}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# Keep the existing token transport/clone/fetch handling untouched. This extra
# REST use keeps Authorization out of argv, logs, redirects and stored files.
github_api() {
  printf 'header = "Authorization: Bearer %s"\n' "$GITHUB_TOKEN" |
    curl --config - --fail --silent --show-error --retry 2 --connect-timeout 10 \
      --max-time 600 --speed-limit 1048576 --speed-time 60 -H 'Accept: application/vnd.github+json' \
      -H 'X-GitHub-Api-Version: 2022-11-28' "$@"
}
name="supportchat-images-${DEPLOY_SHA}-${CI_ARTIFACT_RUN_ID}-${CI_ARTIFACT_RUN_ATTEMPT:-1}"
github_api "https://api.github.com/repos/${GITHUB_REPO}/actions/runs/${CI_ARTIFACT_RUN_ID}/artifacts?per_page=100" -o "$work/list.json"
artifact_id="$(python3 - "$work/list.json" "$name" <<'PY'
import json, shutil, sys
artifacts = json.load(open(sys.argv[1]))["artifacts"]
found = [a for a in artifacts if a["name"] == sys.argv[2] and not a["expired"]]
if len(found) != 1:
    sys.exit("No unique unexpired image artifact; use host build")
if int(found[0]["size_in_bytes"]) + 1024**3 > shutil.disk_usage(sys.argv[1]).free:
    sys.exit("Insufficient disk to download artifact")
print(int(found[0]["id"]))
PY
)"
# -L strips Authorization on the redirect to GitHub's signed blob URL (no --location-trusted).
github_api -L "https://api.github.com/repos/${GITHUB_REPO}/actions/artifacts/${artifact_id}/zip" -o "$work/artifact.zip"
python3 - "$work" "$DEPLOY_SHA" "$CI_ARTIFACT_RUN_ID" "${CI_ARTIFACT_RUN_ATTEMPT:-1}" <<'PY'
import hashlib, json, re, sys, zipfile
from pathlib import Path
root = Path(sys.argv[1])
expected = {"images.tar.zst", "images.tar.zst.sha256", "manifest.json"}
with zipfile.ZipFile(root / "artifact.zip") as archive:
    if set(archive.namelist()) != expected or len(archive.infolist()) != 3:
        sys.exit("Unexpected artifact members")
    # Bound zip expansion by free disk, without ever extracting paths from an archive.
    import shutil
    if sum(i.file_size for i in archive.infolist()) + 1024**3 > shutil.disk_usage(root).free:
        sys.exit("Insufficient disk for image artifact extraction")
    for name in expected:
        with archive.open(name) as src, (root / name).open("wb") as dst:
            shutil.copyfileobj(src, dst)
manifest = json.loads((root / "manifest.json").read_text())
if [manifest[k] for k in ("sha", "run_id", "run_attempt")] != sys.argv[2:]:
    sys.exit("Artifact commit/run mismatch")
for name, info in manifest["images"].items():
    if name not in {"backend", "frontend"} or info["ref"] != f"supportchat-ci-{name}:{sys.argv[2]}" or not isinstance(info.get("layers"), list) or not info.get("layers") or any(not re.fullmatch(r"sha256:[a-f0-9]{64}", layer) for layer in info["layers"]):
        sys.exit("Invalid artifact image identity")
if set(manifest["images"]) != {"backend", "frontend"}:
    sys.exit("Missing artifact image")
checksum = (root / "images.tar.zst.sha256").read_text().split()
with (root / "images.tar.zst").open("rb") as src:
    digest = hashlib.file_digest(src, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
    if digest is None:  # host Python 3.8–3.10
        src.seek(0)
        h = hashlib.sha256()
        for chunk in iter(lambda: src.read(1024 * 1024), b""): h.update(chunk)
        digest = h.hexdigest()
if checksum != [digest, "images.tar.zst"]:
    sys.exit("Artifact checksum mismatch")
PY
# The frontend image bakes public build arguments in. Refuse an artifact built with values
# that differ from this checkout's resolved ones (fall back to a host build).
expected_build_args="$(python3 deploy/compose-config.py build-args)"
python3 - "$work/manifest.json" "$expected_build_args" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest.get("build_args") != json.loads(sys.argv[2]):
    sys.exit("Artifact build arguments differ from this checkout; use host build")
PY
case "$(uname -m)" in
  x86_64) host_arch=amd64 ;;
  aarch64 | arm64) host_arch=arm64 ;;
  *) echo "Unsupported host architecture; use host build" >&2; exit 1 ;;
esac
zstd --test "$work/images.tar.zst"
zstd -dc "$work/images.tar.zst" | docker load
# RootFS diff IDs describe the image contents independently of classic/containerd image IDs.
python3 - "$work/manifest.json" <<'PY'
import json, subprocess, sys
manifest = json.load(open(sys.argv[1]))
for name, info in manifest["images"].items():
    actual = json.loads(subprocess.check_output(["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", info["ref"]], text=True))
    if actual != info["layers"]: sys.exit("Loaded image identity mismatch")
PY
for ref in "supportchat-ci-backend:${DEPLOY_SHA}" "supportchat-ci-frontend:${DEPLOY_SHA}"; do
  loaded_arch="$(docker image inspect --format '{{.Architecture}}' "$ref")"
  if [ "$loaded_arch" != "$host_arch" ]; then
    echo "Loaded image architecture ${loaded_arch} does not match host ${host_arch}; use host build" >&2
    exit 1
  fi
done
for service in backend worker migrate frontend; do
  case "$service" in frontend) source="supportchat-ci-frontend:${DEPLOY_SHA}" ;; *) source="supportchat-ci-backend:${DEPLOY_SHA}" ;; esac
  target="$(python3 deploy/compose-config.py images "$service")"
  docker tag "$source" "$target"
done
mv "$work/images.tar.zst" "$IMAGE_ARCHIVE_DIR/${name}.tar.zst"
mv "$work/images.tar.zst.sha256" "$IMAGE_ARCHIVE_DIR/${name}.sha256"
mv "$work/manifest.json" "$IMAGE_ARCHIVE_DIR/${name}.json"
python3 - "$IMAGE_ARCHIVE_DIR" <<'PY'
import sys
from pathlib import Path
files = sorted(Path(sys.argv[1]).glob("supportchat-images-*.tar.zst"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
# Retained checksum paths must name the retained (renamed) archive.
for path in files[:2]:
    checksum = path.with_name(path.name[:-len(".tar.zst")] + ".sha256")
    if checksum.exists():
        checksum.write_text(checksum.read_text().split()[0] + "  " + path.name + "\n")
for path in files[2:]:
    stem = path.name[:-len(".tar.zst")]
    path.unlink()
    for suffix in (".sha256", ".json"):
        path.with_name(stem + suffix).unlink(missing_ok=True)
PY
echo "Loaded verified CI images; last two archives retained."
