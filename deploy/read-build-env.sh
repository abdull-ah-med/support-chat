#!/usr/bin/env bash
# CI reads ONLY resolved public build arguments via the existing SSM send/poll protocol.
# The host may still be on an older checkout, so the read command is self-contained.
set -euo pipefail
: "${INSTANCE_ID:?}" "${REPO_DIR:?}" "${GITHUB_OUTPUT:?}"
[[ "$REPO_DIR" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo "Invalid repo path" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat > "$work/read.sh" <<SH_REMOTE
#!/usr/bin/env bash
set -euo pipefail
cd '$REPO_DIR'
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --format json 2>/dev/null | python3 -c '
import json, platform, sys
config = json.load(sys.stdin)
args = config["services"]["frontend"]["build"]["args"]
keys = ("API_ORIGIN", "NEXT_PUBLIC_API_ORIGIN", "NEXT_PUBLIC_WIDGET_ORIGIN", "NEXT_PUBLIC_STAFF_APP_ORIGIN", "NEXT_PUBLIC_MARKETING_HOST_ORIGIN", "NEXT_PUBLIC_DEMO_SITE_KEY", "NEXT_PUBLIC_DEMO_PUBLIC_KEY")
result = {key: args[key] for key in keys}
arch = platform.machine()
if arch not in ("x86_64", "aarch64"): sys.exit("Unsupported host architecture")
result["platform"] = "linux/amd64" if arch == "x86_64" else "linux/arm64"
print(json.dumps(result))
'
SH_REMOTE
commands="$(bash "$REPO_ROOT/deploy/ssm-commands-json.sh" "$work/read.sh")"
id="$(aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --comment 'Read public frontend build arguments' --parameters "{\"commands\":$commands}" \
  --query Command.CommandId --output text)"
status=Pending
for _ in $(seq 1 "${SSM_POLL_ITERATIONS:-60}"); do
  status="$(aws ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE_ID" \
    --query Status --output text 2>/dev/null || echo Pending)"
  case "$status" in
    Success) break ;;
    Failed|Cancelled|TimedOut) echo "Cannot read host public build arguments ($status)" >&2; exit 1 ;;
    *) sleep "${SSM_POLL_INTERVAL_SECONDS:-2}" ;;
  esac
done
[ "$status" = Success ] || { echo "Timed out reading public build arguments" >&2; exit 1; }
aws ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE_ID" \
  --output json > "$work/response.json"
python3 - "$work/response.json" "$GITHUB_OUTPUT" <<'PY'
import json, sys
response = json.load(open(sys.argv[1]))
values = json.loads(response["StandardOutputContent"])
keys = ("API_ORIGIN", "NEXT_PUBLIC_API_ORIGIN", "NEXT_PUBLIC_WIDGET_ORIGIN", "NEXT_PUBLIC_STAFF_APP_ORIGIN", "NEXT_PUBLIC_MARKETING_HOST_ORIGIN", "NEXT_PUBLIC_DEMO_SITE_KEY", "NEXT_PUBLIC_DEMO_PUBLIC_KEY", "platform")
with open(sys.argv[2], "a") as output:
    for key in keys:
        value = values[key]
        if not isinstance(value, str) or any(c in value for c in "\r\n\0"):
            sys.exit("Invalid public build argument: " + key)
        output.write(key + "=" + value + "\n")
PY
