import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# What the stubbed `docker compose config` resolves for the frontend build (see _deploy_host).
BUILD_ARGS = {
    "API_ORIGIN": "http://backend:8000",
    "NEXT_PUBLIC_API_ORIGIN": "",
    "NEXT_PUBLIC_WIDGET_ORIGIN": "https://widget.example.test",
    "NEXT_PUBLIC_STAFF_APP_ORIGIN": "https://staff.example.test",
    "NEXT_PUBLIC_MARKETING_HOST_ORIGIN": "https://marketing.example.test",
    "NEXT_PUBLIC_DEMO_SITE_KEY": "demo",
    "NEXT_PUBLIC_DEMO_PUBLIC_KEY": "",
}


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def _code(text: str) -> str:
    """Drop whole-line comments so assertions inspect behaviour, not the prose around it."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _services(compose: str) -> dict[str, str]:
    """Split a compose file into {service name: block text} without a YAML parser.

    The files use the custom `!override` / `!reset` tags, which safe_load rejects.
    """
    body = compose.split("\nservices:\n", 1)[1]
    body = re.split(r"\n(?=[a-z]\w*:)", body, maxsplit=1)[0]  # stop at the next top-level key
    parts = re.split(r"^  ([a-z][\w-]*):\n", body, flags=re.MULTILINE)
    return {name: block for name, block in zip(parts[1::2], parts[2::2], strict=True)}


# ---- deploy script -------------------------------------------------------------------------


def test_deploy_always_updates_the_worker_service() -> None:
    deploy = _read("deploy/ssm-deploy.sh")
    sender = _read("deploy/github-actions-ssm-deploy.sh")
    assert "COMPOSE_APP_SERVICES:-backend worker frontend" in deploy
    assert "COMPOSE_APP_SERVICES:-backend worker frontend" in sender


def test_deploy_fails_unless_required_services_become_healthy() -> None:
    deploy = _read("deploy/ssm-deploy.sh")
    assert "up -d --no-build --no-deps --wait --wait-timeout 180" in deploy
    assert "set -euo pipefail" in deploy


def test_deploy_migrates_explicitly_before_starting_the_app(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    events = _host_events(env)
    migrate = next(i for i, event in enumerate(events) if "run" in event and event[-1] == "migrate")
    up = next(i for i, event in enumerate(events) if "up" in event and "backend" in event)
    assert migrate < up
    assert "--no-deps" in events[up]
    assert not any("up" in event and "postgres" in event for event in events)


def test_deploy_keeps_a_rollback_image_and_never_prunes_it() -> None:
    deploy = _read("deploy/ssm-deploy.sh")
    tag = deploy.index('docker tag "$image_id" "${repo}:previous"')
    assert tag < deploy.index('"${COMPOSE[@]}" build ')
    # `image prune -a` deletes every image no container uses, including the `:previous` tag.
    assert not re.search(r"docker image prune[^\n]*-\w*a", deploy)
    assert "docker image prune -f" in deploy
    rollback = _read("deploy/rollback.sh")
    assert ":previous" in rollback and "--no-build" in rollback
    assert "set -euo pipefail" in rollback


# ---- redis --------------------------------------------------------------------------------


def _app_acl_line() -> str:
    entrypoint = _read("deploy/redis-secure-entrypoint.sh")
    return next(line for line in entrypoint.splitlines() if "user app on" in line)


def test_production_redis_uses_acl_app_identity() -> None:
    compose = _read("docker-compose.prod.yml")
    entrypoint = _read("deploy/redis-secure-entrypoint.sh")
    assert "--aclfile" in entrypoint
    assert "user default off" in entrypoint
    assert "user app on" in entrypoint
    assert "+select" in entrypoint
    assert "+hello" in entrypoint
    assert "REDIS_APP_PASSWORD" in compose
    assert "rediss://app:" in _read("backend/.env.prod.example")
    assert "rediss://app:" in _read("backend/.env.example")


def test_redis_app_acl_grants_only_what_the_app_calls() -> None:
    acl = _app_acl_line()
    assert not re.search(r"\+client(\s|$)", acl), "bare +client allows CLIENT KILL / PAUSE"
    for sub in ("setinfo", "setname", "id"):
        assert f"+client|{sub}" in acl
    for dangerous in ("+@all", "+flush", "+config", "+keys", "+debug", "+script", "+shutdown"):
        assert dangerous not in acl
    # rate_limit.py and rate_ceiling.py run Lua, so +eval is required and must stay.
    app_sources = "".join(p.read_text() for p in (ROOT / "backend/app").rglob("*.py"))
    assert ".eval(" in app_sources
    assert "+eval" in acl


def test_redis_is_memory_bounded_and_drops_privileges() -> None:
    entrypoint = _read("deploy/redis-secure-entrypoint.sh")
    assert '--maxmemory "$REDIS_MAXMEMORY"' in entrypoint
    assert "--maxmemory-policy volatile-lru" in entrypoint
    assert 'REDIS_MAXMEMORY="${REDIS_MAXMEMORY:-512mb}"' in entrypoint
    assert "setpriv --reuid redis" in entrypoint
    assert "REDIS_MAXMEMORY" in _read("docker-compose.prod.yml")
    # The key is handed to the server as a private 0600 copy owned by redis; the host copy is 0600.
    assert "redis.key" in entrypoint and "chown redis:redis" in entrypoint
    assert 'chmod 600 "$TLS_DIR/redis.key"' in _read("deploy/generate-redis-tls.sh")
    assert "rm -f" in _read("deploy/generate-redis-tls.sh")  # CA key is not left on the host


def test_query_vector_cache_ttl_is_short_in_env_examples() -> None:
    for name in ("backend/.env.example", "backend/.env.prod.example"):
        assert re.search(r"^QUERY_VECTOR_CACHE_TTL=3600$", _read(name), flags=re.MULTILINE), name


# ---- images ------------------------------------------------------------------------------


def test_backend_container_runs_as_non_root() -> None:
    dockerfile = _read("backend/Dockerfile")
    assert "USER app" in dockerfile


def test_backend_image_caps_websocket_message_size() -> None:
    dockerfile = _read("backend/Dockerfile")
    assert '"--ws-max-size", "65536"' in dockerfile
    assert "--ws-max-size" not in _read("docker-compose.yml")  # no override that would undo it


def test_images_declare_healthchecks() -> None:
    assert "HEALTHCHECK" in _read("backend/Dockerfile")
    assert "/health" in _read("backend/Dockerfile").split("HEALTHCHECK", 1)[1]
    assert "HEALTHCHECK" in _read("frontend/Dockerfile")


def test_backend_image_ships_only_scripts_the_app_imports_or_operators_run() -> None:
    dockerfile = _code(_read("backend/Dockerfile"))
    assert "COPY scripts ./scripts" not in dockerfile
    imported = set()
    for path in (ROOT / "backend/app").rglob("*.py"):
        imported.update(re.findall(r"^\s*from scripts\.(\w+) import", path.read_text(), re.M))
    assert imported, "expected app/workers/maintenance.py to import the purge scripts"
    for module in imported:
        assert f"scripts/{module}.py" in dockerfile, (
            f"app imports scripts.{module} but the image lacks it"
        )
    assert "scripts/seed_user.py" in dockerfile  # provisioning
    for dev_only in ("harness_chat", "live_bot_probe", "run_site_eval", "docker_up"):
        assert dev_only not in dockerfile


def test_shell_entrypoints_fail_on_pipeline_errors() -> None:
    assert "set -euo pipefail" in _read("backend/docker/entrypoint.sh")
    assert "set -euo pipefail" in _read("backend/scripts/docker_up.sh")
    assert _read("backend/docker/entrypoint.sh").startswith("#!/bin/bash")  # dash has no pipefail


# ---- compose -----------------------------------------------------------------------------


def test_runtime_database_role_is_not_the_migration_or_admin_role() -> None:
    compose = _read("docker-compose.yml")
    assert "APP_DB_USER" in compose
    assert "MIGRATION_DB_USER" in compose
    assert "service_completed_successfully" in compose
    assert "alembic upgrade head" not in _read("backend/docker/entrypoint.sh")
    provision = _read("deploy/provision-postgres-roles.sh")
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC" in provision
    assert "ALTER DEFAULT PRIVILEGES" in provision
    assert "GRANT EXECUTE ON ALL FUNCTIONS" in provision
    assert "GRANT USAGE ON ALL TYPES" in provision


def test_production_overlay_has_no_local_credential_fallbacks() -> None:
    prod = _read("docker-compose.prod.yml")
    provision = _services(prod)["db-provision"]
    for var in (
        "POSTGRES_PASSWORD",
        "APP_DB_PASSWORD",
        "MIGRATION_DB_PASSWORD",
        "APP_DB_USER",
        "MIGRATION_DB_USER",
    ):
        assert f"${{{var}:?" in provision, var
    assert ":-" not in provision
    assert not re.search(r"local-(chat|dev|admin)", _code(prod))


def test_every_production_service_is_hardened_and_resource_limited() -> None:
    services = _services(_read("docker-compose.prod.yml"))
    assert set(services) == {
        "postgres",
        "db-provision",
        "redis",
        "backend",
        "migrate",
        "worker",
        "frontend",
    }
    for name, block in services.items():
        if name == "postgres":
            continue  # preserve the database container on this deploy
        assert "no-new-privileges:true" in block, name
        assert 'cap_drop: ["ALL"]' in block, name
        assert "mem_limit:" in block, name
        # No `cpus:`: Docker refuses to start a container whose limit exceeds the host's CPUs.
        assert "cpus:" not in _code(block), name
        assert "pids_limit:" in block, name
    # Images that start as root and drop privileges need exactly these capabilities back.
    for cap in ("CHOWN", "SETGID", "SETUID"):
        assert cap in services["redis"]
    # Chromium crawls untrusted URLs next to the database: the worker must be bounded.
    assert "WORKER_MEM_LIMIT" in services["worker"]
    assert "pids_limit: 1024" in services["worker"]


def test_production_database_images_are_pinned_by_digest() -> None:
    services = _services(_read("docker-compose.prod.yml"))
    for name in ("db-provision", "redis"):
        assert re.search(r"image: \S+@sha256:[0-9a-f]{64}", services[name]), name


def test_compose_network_is_not_pinned_and_trusted_proxy_is_explicit() -> None:
    # Recreating the existing host's default network is disruptive and would orphan its
    # TRUSTED_PROXY_CIDRS=172.18.0.1/32, so the overlay must not define one.
    prod = _code(_read("docker-compose.prod.yml"))
    assert "subnet" not in prod and "gateway" not in prod and "\nnetworks:" not in prod
    assert "TRUSTED_PROXY_CIDRS=172.18.0.1/32" in _read("backend/.env.prod.example")


def _run_entrypoint(
    tmp_path: Path, route: str | None, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run the real entrypoint with a fake `python` (both wait loops pass) and print the env var."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    fake_python.chmod(0o755)
    route_file = tmp_path / "route"
    route_file.write_text(route if route is not None else "")
    base = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "PROC_NET_ROUTE": str(route_file)}
    return subprocess.run(
        [
            "bash",
            str(ROOT / "backend/docker/entrypoint.sh"),
            "sh",
            "-c",
            'echo "CIDRS=[${TRUSTED_PROXY_CIDRS-unset}]"',
        ],
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


_ROUTE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
    "eth0\t00000000\t010012AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
    "eth0\t000012AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"
)


def test_entrypoint_resolves_auto_to_the_default_gateway(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, _ROUTE, TRUSTED_PROXY_CIDRS="auto")
    assert result.returncode == 0, result.stderr
    assert "CIDRS=[172.18.0.1/32]" in result.stdout


def test_entrypoint_resolves_unset_only_in_production(tmp_path: Path) -> None:
    prod = _run_entrypoint(tmp_path, _ROUTE, APP_ENV="production")
    assert "CIDRS=[172.18.0.1/32]" in prod.stdout
    local = _run_entrypoint(tmp_path, _ROUTE, APP_ENV="local")
    assert "CIDRS=[unset]" in local.stdout


def test_entrypoint_leaves_explicit_values_untouched(tmp_path: Path) -> None:
    explicit = _run_entrypoint(
        tmp_path, _ROUTE, TRUSTED_PROXY_CIDRS="10.9.8.1/32", APP_ENV="production"
    )
    assert "CIDRS=[10.9.8.1/32]" in explicit.stdout
    empty = _run_entrypoint(tmp_path, _ROUTE, TRUSTED_PROXY_CIDRS="")
    assert "CIDRS=[]" in empty.stdout


def test_entrypoint_fails_when_auto_cannot_find_a_gateway(tmp_path: Path) -> None:
    no_default = "Iface\tDestination\tGateway\neth0\t000012AC\t00000000\n"
    result = _run_entrypoint(tmp_path, no_default, TRUSTED_PROXY_CIDRS="auto")
    assert result.returncode != 0
    assert "CIDRS=" not in result.stdout


def test_background_workers_are_enabled_explicitly_per_service() -> None:
    for compose in ("docker-compose.yml", "docker-compose.prod.yml"):
        services = _services(_read(compose))
        assert 'ENABLE_BACKGROUND_WORKERS: "false"' in services["backend"], compose
        assert 'ENABLE_BACKGROUND_WORKERS: "true"' in services["worker"], compose


def test_edge_sets_transport_and_capability_headers() -> None:
    nginx = _read("deploy/nginx.conf.template")
    assert "Strict-Transport-Security" in nginx
    assert "Permissions-Policy" in nginx
    assert "client_max_body_size 4k;" in nginx


def test_production_backend_drops_capabilities_like_the_worker() -> None:
    compose = _read("docker-compose.prod.yml")
    backend_block = compose.split("backend:", 1)[1].split("\n  migrate:", 1)[0]
    assert 'cap_drop: ["ALL"]' in backend_block
    assert "no-new-privileges:true" in backend_block


def test_local_compose_publishes_only_on_loopback() -> None:
    compose = _read("docker-compose.yml")
    assert '"127.0.0.1:${POSTGRES_PORT:-55432}:5432"' in compose
    assert '"127.0.0.1:${REDIS_PORT:-56379}:6379"' in compose
    assert '"127.0.0.1:${BACKEND_PORT:-8000}:8000"' in compose
    assert '"127.0.0.1:${FRONTEND_PORT:-3000}:3000"' in compose


# ---- CI ------------------------------------------------------------------------------------


def test_workflows_default_to_read_only_token_and_scope_oidc_to_the_deploy_job() -> None:
    for name in ("ci-cd.yml", "deploy-dev.yml"):
        workflow = _read(f".github/workflows/{name}")
        header = workflow.split("\njobs:", 1)[0]
        assert re.search(r"^permissions:\n  contents: read$", header, flags=re.MULTILINE), name
        deploy_job = workflow.split("name: Deploy to EC2", 1)[1]
        assert "id-token: write" in deploy_job, name
        assert "id-token: write" not in header


def test_ci_audits_dependencies_and_dependabot_is_configured() -> None:
    ci = _read(".github/workflows/ci-cd.yml")
    assert "pip-audit" in ci
    assert "npm audit --omit=dev --audit-level=high" in ci
    dependabot = _read(".github/dependabot.yml")
    for ecosystem in ("uv", "npm", "github-actions", "docker", "docker-compose"):
        assert f"package-ecosystem: {ecosystem}" in dependabot


# Script boundary tests: the owner request is the oracle. The fake engine persists
# which images serve traffic, so rollback assertions observe an outcome too.
def _deploy_host(tmp_path: Path, **scenario: str) -> tuple[Path, dict[str, str]]:
    import json
    import shutil

    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / ".git").mkdir()
    shutil.copytree(ROOT / "deploy", repo / "deploy")
    for name in ("docker-compose.yml", "docker-compose.prod.yml"):
        shutil.copy(ROOT / name, repo / name)
    required = {
        "APP_ENV": "production",
        "APP_DB_USER": "app",
        "APP_DB_PASSWORD": "private-app-value",
        "MIGRATION_DB_USER": "migrator",
        "MIGRATION_DB_PASSWORD": "private-migration-value",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "private-postgres-value",
        "POSTGRES_DB": "production_database",
        "REDIS_APP_PASSWORD": "private-redis-value",
        "REDIS_ADMIN_PASSWORD": "private-admin-value",
        "JWT_SECRET": "private-jwt-value",
        "WIDGET_TOKEN_SECRET": "private-widget-value",
        "RATE_KEY_SECRET": "private-rate-value",
        "WIDGET_CSP_SERVICE_SECRET": "private-csp-value",
        "NEXT_PUBLIC_STAFF_APP_ORIGIN": "https://staff.example.test",
        "NEXT_PUBLIC_WIDGET_ORIGIN": "https://widget.example.test",
        "NEXT_PUBLIC_MARKETING_HOST_ORIGIN": "https://marketing.example.test",
    }
    (repo / ".env").write_text("".join(f"{k}={v}\n" for k, v in required.items()))
    (repo / "backend").mkdir()
    (repo / "backend/.env.prod").write_text("TRUSTED_PROXY_CIDRS=172.18.0.1/32\n")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"serving": "old", "migrations": 0, "tags": {}}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = """#!/usr/bin/env python3
import json, os, pathlib, platform, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
state_file = pathlib.Path(os.environ["HOST_STATE"])
s = json.loads(state_file.read_text())
with open(os.environ["HOST_EVENTS"], "a") as f:
    f.write(json.dumps([name, *args]) + "\\n")
def save(): state_file.write_text(json.dumps(s))
if name == "git": sys.exit(0)
if name == "flock": sys.exit(1 if os.environ.get("LOCK_BUSY") else 0)
if name == "timeout":
    sys.exit(subprocess.call(args[2:]))
if name == "pg_dump":
    if os.environ.get("BACKUP_FAIL"): sys.exit(2)
    sys.stdout.write("PGDMP-test-backup")
    sys.exit(0)
if name == "df":
    print("Filesystem 1024-blocks Used Available Capacity Mounted on")
    print("disk 999999999 1 " + os.environ.get("FREE_KB", "999999998") + " 1% /")
    sys.exit(0)
if name == "curl":
    if any("api.github.com" in a for a in args):
        if os.environ.get("ARTIFACT_FAIL"): sys.exit(22)
        target = pathlib.Path(args[args.index("-o") + 1])
        if any("/artifacts/" in a and a.endswith("/zip") for a in args):
            import shutil
            shutil.copyfile(os.environ["ARTIFACT_ZIP"], target)
        else:
            target.write_text(json.dumps({"artifacts": [{"id": 99, "expired": False, "size_in_bytes": 1024,
                "name": "supportchat-images-" + "a"*40 + "-12345-1"}]}))
        sys.exit(0)
    if "--write-out" in args: print(os.environ.get("SMOKE_STATUS", "200"))
    sys.exit(22 if os.environ.get("SMOKE_FAIL") else 0)
if name == "aws": sys.exit("Unexpected AWS access on the host")
if name == "nginx": sys.exit(0)
if name == "systemctl": sys.exit(0)
if name != "docker": sys.exit(0)
if args[:2] == ["volume", "inspect"]:
    sys.exit(1 if os.environ.get("VOLUME_MISSING") or os.environ.get("EXACT_VOLUME_MISSING") else 0)
if args[:2] == ["volume", "ls"]:
    if not os.environ.get("VOLUME_MISSING"): print("throwaway_postgres_data")
    sys.exit(0)
if args[0] == "inspect":
    fmt, cid = args[2], args[-1]
    service = cid.removeprefix("cid-")
    if "Health" in fmt: print("healthy")
    elif "Config.Image" in fmt: print("support-chat-" + service)
    else: print("old-" + service)
    sys.exit(0)
if args[:2] == ["image", "inspect"]:
    if args[-1].endswith(":previous") and os.environ.get("MISSING_PREVIOUS", "__none__") in args[-1]: sys.exit(1)
    if "--format" in args:
        fmt = args[args.index("--format") + 1]
        if "RootFS.Layers" in fmt:
            print(json.dumps(["sha256:" + ("0" if os.environ.get("LAYER_MISMATCH") else "b" if "backend" in args[-1] else "f") * 64]))
        elif "Architecture" in fmt:
            print(os.environ.get("LOADED_ARCH") or ("amd64" if platform.machine() == "x86_64" else "arm64"))
        else: print("sha256:" + "d" * 64)
    sys.exit(0)
if args[0] == "load":
    sys.stdin.buffer.read()
    sys.exit(2 if os.environ.get("LOAD_FAIL") else 0)
if args[0] == "tag":
    target = args[2] if ":" in args[2].split("/")[-1] else args[2] + ":latest"
    source = "ci-backend" if args[1].startswith("supportchat-ci-backend:") else "ci-frontend" if args[1].startswith("supportchat-ci-frontend:") else s["tags"].get(args[1], args[1])
    s["tags"][target] = source; save(); sys.exit(0)
if args[0] in ("image", "builder"): sys.exit(0)
if args[0] == "exec":
    if "pg_dump" in " ".join(args): sys.exit(subprocess.call(["pg_dump"]))
    if "pg_database_size" in " ".join(args):
            if os.environ.get("PSQL_FAIL"): sys.exit("STATEMENT: SELECT private-password")
            print("1048576")
    elif "INFO" in args or "INFO memory" in " ".join(args):
        print("used_memory:" + os.environ.get("REDIS_USED", "1048576") + "\\r\\nmaxmemory:0")
    sys.exit(0)
if args[0] == "compose":
    args = args[1:]
    while args and args[0] in ("-f", "--env-file", "--project-name", "-p"):
        args = args[2:]
    command = args[0]
    if command == "config":
        env = dict(line.split("=", 1) for line in pathlib.Path(".env").read_text().splitlines() if "=" in line)
        if "--environment" in args:
            print("\\n".join(k + "=" + v for k,v in env.items())); sys.exit(0)
        import re
        required = set(re.findall(r"\\$\\{([A-Z_]+):\\?", "\\n".join(line for line in pathlib.Path("docker-compose.prod.yml").read_text().splitlines() if not line.lstrip().startswith("#"))))
        for k in required:
            if not env.get(k): sys.exit("required variable " + k + " is missing a value")
        print(json.dumps({"name":"support-chat", "services": {
            "frontend": {"build":{"args": {"API_ORIGIN":"http://backend:8000", "NEXT_PUBLIC_API_ORIGIN":"", "NEXT_PUBLIC_DEMO_SITE_KEY":"demo", "NEXT_PUBLIC_DEMO_PUBLIC_KEY":"", **{k:v for k,v in env.items() if k.startswith("NEXT_PUBLIC_")}}}},
            "redis":{"environment":{"REDIS_MAXMEMORY":"512mb"},"mem_limit":1073741824},
            **{n:{"image":"support-chat-" + n} for n in ("backend","worker","migrate")}
        }})); sys.exit(0)
    if command == "ps":
        service = args[-1]; print("cid-" + service); sys.exit(0)
    if command == "exec":
        if "pg_dump" in " ".join(args): sys.exit(subprocess.call(["pg_dump"]))
        if "pg_database_size" in " ".join(args):
            if os.environ.get("PSQL_FAIL"): sys.exit("STATEMENT: SELECT private-password")
            print("1048576")
        elif "INFO" in args: print("used_memory:" + os.environ.get("REDIS_USED", "1048576") + "\\r\\nmaxmemory:0")
        sys.exit(0)
    if command == "build":
        for service in ("backend","worker","frontend","migrate"):
            s["tags"]["support-chat-" + service + ":latest"] = "new-" + service
        save(); sys.exit(0)
    if command == "run":
        if args[-1] == "migrate": s["migrations"] += 1
        save()
        sys.exit(2 if args[-1] == "migrate" and os.environ.get("MIGRATE_FAIL") else 0)
    if command == "up":
        if args[-1] == "redis": sys.exit(0)
        if os.environ.get("UP_FAIL") and s["tags"].get("support-chat-backend:latest") == "new-backend": sys.exit(2)
        s["serving"] = s["tags"].get("support-chat-backend:latest", "new-backend"); save()
        sys.exit(0)
sys.exit(0)
"""
    # python3 resolves to the installed test interpreter; no packages are installed.
    import sys

    (bin_dir / "python3").symlink_to(sys.executable)
    for name in (
        "docker",
        "git",
        "curl",
        "aws",
        "pg_dump",
        "df",
        "nginx",
        "systemctl",
        "flock",
        "timeout",
    ):
        script = bin_dir / name
        script.write_text(stub)
        script.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_DIR": str(repo),
        "GITHUB_REPO": "owner/repo",
        "BRANCH": "dev",
        "DEPLOY_SHA": "a" * 40,
        "GITHUB_TOKEN": "never-log-this-token",
        "CI_ARTIFACT_RUN_ID": "12345",
        "CI_ARTIFACT_RUN_ATTEMPT": "1",
        "BACKUP_DIR": str(tmp_path / "backups"),
        "IMAGE_ARCHIVE_DIR": str(tmp_path / "images"),
        "NGINX_CONF_DEST": str(tmp_path / "missing-nginx.conf"),
        "DEPLOY_LOCK_FILE": str(tmp_path / "deploy.lock"),
        "HOST_STATE": str(state),
        "HOST_EVENTS": str(tmp_path / "events.jsonl"),
        **scenario,
    }
    return repo, env


def _host_run(
    env: dict[str, str], script: str = "ssm-deploy.sh"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(Path(env["REPO_DIR"]) / "deploy" / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _host_state(env: dict[str, str]) -> dict:
    import json

    return json.loads(Path(env["HOST_STATE"]).read_text())


def _host_events(env: dict[str, str]) -> list[list[str]]:
    import json

    return [json.loads(line) for line in Path(env["HOST_EVENTS"]).read_text().splitlines()]


def test_backup_failure_keeps_old_app_serving_without_migration(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, BACKUP_FAIL="1")
    result = _host_run(env)
    assert result.returncode != 0
    assert _host_state(env)["migrations"] == 0
    assert _host_state(env)["serving"] == "old"
    assert not list(Path(env["BACKUP_DIR"]).glob("*.dump"))


def test_smoke_failure_restores_app_images_without_rerunning_migrate(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, SMOKE_FAIL="1")
    result = _host_run(env)
    assert result.returncode != 0
    assert _host_state(env)["serving"] == "old-backend"
    assert _host_state(env)["migrations"] == 1
    rollback = [e for e in _host_events(env) if "up" in e and "backend" in e]
    assert len(rollback) == 2
    assert "--no-deps" in rollback[-1]
    assert rollback[-1][-3:] == ["backend", "worker", "frontend"]
    assert "support-chat-migrate:previous" not in _host_state(env)["tags"]


def test_unhealthy_up_restores_app_images(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, UP_FAIL="1")
    result = _host_run(env)
    assert result.returncode != 0
    assert _host_state(env)["serving"] == "old-backend"
    assert _host_state(env)["migrations"] == 1


def test_artifact_failure_builds_on_host_and_finishes_deploy(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert _host_state(env)["serving"] == "new-backend"
    assert _host_state(env)["migrations"] == 1
    assert any(e[0] == "curl" for e in _host_events(env))
    backups = list(Path(env["BACKUP_DIR"]).glob("*.dump"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"PGDMP-test-backup"
    assert backups[0].stat().st_mode & 0o777 == 0o600


def test_preflight_missing_var_aborts_without_printing_values(tmp_path: Path) -> None:
    repo, env = _deploy_host(tmp_path)
    dotenv = repo / ".env"
    dotenv.write_text(dotenv.read_text().replace("APP_DB_PASSWORD=private-app-value\n", ""))
    result = _host_run(env, "preflight.sh")
    assert result.returncode != 0
    assert "APP_DB_PASSWORD" in result.stderr
    for secret in ("private-", "never-log-this-token"):
        assert secret not in result.stdout + result.stderr
    assert _host_state(env)["migrations"] == 0


def test_postgres_volume_guard_blocks_all_build_and_migration(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, VOLUME_MISSING="1")
    result = _host_run(env)
    assert result.returncode != 0
    assert "refusing fresh database" in result.stderr
    assert _host_state(env) == {"serving": "old", "migrations": 0, "tags": {}}
    assert not any("build" in e or "run" in e or "up" in e for e in _host_events(env))


def test_preflight_low_disk_aborts_before_build(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, FREE_KB="100")
    result = _host_run(env)
    assert result.returncode != 0
    assert "disk" in result.stderr.lower()
    assert _host_state(env)["migrations"] == 0
    assert _host_state(env)["serving"] == "old"


def test_preflight_rejects_redis_above_eighty_percent_of_new_limit(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, REDIS_USED="450887680")
    result = _host_run(env)
    assert result.returncode != 0
    assert "Redis" in result.stderr
    assert _host_state(env)["serving"] == "old"


def _image_artifact(
    tmp_path: Path, env: dict[str, str], build_args: dict[str, str] | None = None
) -> None:
    import hashlib
    import json
    import zipfile

    archive = subprocess.check_output(["zstd", "-q", "-c"], input=b"test-image-tar")
    manifest = {
        "sha": "a" * 40,
        "run_id": "12345",
        "run_attempt": "1",
        "build_args": BUILD_ARGS if build_args is None else build_args,
        "images": {
            "backend": {"ref": "supportchat-ci-backend:" + "a" * 40, "layers": ["sha256:" + "b" * 64]},
            "frontend": {
                "ref": "supportchat-ci-frontend:" + "a" * 40,
                "layers": ["sha256:" + "f" * 64],
            },
        },
    }
    artifact = tmp_path / "artifact.zip"
    with zipfile.ZipFile(artifact, "w") as zipped:
        zipped.writestr("images.tar.zst", archive)
        zipped.writestr(
            "images.tar.zst.sha256", hashlib.sha256(archive).hexdigest() + "  images.tar.zst\n"
        )
        zipped.writestr("manifest.json", json.dumps(manifest))
    env["ARTIFACT_ZIP"] = str(artifact)


def test_verified_ci_artifact_serves_without_host_build_and_keeps_two_archives(
    tmp_path: Path,
) -> None:
    _, env = _deploy_host(tmp_path)
    _image_artifact(tmp_path, env)
    directory = Path(env["IMAGE_ARCHIVE_DIR"])
    directory.mkdir()
    for n in (1, 2):
        (directory / f"supportchat-images-older-{n}.tar.zst").write_bytes(b"old")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert _host_state(env)["serving"] == "ci-backend"
    assert _host_state(env)["migrations"] == 1
    assert not any("build" in event for event in _host_events(env))
    assert len(list(directory.glob("*.tar.zst"))) == 2


def test_docker_load_failure_falls_back_to_host_build(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, LOAD_FAIL="1")
    _image_artifact(tmp_path, env)
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert _host_state(env)["serving"] == "new-backend"
    assert _host_state(env)["migrations"] == 1
    assert any(event[:2] == ["docker", "load"] for event in _host_events(env))
    assert not list(Path(env["IMAGE_ARCHIVE_DIR"]).glob("*.tar.zst"))


def test_backup_retains_only_five_complete_dumps(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1")
    directory = Path(env["BACKUP_DIR"])
    directory.mkdir()
    for n in range(5):
        path = directory / f"20260101-00000{n}.dump"
        path.write_bytes(b"PGDMP-old")
        os.utime(path, (n + 1, n + 1))
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert len(list(directory.glob("*.dump"))) == 5
    assert not (directory / "20260101-000000.dump").exists()
    assert (directory / "20260101-000004.dump").exists()


def test_failed_migration_does_not_start_new_apps_or_rerun_migrate(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1", MIGRATE_FAIL="1")
    result = _host_run(env)
    assert result.returncode != 0
    assert _host_state(env)["serving"] == "old"
    assert _host_state(env)["migrations"] == 1
    assert not any("up" in event and "backend" in event for event in _host_events(env))


def test_ci_reads_exact_public_build_args_through_ssm_without_secret_output(tmp_path: Path) -> None:
    import json

    _, env = _deploy_host(tmp_path)
    output = tmp_path / "outputs"
    response = tmp_path / "public-response.json"
    response.write_text(
        json.dumps(
            {
                "StandardOutputContent": json.dumps(
                    {
                        "API_ORIGIN": "http://backend:8000",
                        "NEXT_PUBLIC_API_ORIGIN": "",
                        "NEXT_PUBLIC_WIDGET_ORIGIN": "https://widget.example.test",
                        "NEXT_PUBLIC_STAFF_APP_ORIGIN": "https://staff.example.test",
                        "NEXT_PUBLIC_MARKETING_HOST_ORIGIN": "https://marketing.example.test",
                        "NEXT_PUBLIC_DEMO_SITE_KEY": "demo",
                        "NEXT_PUBLIC_DEMO_PUBLIC_KEY": "",
                        "platform": "linux/arm64",
                    }
                )
            }
        )
    )
    env.update(
        INSTANCE_ID="i-local-test",
        GITHUB_OUTPUT=str(output),
        SSM_POLL_INTERVAL_SECONDS="0",
        PUBLIC_RESPONSE=str(response),
    )
    bin_dir = Path(env["PATH"].split(":")[0])
    aws = bin_dir / "aws"
    aws.write_text("""#!/usr/bin/env python3
import os, pathlib, sys
args=sys.argv[1:]
if args[1] == "send-command": print("local-command-id")
elif "--query" in args: print("Success")
else: print(pathlib.Path(os.environ["PUBLIC_RESPONSE"]).read_text())
""")
    result = _host_run(env, "read-build-env.sh")
    assert result.returncode == 0, result.stderr
    assert output.read_text() == (
        "API_ORIGIN=http://backend:8000\n"
        "NEXT_PUBLIC_API_ORIGIN=\n"
        "NEXT_PUBLIC_WIDGET_ORIGIN=https://widget.example.test\n"
        "NEXT_PUBLIC_STAFF_APP_ORIGIN=https://staff.example.test\n"
        "NEXT_PUBLIC_MARKETING_HOST_ORIGIN=https://marketing.example.test\n"
        "NEXT_PUBLIC_DEMO_SITE_KEY=demo\n"
        "NEXT_PUBLIC_DEMO_PUBLIC_KEY=\n"
        "platform=linux/arm64\n"
    )
    assert "private-" not in output.read_text() + result.stdout + result.stderr
    assert "never-log-this-token" not in output.read_text() + result.stdout + result.stderr


def test_entrypoint_reports_settings_error_on_final_postgres_attempt(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        (
            "python",
            '#!/bin/sh\necho "input_value=private-password STATEMENT: SELECT secret" >&2\nexit 1\n',
        ),
        ("sleep", "#!/bin/sh\nexit 0\n"),
    ):
        script = bin_dir / name
        script.write_text(body)
        script.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "backend/docker/entrypoint.sh"), "sh", "-c", "echo started"],
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "TRUSTED_PROXY_CIDRS": "172.18.0.1/32"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "Postgres did not become ready" in result.stderr
    assert "input_value" not in result.stderr
    assert "private-password" not in result.stderr
    assert "STATEMENT" not in result.stderr
    assert "started" not in result.stdout


def test_public_redirect_is_not_a_healthy_deployment(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1", SMOKE_STATUS="302")
    result = _host_run(env)
    assert result.returncode != 0
    assert _host_state(env)["serving"] == "old-backend"
    assert _host_state(env)["migrations"] == 1


# Oracle: the user's deployment review findings; executables are local OS-boundary stubs.
def test_busy_host_lock_prevents_checkout(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, LOCK_BUSY="1")
    result = _host_run(env)
    assert result.returncode == 1
    assert "another deploy is running" in result.stderr
    assert not any(event[0] == "git" for event in _host_events(env))


def test_smoke_resolves_public_tls_hosts_to_local_nginx(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path)
    result = _host_run(env, "smoke-test.sh")
    assert result.returncode == 0, result.stderr
    calls = [event for event in _host_events(env) if event[0] == "curl"]
    assert [event[-1] for event in calls] == [
        "https://staff.example.test/login",
        "https://widget.example.test/widget",
        "https://marketing.example.test/demo",
        "https://staff.example.test/health",
    ]
    for event in calls:
        host = event[-1].split("/")[2]
        assert event[event.index("--resolve") + 1] == host + ":443:127.0.0.1"
        assert "--insecure" not in event
        assert event[event.index("--retry-max-time") + 1] == "60"


def test_rollback_recovers_available_services_when_one_previous_is_missing(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, MISSING_PREVIOUS="worker")
    result = _host_run(env, "rollback.sh")
    assert result.returncode != 0  # report incomplete recovery, after restoring available services
    assert "worker" in result.stderr
    up = [e for e in _host_events(env) if "up" in e]
    assert len(up) == 1
    assert up[0][-2:] == ["backend", "frontend"]
    assert "--no-deps" in up[0]
    assert _host_state(env)["tags"]["support-chat-backend:latest"] == "support-chat-backend:previous"
    assert "support-chat-worker:latest" not in _host_state(env)["tags"]


def test_exact_postgres_volume_precedes_legacy_filter_fallback(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    calls = [e for e in _host_events(env) if e[1:2] == ["volume"]]
    assert calls == [["docker", "volume", "inspect", "support-chat_postgres_data"]]


def test_legacy_postgres_volume_remains_usable(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, EXACT_VOLUME_MISSING="1", ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert ["docker", "volume", "ls", "-q", "--filter", "name=postgres_data"] in _host_events(env)


def test_database_size_failure_summarizes_instead_of_printing_sql(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, PSQL_FAIL="1")
    result = _host_run(env, "preflight.sh")
    assert result.returncode != 0
    assert "Cannot estimate database backup size" in result.stderr
    assert "STATEMENT" not in result.stderr
    assert "private-password" not in result.stderr


def test_artifact_transfer_and_fallback_have_deadlines(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    calls = _host_events(env)
    transfer = next(e for e in calls if e[0] == "curl")
    for flag, value in (
        ("--max-time", "600"),
        ("--speed-limit", "1048576"),
        ("--speed-time", "60"),
    ):
        assert transfer[transfer.index(flag) + 1] == value
    limits = [int(e[2]) for e in calls if e[0] == "timeout"]
    assert len(limits) == 2
    assert sum(limits) <= 2400
    # Each optional stage leaves the rollout reserve inside the 3300 s executionTimeout.
    assert all(limit <= 3300 - ROLLOUT_RESERVE for limit in limits)


# Worst case after the optional stages: Redis wait + app wait + smoke (4 endpoints, each a
# 60 s retry window plus one 20 s attempt) + rollback wait, all in seconds.
WORST_ROLLOUT = 180 + 180 + 4 * (60 + 20) + 180
ROLLOUT_RESERVE = 1200


def test_rollout_reserve_covers_worst_case_rollout_and_leaves_backup_headroom() -> None:
    script = _code(_read("deploy/ssm-deploy.sh"))
    assert script.count(f"- {ROLLOUT_RESERVE}))") == 2  # artifact and host-build budgets
    assert "--wait-timeout 180" in script and "--wait-timeout 180" in _read("deploy/rollback.sh")
    assert "--retry-max-time 60" in _read("deploy/smoke-test.sh")
    assert "--max-time 20" in _read("deploy/smoke-test.sh")
    assert WORST_ROLLOUT + 300 <= ROLLOUT_RESERVE  # at least 5 minutes for backup/provision/migrate
    rollback_doc = _read("deploy/ROLLBACK.md")
    assert f"{ROLLOUT_RESERVE:,} seconds for backup/migration/rollout" in rollback_doc
    assert f"{WORST_ROLLOUT} seconds" in rollback_doc


def test_ssm_sender_sets_execution_timeout_with_longer_poll_window(tmp_path: Path) -> None:
    import json

    _, env = _deploy_host(tmp_path)
    env.update(INSTANCE_ID="local-instance", SSM_POLL_INTERVAL_SECONDS="0")
    aws = Path(env["PATH"].split(":")[0]) / "aws"
    aws.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["HOST_EVENTS"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")
if "send-command" in sys.argv: print("local-command")
elif "Status" in sys.argv: print("Success")
else: print("done")
""")
    result = _host_run(env, "github-actions-ssm-deploy.sh")
    assert result.returncode == 0, result.stderr
    send = _host_events(env)[0]
    params = json.loads(send[send.index("--parameters") + 1])
    assert params["executionTimeout"] == ["3300"]
    assert "export GITHUB_TOKEN='never-log-this-token'" in params["commands"][0]


WORKFLOWS = (("ci-cd.yml", "deploy-dev"), ("deploy-dev.yml", "deploy"))
PINNED_USE = re.compile(r"^[\w.-]+/[\w./-]+@[a-f0-9]{40}$")


def test_every_workflow_action_is_pinned_to_a_commit_sha() -> None:
    import yaml

    for filename, _ in WORKFLOWS:
        workflow = yaml.safe_load(_read(".github/workflows/" + filename))
        uses = [
            step["uses"]
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if "uses" in step
        ]
        assert uses, filename
        for ref in uses:
            assert PINNED_USE.fullmatch(ref), f"{filename}: {ref} is not pinned to a 40-hex SHA"
        text = _read(".github/workflows/" + filename)
        assert "TODO" not in text, filename


def test_build_job_uses_plain_docker_build_with_gated_package_and_upload() -> None:
    import yaml

    for filename, _ in WORKFLOWS:
        text = _read(".github/workflows/" + filename)
        workflow = yaml.safe_load(text)
        builder = workflow["jobs"]["build-images"]
        assert builder["permissions"].get("id-token", "none") == "none"
        assert builder["continue-on-error"] is True
        assert builder["outputs"]["attempt"] == "${{ github.run_attempt }}"
        uses = [step.get("uses", "") for step in builder["steps"]]
        assert not any(u.startswith(("aws-actions/", "docker/")) for u in uses)
        for banned in ("qemu", "buildx", "build-push-action", "type=gha"):
            assert banned not in text.lower(), (filename, banned)
        by_id = {step["id"]: step for step in builder["steps"] if "id" in step}
        images, package, upload = by_id["images"], by_id["package"], by_id["upload"]
        assert images["continue-on-error"] is True and images["timeout-minutes"] == 40
        env_ready = "needs.build-env.outputs.ready == 'success'"
        assert env_ready in images["if"]
        assert "needs.build-env.outputs.platform == 'linux/amd64'" in images["if"]
        keys = list(BUILD_ARGS)
        for step in (images, package):
            for key in keys:
                assert step["env"][key] == "${{ needs.build-env.outputs." + key + " }}"
        # No expression interpolation inside the script: values arrive by environment only.
        assert "${{" not in images["run"]
        assert 'docker build --pull -t "supportchat-ci-backend:$GITHUB_SHA" ./backend' in images["run"]
        assert 'docker build --pull -t "supportchat-ci-frontend:$GITHUB_SHA"' in images["run"]
        for key in keys:
            assert re.search(rf"--build-arg {key}( |\\|$)", images["run"], re.M), key
        assert not re.search(r"--build-arg \w+=", images["run"])
        assert package["continue-on-error"] is True
        assert package["if"] == "steps.images.outcome == 'success'"
        assert upload["continue-on-error"] is True
        assert upload["if"] == (
            "steps.images.outcome == 'success' && steps.package.outcome == 'success'"
        )
        assert upload["uses"].startswith("actions/upload-artifact@")
        assert workflow["jobs"]["build-env"]["steps"][-1]["continue-on-error"] is True


def test_workflow_image_names_match_the_packager() -> None:
    packager = _read("deploy/package-images.sh")
    assert 'backend="supportchat-ci-backend:${GITHUB_SHA}"' in packager
    assert 'frontend="supportchat-ci-frontend:${GITHUB_SHA}"' in packager
    for filename, _ in WORKFLOWS:
        text = _read(".github/workflows/" + filename)
        assert '"supportchat-ci-backend:$GITHUB_SHA"' in text
        assert '"supportchat-ci-frontend:$GITHUB_SHA"' in text


def test_build_env_reports_the_host_platform() -> None:
    assert 'result["platform"] = "linux/amd64" if arch == "x86_64" else "linux/arm64"' in _read(
        "deploy/read-build-env.sh"
    )
    for filename, _ in WORKFLOWS:
        text = _read(".github/workflows/" + filename)
        assert "platform: ${{ steps.public-env.outputs.platform }}" in text


def test_deploy_job_has_aws_credentials_last_and_no_build_actions() -> None:
    import yaml

    for filename, jobname in WORKFLOWS:
        steps = yaml.safe_load(_read(".github/workflows/" + filename))["jobs"][jobname]["steps"]
        assert not any(
            step.get("uses", "").startswith(("docker/", "actions/upload-artifact"))
            for step in steps
        )
        credentials = next(
            i for i, step in enumerate(steps) if step.get("uses", "").startswith("aws-actions/")
        )
        assert credentials == len(steps) - 2


def test_manual_deploy_checks_ci_first_and_deploy_waits_for_that_gate() -> None:
    import yaml

    workflow = yaml.safe_load(_read(".github/workflows/deploy-dev.yml"))
    build_env = workflow["jobs"]["build-env"]
    assert build_env["permissions"]["actions"] == "read"
    assert build_env["permissions"]["checks"] == "read"
    # Checkout provides the script; the gate precedes any AWS credential step.
    first, second = build_env["steps"][:2]
    assert first["uses"].startswith("actions/checkout@")
    assert second["name"] == "Verify CI checks passed"
    assert "continue-on-error" not in second
    assert "${{" not in second["run"]
    deploy = workflow["jobs"]["deploy"]
    assert not any("verify-ci-checks" in step.get("run", "") for step in deploy["steps"])
    assert "build-env" in deploy["needs"]
    assert "needs.build-env.result == 'success'" in deploy["if"]


def _run_deploy_step_injection(tmp_path: Path, filename: str, jobname: str, **extra: str):
    import yaml

    steps = yaml.safe_load(_read(".github/workflows/" + filename))["jobs"][jobname]["steps"]
    repo = tmp_path / (filename + "-".join(extra.values()))
    (repo / "deploy").mkdir(parents=True)
    (repo / "deploy/ssm-deploy.sh").write_text("#!/bin/bash\necho pinned\n")
    (repo / "deploy/github-actions-ssm-deploy.sh").write_text(
        "#!/bin/bash\ncp deploy/ssm-deploy.sh sent-script\n"
    )
    # Use the actual workflow shell with only portable sed replacing GNU sed on macOS.
    bindir = repo / "bin"
    bindir.mkdir()
    sed = bindir / "sed"
    sed.write_text("""#!/usr/bin/env python3
import pathlib, sys
p=pathlib.Path(sys.argv[-1]); s=p.read_text().splitlines(True)
s.insert(1, sys.argv[2][2:].replace("\\\\n", "\\n") + "\\n"); p.write_text("".join(s))
""")
    sed.chmod(0o755)
    result = subprocess.run(
        ["bash", "-ec", steps[-1]["run"]],
        cwd=repo,
        env={
            **os.environ,
            "PATH": str(bindir) + ":" + os.environ["PATH"],
            "GITHUB_RUN_ID": "456",
            "GITHUB_RUN_ATTEMPT": "2",
            **extra,
        },
        capture_output=True,
        text=True,
    )
    sent = repo / "sent-script"
    return result, sent.read_text() if sent.exists() else None


def test_deploy_pins_the_uploading_attempt_not_the_current_one(tmp_path: Path) -> None:
    for filename, jobname in WORKFLOWS:
        # Re-running only the deploy job: this attempt is 2, the artifact came from attempt 1.
        result, sent = _run_deploy_step_injection(
            tmp_path, filename, jobname, CI_UPLOAD_SUCCEEDED="true", CI_ARTIFACT_ATTEMPT="1"
        )
        assert result.returncode == 0, result.stderr
        assert "CI_ARTIFACT_RUN_ID=456" in sent
        assert "CI_ARTIFACT_RUN_ATTEMPT=1" in sent
        assert "CI_ARTIFACT_RUN_ATTEMPT=2" not in sent
        result, sent = _run_deploy_step_injection(
            tmp_path, filename, jobname, CI_UPLOAD_SUCCEEDED="false", CI_ARTIFACT_ATTEMPT="1"
        )
        assert result.returncode == 0, result.stderr
        assert "CI_ARTIFACT" not in sent


def test_deploy_rejects_a_malformed_artifact_attempt(tmp_path: Path) -> None:
    for filename, jobname in WORKFLOWS:
        for bad in ("", "1; touch pwned", "$(id)"):
            result, sent = _run_deploy_step_injection(
                tmp_path, filename, jobname, CI_UPLOAD_SUCCEEDED="true", CI_ARTIFACT_ATTEMPT=bad
            )
            assert result.returncode != 0, (filename, bad)
            assert sent is None


def test_provision_failure_does_not_echo_sql_or_passwords(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    psql = bindir / "psql"
    psql.write_text(
        "#!/bin/sh\necho 'STATEMENT: ALTER ROLE PASSWORD private-password' >&2\nexit 1\n"
    )
    psql.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bindir) + ":" + os.environ["PATH"],
        **{
            key: "private-password"
            for key in (
                "POSTGRES_USER",
                "POSTGRES_PASSWORD",
                "POSTGRES_DB",
                "APP_DB_USER",
                "APP_DB_PASSWORD",
                "MIGRATION_DB_USER",
                "MIGRATION_DB_PASSWORD",
            )
        },
    }
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/provision-postgres-roles.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "provisioning failed" in result.stderr
    assert "STATEMENT" not in result.stderr
    assert "private-password" not in result.stderr


def test_documented_redis_recovery_targets_only_redis(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path)
    doc = _read("deploy/ROLLBACK.md")
    section = doc.split("## Manual Redis recovery", 1)[1].split("## ", 1)[0]
    command = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=env["REPO_DIR"],
        env={**env, "PREVIOUS_COMMIT": "previous-reviewed-commit"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    events = _host_events(env)
    checkout = next(i for i, e in enumerate(events) if e[0] == "git" and "checkout" in e)
    up = next(i for i, e in enumerate(events) if e[:3] == ["docker", "compose", "-f"])
    assert checkout < up
    assert events[checkout][:2] == ["git", "-c"]
    assert Path(events[checkout][2].split("=", 1)[1]).resolve() == Path(env["REPO_DIR"]).resolve()
    assert events[checkout][2].startswith("safe.directory=")
    assert events[checkout][3:] == ["checkout", "--detach", "previous-reviewed-commit"]
    assert events[up] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.prod.yml",
        "up",
        "-d",
        "--no-deps",
        "--no-build",
        "--wait",
        "redis",
    ]
    assert "Do not copy deploy scripts onto the host by hand" in doc
    assert "deliberately uses the previous base Compose" not in doc
    for stale in ("Buildx", "GHA cache", "TODO", "PIN_OFFLINE", "QEMU"):
        assert stale not in doc, stale


def test_dependabot_preserves_postgres_major_version_boundary() -> None:
    import yaml

    updates = yaml.safe_load(_read(".github/dependabot.yml"))["updates"]
    compose = next(u for u in updates if u["package-ecosystem"] == "docker-compose")
    ignored = {item["dependency-name"]: item["update-types"] for item in compose["ignore"]}
    assert ignored["pgvector/pgvector"] == ["version-update:semver-major"]
    assert ignored["postgres"] == ["version-update:semver-major"]
    assert ignored["redis"] == ["version-update:semver-major"]
    docker = next(u for u in updates if u["package-ecosystem"] == "docker")
    ignored_images = {i["dependency-name"]: i["update-types"] for i in docker["ignore"]}
    assert ignored_images["redis"] == ["version-update:semver-major"]
    # Node LTS upgrades are deliberate: runtime image, @types/node and CI move together.
    assert ignored_images["node"] == ["version-update:semver-major"]
    npm = next(u for u in updates if u["package-ecosystem"] == "npm")
    assert {i["dependency-name"]: i["update-types"] for i in npm["ignore"]}["@types/node"] == [
        "version-update:semver-major"
    ]


def test_dependabot_batches_updates_into_few_weekly_pull_requests() -> None:
    import yaml

    updates = yaml.safe_load(_read(".github/dependabot.yml"))["updates"]
    assert {u["package-ecosystem"] for u in updates} == {
        "uv",
        "npm",
        "github-actions",
        "docker",
        "docker-compose",
    }
    for update in updates:
        name = update["package-ecosystem"]
        assert update["schedule"]["interval"] == "weekly", name
        assert update["open-pull-requests-limit"] == 3, name
        assert update["cooldown"]["default-days"] == 5, name
        groups = update["groups"]
        assert len(groups) == 2, name  # one for minor+patch, one for majors
        assert [g for g in groups.values() if g.get("update-types") == ["major"]], name
        others = [g for g in groups.values() if g.get("update-types") != ["major"]]
        assert len(others) == 1, name


def test_ci_package_round_trips_across_docker_image_stores(tmp_path: Path) -> None:
    import json
    import zipfile

    _, env = _deploy_host(tmp_path)
    env.update(
        GITHUB_SHA="a" * 40,
        GITHUB_RUN_ID="12345",
        GITHUB_RUN_ATTEMPT="1",
        RUNNER_TEMP=str(tmp_path),
        **BUILD_ARGS,
    )
    packaged = _host_run(env, "package-images.sh")
    assert packaged.returncode == 0, packaged.stderr
    directory = tmp_path / "supportchat-images"
    metadata = json.loads((directory / "manifest.json").read_text())
    assert metadata["images"]["backend"] == {
        "ref": "supportchat-ci-backend:" + "a" * 40,
        "layers": ["sha256:" + "b" * 64],
    }
    assert metadata["build_args"] == BUILD_ARGS
    artifact = tmp_path / "packaged.zip"
    with zipfile.ZipFile(artifact, "w") as zipped:
        for name in ("images.tar.zst", "images.tar.zst.sha256", "manifest.json"):
            zipped.write(directory / name, name)
    env["ARTIFACT_ZIP"] = str(artifact)
    deployed = _host_run(env)
    assert deployed.returncode == 0, deployed.stderr
    assert _host_state(env)["serving"] == "ci-backend"
    assert not any("build" in event for event in _host_events(env))


def test_loaded_layer_mismatch_falls_back_without_promoting_artifact(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, LAYER_MISMATCH="1")
    _image_artifact(tmp_path, env)
    deployed = _host_run(env)
    assert deployed.returncode == 0, deployed.stderr
    assert "Loaded image identity mismatch" in deployed.stderr
    assert _host_state(env)["serving"] == "new-backend"
    assert not list(Path(env["IMAGE_ARCHIVE_DIR"]).glob("*.tar.zst"))


def test_no_uploaded_artifact_uses_host_build_without_github_download(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, CI_ARTIFACT_RUN_ID="")
    deployed = _host_run(env)
    assert deployed.returncode == 0, deployed.stderr
    assert _host_state(env)["serving"] == "new-backend"
    assert _host_state(env)["migrations"] == 1
    assert not any("api.github.com" in arg for event in _host_events(env) for arg in event)


def test_compose_config_build_args_prints_the_seven_resolved_keys(tmp_path: Path) -> None:
    import json

    repo, env = _deploy_host(tmp_path)
    result = subprocess.run(
        ["python3", "deploy/compose-config.py", "build-args"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == BUILD_ARGS
    assert "private-" not in result.stdout + result.stderr


def test_stale_build_args_fall_back_to_host_build_without_promoting_artifact(
    tmp_path: Path,
) -> None:
    _, env = _deploy_host(tmp_path)
    _image_artifact(
        tmp_path,
        env,
        build_args={**BUILD_ARGS, "NEXT_PUBLIC_STAFF_APP_ORIGIN": "https://old-staff.example.test"},
    )
    loader = _host_run(env, "load-ci-images.sh")
    assert loader.returncode == 1
    assert "build arguments differ" in loader.stderr
    assert not any(event[:2] == ["docker", "load"] for event in _host_events(env))
    deployed = _host_run(env)
    assert deployed.returncode == 0, deployed.stderr
    assert "building on host" in deployed.stderr
    assert _host_state(env)["serving"] == "new-backend"
    assert _host_state(env)["migrations"] == 1
    assert not list(Path(env["IMAGE_ARCHIVE_DIR"]).glob("*.tar.zst"))


def test_artifact_without_recorded_build_args_is_not_trusted(tmp_path: Path) -> None:
    import json
    import zipfile

    _, env = _deploy_host(tmp_path)
    _image_artifact(tmp_path, env)
    source = Path(env["ARTIFACT_ZIP"])
    stripped = tmp_path / "stripped.zip"
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(stripped, "w") as dst:
        for item in src.namelist():
            data = src.read(item)
            if item == "manifest.json":
                manifest = json.loads(data)
                del manifest["build_args"]
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    env["ARTIFACT_ZIP"] = str(stripped)
    deployed = _host_run(env)
    assert deployed.returncode == 0, deployed.stderr
    assert _host_state(env)["serving"] == "new-backend"


def test_loaded_image_architecture_mismatch_falls_back_to_host_build(tmp_path: Path) -> None:
    _, env = _deploy_host(tmp_path, LOADED_ARCH="riscv64")
    _image_artifact(tmp_path, env)
    deployed = _host_run(env)
    assert deployed.returncode == 0, deployed.stderr
    assert "architecture riscv64 does not match host" in deployed.stderr
    assert any(event[:2] == ["docker", "load"] for event in _host_events(env))
    assert _host_state(env)["serving"] == "new-backend"
    assert not list(Path(env["IMAGE_ARCHIVE_DIR"]).glob("*.tar.zst"))
    # The mismatched image is never aliased to the Compose image names.
    assert "support-chat-backend:latest" not in {
        key for key, value in _host_state(env)["tags"].items() if value == "ci-backend"
    }


def test_preflight_image_reserve_only_warns_while_backup_reserve_fails_hard(
    tmp_path: Path,
) -> None:
    # Stubbed backup estimate: 1 MiB + 1 GiB = 1049600 KiB. Image reserve defaults to 12 GiB.
    _, env = _deploy_host(tmp_path, FREE_KB="2000000")
    result = _host_run(env, "preflight.sh")
    assert result.returncode == 0, result.stderr
    assert "WARN: low free disk" in result.stderr
    assert "Insufficient free disk" not in result.stderr
    assert "Preflight passed" in result.stdout

    (tmp_path / "enough").mkdir()
    _, env = _deploy_host(tmp_path / "enough", FREE_KB="999999998")
    result = _host_run(env, "preflight.sh")
    assert result.returncode == 0, result.stderr
    assert "WARN: low free disk" not in result.stderr

    (tmp_path / "short").mkdir()
    _, env = _deploy_host(tmp_path / "short", FREE_KB="1000000")
    result = _host_run(env, "preflight.sh")
    assert result.returncode != 0
    assert "Insufficient free disk" in result.stderr
    assert "Preflight passed" not in result.stdout
