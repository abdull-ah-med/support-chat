import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NGINX_CONFIG = ROOT / "deploy" / "nginx.conf.template"

PROXY_HEADERS = (
    "proxy_set_header Host $host;",
    "proxy_set_header X-Real-IP $remote_addr;",
    "proxy_set_header X-Forwarded-For $remote_addr;",
    "proxy_set_header X-Forwarded-Proto $scheme;",
)


def _config() -> str:
    return NGINX_CONFIG.read_text(encoding="utf-8")


def _location_block(config: str, header: str) -> str:
    start = config.index(header)
    return config[start : config.index("}", start)]


def test_public_nginx_denies_internal_api_before_general_api_proxy() -> None:
    config = _config()

    internal = "location ^~ /api/internal/"
    public_api = "location /api/"
    assert internal in config
    assert config.index(internal) < config.index(public_api)
    internal_block = config[config.index(internal) : config.index(public_api)]
    assert "return 404;" in internal_block


def test_nginx_rate_limits_login_and_widget_bootstrap() -> None:
    config = _config()
    assert "limit_req_zone $binary_remote_addr zone=auth_login:10m rate=10r/m;" in config
    assert "limit_req_zone $binary_remote_addr zone=bootstrap:10m rate=60r/m;" in config
    login = _location_block(config, "location = /auth/login {")
    assert "limit_req zone=auth_login burst=5 nodelay;" in login
    assert "client_max_body_size 4k;" in login
    widget_start = config.rindex("location = /api/public/widget-bootstrap {")
    widget_bootstrap = config[widget_start : config.index("}", widget_start)]
    assert "limit_req zone=bootstrap burst=20 nodelay;" in widget_bootstrap
    assert "proxy_pass http://127.0.0.1:8000;" in widget_bootstrap


def test_session_upkeep_endpoints_use_a_looser_bucket_than_login() -> None:
    config = _config()
    rate = {
        zone: int(per_min)
        for zone, per_min in re.findall(r"zone=(auth_\w+):\d+m rate=(\d+)r/m", config)
    }
    assert rate["auth_session"] >= 6 * rate["auth_login"]
    upkeep = _location_block(config, "location ^~ /auth/ {")
    assert "zone=auth_session" in upkeep
    assert "zone=auth_login" not in upkeep
    # Only credential checks share the strict bucket.
    assert config.count("zone=auth_login burst") == 2
    assert "limit_req zone=auth_login" in _location_block(
        config, "location = /auth/change-password {"
    )


def test_rate_limit_rejections_are_429_not_503() -> None:
    config = _config()
    assert "limit_req_status 429;" in config
    assert "limit_conn_status 429;" in config


def test_websocket_locations_cap_connections_and_upgrade_rate_per_ip() -> None:
    config = _config()
    assert "limit_conn_zone $binary_remote_addr zone=ws_conn:10m;" in config
    assert "limit_req_zone $binary_remote_addr zone=ws_upgrade:" in config
    for header in ("location = /ws/visitor {", "location = /ws/agent {"):
        block = _location_block(config, header)
        assert re.search(r"limit_conn ws_conn \d+;", block), header
        assert "limit_req zone=ws_upgrade" in block, header


def test_every_upgrade_location_repeats_the_forwarding_headers() -> None:
    config = _config()
    blocks = re.findall(r"location [^{]+\{[^}]*\}", config)
    upgrade = [b for b in blocks if "proxy_set_header Upgrade" in b]
    assert len(upgrade) == 3  # /ws/agent, staff `location /`, /ws/visitor
    for block in upgrade:
        for header in PROXY_HEADERS:
            assert header in block, f"{header} missing in {block.splitlines()[0]}"


def test_unknown_hosts_are_dropped_and_server_version_is_hidden() -> None:
    config = _config()
    assert "server_tokens off;" in config
    plain = _location_block(config, "listen 80 default_server;")
    assert "return 444;" in plain
    tls = _location_block(config, "listen 443 ssl default_server;")
    assert "ssl_reject_handshake on;" in tls
    # The catch-all precedes every named server so it is also the first match.
    assert config.index("listen 80 default_server;") < config.index(
        "server_name __MARKETING_HOST__;"
    )


def _nginx_host(tmp_path, **scenario):
    from tests.test_deploy_hardening import _deploy_host

    repo, env = _deploy_host(tmp_path, **scenario)
    nginx_root = tmp_path / "nginx"
    (nginx_root / "conf.d").mkdir(parents=True)
    (nginx_root / "nginx.conf").write_text(
        f"events {{}}\nhttp {{ include {nginx_root}/conf.d/*.conf; }}\n"
    )
    dest = nginx_root / "conf.d/support-chat.conf"
    dest.write_text("# old nginx config\n")
    cert_root = tmp_path / "letsencrypt/live"
    bundle = cert_root / "marketing.example.test"
    bundle.mkdir(parents=True)
    (bundle / "fullchain.pem").write_text("test-certificate")
    (bundle / "privkey.pem").write_text("test-private-key")
    env.update(
        NGINX_ROOT=str(nginx_root),
        NGINX_CONF_DEST=str(dest),
        LETSENCRYPT_LIVE_ROOT=str(cert_root),
        NGINX_SWITCH_MARKER=str(tmp_path / "switched"),
    )
    # Only the OS boundary is stubbed; candidate/active/previous files are real.
    (Path(env["PATH"].split(":")[0]) / "nginx").write_text(
        '#!/bin/sh\n[ "${NGINX_INVALID:-}" != 1 ]\n'
    )
    (Path(env["PATH"].split(":")[0]) / "systemctl").write_text(
        '#!/bin/sh\nif [ "${RELOAD_FAIL:-}" = 1 ] && [ ! -f "$NGINX_ROOT/failed" ]; then\n'
        'touch "$NGINX_ROOT/failed"\nexit 1\nfi\n'
    )
    return repo, env, dest, bundle


def test_nginx_invalid_candidate_preserves_active_config(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path, NGINX_INVALID="1")
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode != 0
    assert dest.read_text() == "# old nginx config\n"
    assert not Path(env["NGINX_SWITCH_MARKER"]).exists()


def test_nginx_reload_failure_restores_previous_config(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path, RELOAD_FAIL="1")
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode != 0
    assert dest.read_text() == "# old nginx config\n"
    assert (dest.parent.parent / "failed").exists()


def test_nginx_missing_cert_skips_switch_with_warning(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, bundle = _nginx_host(tmp_path)
    (bundle / "privkey.pem").unlink()
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode == 0, result.stderr
    assert "WARN" in result.stderr
    assert dest.read_text() == "# old nginx config\n"


def test_nginx_success_keeps_tls_and_previous_config(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, bundle = _nginx_host(tmp_path)
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode == 0, result.stderr
    assert f"ssl_certificate {bundle}/fullchain.pem;" in dest.read_text()
    assert f"ssl_certificate_key {bundle}/privkey.pem;" in dest.read_text()
    assert Path(str(dest) + ".previous").read_text() == "# old nginx config\n"
    assert Path(env["NGINX_SWITCH_MARKER"]).exists()


def test_smoke_failure_rolls_back_apps_without_reverting_valid_nginx(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run, _host_state

    _, env, dest, _ = _nginx_host(tmp_path, SMOKE_FAIL="1", ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode != 0
    assert "server_name staff.example.test;" in dest.read_text()
    assert _host_state(env)["serving"] == "old-backend"
    assert _host_state(env)["migrations"] == 1


def test_invalid_nginx_in_preflight_warns_without_app_rollback(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run, _host_state

    _, env, dest, _ = _nginx_host(tmp_path, NGINX_INVALID="1", ARTIFACT_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert "WARN" in result.stderr
    assert dest.read_text() == "# old nginx config\n"
    assert _host_state(env)["serving"] == "new-backend"
    assert _host_state(env)["migrations"] == 1


def test_nginx_candidate_is_checked_before_backup_and_migrate(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path, ARTIFACT_FAIL="1")
    # Fail validation if backup has already started. This script only represents nginx.
    (Path(env["PATH"].split(":")[0]) / "nginx").write_text(
        '#!/bin/sh\n[ "$2" = -c ] || exit 0  # only the staged-candidate test is time-sensitive\n'
        'if [ -d "$BACKUP_DIR" ]; then echo late-validation >&2; exit 1; fi\n'
        'touch "$NGINX_ROOT/validated"\n'
    )
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert (dest.parent.parent / "validated").exists()
    assert "late-validation" not in result.stderr
    assert "server_name staff.example.test;" in dest.read_text()


def test_nginx_staging_ignores_dangling_links(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path, ARTIFACT_FAIL="1")
    (dest.parent / "unused.conf").symlink_to(dest.parent.parent / "missing.conf")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert "server_name staff.example.test;" in dest.read_text()


def test_nginx_reload_failure_does_not_rollback_healthy_apps(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run, _host_state

    _, env, dest, _ = _nginx_host(tmp_path, ARTIFACT_FAIL="1", RELOAD_FAIL="1")
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert dest.read_text() == "# old nginx config\n"
    assert "WARN" in result.stderr
    assert _host_state(env)["serving"] == "new-backend"


def test_nginx_staging_preserves_valid_relative_symlinks(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path, ARTIFACT_FAIL="1")
    root = dest.parent.parent
    (root / "sites-enabled").mkdir()
    (root / "sites-available").mkdir()
    (root / "sites-available/extra.conf").write_text("# valid extra host config\n")
    (root / "sites-enabled/extra.conf").symlink_to("../sites-available/extra.conf")
    (Path(env["PATH"].split(":")[0]) / "nginx").write_text(
        '#!/bin/sh\n[ "$2" = -c ] || exit 0\n[ -f "$(dirname "$3")/sites-enabled/extra.conf" ]\n'
    )
    result = _host_run(env)
    assert result.returncode == 0, result.stderr
    assert "server_name staff.example.test;" in dest.read_text()


def _logging_nginx(env, tmp_path, installed_fails_once=False):
    """nginx/systemctl stubs that record every call in order, plus the staged config text."""
    log = tmp_path / "nginx-calls.log"
    bindir = Path(env["PATH"].split(":")[0])
    (bindir / "nginx").write_text(
        "#!/bin/sh\n"
        'echo "nginx $*" >> "$NGINX_LOG"\n'
        'if [ "$2" = -c ]; then cp "$3" "$NGINX_STAGED_COPY"; exit 0; fi\n'
        'if [ "${INSTALLED_FAILS_ONCE:-}" = 1 ] && [ ! -f "$NGINX_ROOT/installed-failed" ]; then\n'
        '  touch "$NGINX_ROOT/installed-failed"; exit 1\nfi\n'
    )
    (bindir / "systemctl").write_text('#!/bin/sh\necho "systemctl $*" >> "$NGINX_LOG"\n')
    env.update(
        NGINX_LOG=str(log),
        NGINX_STAGED_COPY=str(tmp_path / "staged-nginx.conf"),
        INSTALLED_FAILS_ONCE="1" if installed_fails_once else "",
    )
    return log


def test_nginx_candidate_is_validated_with_c_only_and_points_into_the_staged_tree(
    tmp_path,
) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path)
    log = _logging_nginx(env, tmp_path)
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    candidate_calls = [c for c in calls if " -c " in c]
    assert len(candidate_calls) == 1
    assert not any(" -p " in c or c.endswith(" -p") for c in calls)
    parts = candidate_calls[0].split()
    assert parts[:3] == ["nginx", "-t", "-c"] and len(parts) == 4
    staged_conf = Path(parts[3])
    assert staged_conf.name == "nginx.conf" and staged_conf.parent.name == "nginx"
    assert staged_conf.parent.resolve() != dest.parent.parent.resolve()  # a copy, not the live tree
    # Absolute includes under NGINX_ROOT now point at the staged copy, not the live tree.
    staged_text = Path(env["NGINX_STAGED_COPY"]).read_text()
    assert f"include {staged_conf.parent}/conf.d/*.conf;" in staged_text
    assert f"include {dest.parent}/" not in staged_text


def test_nginx_installed_config_is_tested_before_reload(tmp_path) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path)
    log = _logging_nginx(env, tmp_path)
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode == 0, result.stderr
    calls = [c for c in log.read_text().splitlines()]
    assert calls[1:] == ["nginx -t", "systemctl reload nginx"]
    assert "server_name staff.example.test;" in dest.read_text()


def test_nginx_installed_config_test_failure_restores_previous_without_reloading_it(
    tmp_path,
) -> None:
    from tests.test_deploy_hardening import _host_run

    _, env, dest, _ = _nginx_host(tmp_path)
    log = _logging_nginx(env, tmp_path, installed_fails_once=True)
    result = _host_run(env, "update-nginx.sh")
    assert result.returncode != 0
    assert dest.read_text() == "# old nginx config\n"
    calls = log.read_text().splitlines()
    # Failing test of the installed file -> restore -> test restored file -> single reload.
    assert calls[1:] == ["nginx -t", "nginx -t", "systemctl reload nginx"]
    assert not Path(env["NGINX_SWITCH_MARKER"]).exists()
