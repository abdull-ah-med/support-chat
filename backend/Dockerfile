# syntax=docker/dockerfile:1

# TODO(INF-11): pin by digest once a build against it has been verified. Index digest seen
# 2026-10-10: ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58
# Dependabot (docker ecosystem) keeps a pinned digest current.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    HOME=/home/app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Browsers depend only on the locked dependencies, so install them before the app source:
# an app code change must not re-download Chromium and rebuild this multi-GB layer.
RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app \
    && playwright install-deps chromium \
    && playwright install chromium \
    && crawl4ai-setup \
    && chown -R app:app /home/app /ms-playwright

# tiktoken counts tokens before every query embedding and downloads its encoding on first use.
# Fetching it at build time keeps that download (several seconds, on the event loop) out of the
# first visitor turn of every new container and removes a runtime dependency on the download host.
# After the browser layer, so this does not invalidate it.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p /opt/tiktoken \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chown -R app:app /opt/tiktoken

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
# Only the scripts the running system needs. Dev/eval tooling (harness_chat, live_bot_probe,
# run_site_eval, docker_up.sh) stays out of the image. The purge_* modules are imported by
# app/workers/maintenance.py; seed_user and generate_push_keys are run by operators with
# `docker compose exec`. tests/test_deploy_hardening.py fails if app/ imports a script not copied here.
COPY scripts/__init__.py scripts/purge_expired_chats.py scripts/purge_expired_logs.py \
     scripts/purge_expired_refresh_tokens.py scripts/seed_user.py scripts/generate_push_keys.py ./scripts/
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/migrate.sh ./docker/migrate.sh

RUN chmod +x /entrypoint.sh /app/docker/migrate.sh \
    && uv sync --frozen --no-dev \
    && chown -R app:app /app

USER app

EXPOSE 8000

# Same probe as the compose healthcheck: send the configured staff Host so the production
# host check accepts the request. The worker and migrate services override or disable this.
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=12 \
    CMD ["python", "-c", "from urllib.parse import urlsplit; import urllib.request; from app.settings import get_settings; host=urlsplit(get_settings().staff_app_origin).netloc; request=urllib.request.Request('http://127.0.0.1:8000/health', headers={'Host': host}); urllib.request.urlopen(request, timeout=4)"]

ENTRYPOINT ["/entrypoint.sh"]
# SEC-3: cap a WebSocket message at 64 KiB while it is read. uvicorn's default is 16 MiB, which
# would be buffered in full before the application's own frame-size check ever ran.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "65536"]
