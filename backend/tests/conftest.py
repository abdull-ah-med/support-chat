import os
from collections.abc import Generator
from urllib.parse import urlparse

import pytest

DEFAULT_TEST_REDIS_URL = "redis://:local-dev-redis@127.0.0.1:56379/15"
# Serial runs use Redis db 15. pytest-xdist worker gwN uses db (14 - N), so a concurrent serial
# run never shares a logical DB with a worker. At most 14 workers (dbs 14..1); db 0 is never used.
SERIAL_REDIS_DB = 15
MAX_XDIST_WORKERS = 14
XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")


def _xdist_worker_index() -> int | None:
    if not XDIST_WORKER or not XDIST_WORKER.startswith("gw"):
        return None
    index = int(XDIST_WORKER[2:])
    if index >= MAX_XDIST_WORKERS:
        raise RuntimeError(f"At most {MAX_XDIST_WORKERS} xdist workers are supported (-n 14)")
    return index


def _worker_database_url(base_url: str) -> str:
    """Each xdist worker gets its own database: <base>_gw0, <base>_gw1, ..."""
    if XDIST_WORKER is None:
        return base_url
    return f"{base_url}_{XDIST_WORKER}"


def _worker_redis_url(base_url: str) -> str:
    index = _xdist_worker_index()
    if index is None:
        return base_url
    parsed = urlparse(base_url)
    return parsed._replace(path=f"/{SERIAL_REDIS_DB - 1 - index}").geturl()


# conftest is imported twice (as `conftest` and `tests.conftest`), and DATABASE_URL is rewritten
# below, so remember the configured base once.
_BASE_DATABASE_URL = os.environ.setdefault(
    "_TEST_BASE_DATABASE_URL",
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://chat:chat@127.0.0.1:55432/support_chat_test",
    ),
)
TEST_DATABASE_URL = _worker_database_url(_BASE_DATABASE_URL)
TEST_DB_NAME = TEST_DATABASE_URL.rsplit("/", 1)[-1]

# Exported unconditionally so subprocess servers and helpers that read the environment at
# call time (tests/notification_e2e_helpers.py) see this worker's database and Redis.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["REDIS_URL"] = _worker_redis_url(os.environ.get("REDIS_URL", DEFAULT_TEST_REDIS_URL))
os.environ.setdefault("JWT_SECRET", "t" * 64)
os.environ.setdefault("WIDGET_TOKEN_SECRET", "w" * 64)
os.environ.setdefault("RATE_KEY_SECRET", "r" * 64)
os.environ.setdefault(
    "WIDGET_CSP_SERVICE_SECRET",
    "e1a4b7c0d3f6a9b2c5d8e1f4a7b0c3d6e9f2a5b8c1d4e7f0a3b6c9d2e5f8a1b4c",
)
os.environ.setdefault("COOKIE_SECURE", "false")
os.environ.setdefault("STAFF_APP_ORIGIN", "http://localhost:3000")
os.environ.setdefault("WIDGET_ORIGIN", "http://widget.localhost:3000")
os.environ.setdefault("KB_INGEST_HOST_DELAY_MS", "0")
os.environ.setdefault("INTERNAL_EVAL_ENABLED", "false")


def _assert_test_database() -> None:
    if "test" not in TEST_DB_NAME:
        raise RuntimeError("Refusing to run tests against a non-test database")


def _admin_url() -> str:
    return TEST_DATABASE_URL.replace("+asyncpg", "").rsplit("/", 1)[0] + "/postgres"


def _ensure_test_database() -> None:
    import psycopg

    _assert_test_database()
    with psycopg.connect(_admin_url(), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (TEST_DB_NAME,),
        ).fetchone()
        if exists is None:
            conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')


def _postgres_is_up() -> bool:
    import psycopg

    try:
        with psycopg.connect(_admin_url(), connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def pytest_sessionstart(session: pytest.Session) -> None:
    _assert_test_database()
    if not _postgres_is_up():
        if os.environ.get("CI"):
            raise RuntimeError("Postgres is required in CI")
        if os.environ.get("ALLOW_DB_TEST_SKIP") != "1":
            session.exitstatus = 1
            pytest.exit(
                "Postgres is required. Set ALLOW_DB_TEST_SKIP=1 to skip the backend suite.",
                returncode=1,
            )
        session.exitstatus = 0
        pytest.exit("Postgres is down; skipping backend suite", returncode=0)
    _ensure_test_database()


_sync_engine = None
_truncate_sql: str | None = None
_redis_client = None


def _truncate_tables() -> None:
    """One TRUNCATE statement over every app table, on a reused sync engine."""
    global _sync_engine, _truncate_sql
    from sqlalchemy import create_engine, text

    from app.models import Base

    if _sync_engine is None:
        sync_url = TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg://")
        _sync_engine = create_engine(sync_url)
        names = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
        _truncate_sql = f"TRUNCATE {names} RESTART IDENTITY CASCADE" if names else ""
    if _truncate_sql:
        with _sync_engine.begin() as conn:
            conn.execute(text(_truncate_sql))


def _assert_test_redis_url(url: str) -> None:
    parsed = urlparse(url)
    path = parsed.path or "/"
    db_index = path.rstrip("/").rsplit("/", 1)[-1] if path not in {"", "/"} else "0"
    host = (parsed.hostname or "").lower()

    if db_index == str(SERIAL_REDIS_DB):
        return
    if "test" in host or "test" in path:
        return
    if (
        XDIST_WORKER
        and db_index.isdigit()
        and SERIAL_REDIS_DB - MAX_XDIST_WORKERS <= int(db_index) < SERIAL_REDIS_DB
    ):
        return

    raise RuntimeError(
        "Refusing to FLUSHDB on non-test Redis URL "
        "(use database index 15 or a host/path marked for tests)"
    )


def _flush_redis() -> None:
    global _redis_client
    import redis as redis_sync

    if _redis_client is None:
        url = os.environ.get("REDIS_URL", DEFAULT_TEST_REDIS_URL)
        _assert_test_redis_url(url)
        _redis_client = redis_sync.Redis.from_url(url)
    _redis_client.flushdb()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _sync_engine, _redis_client
    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None
    if _redis_client is not None:
        _redis_client.close()
        _redis_client = None


@pytest.fixture(scope="session")
def _schema_at_head() -> None:
    """Run Alembic once per session (per xdist worker database)."""
    from alembic.config import Config

    from alembic import command

    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    from app.settings import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _disable_ingest_worker(monkeypatch) -> None:
    async def _noop() -> None:
        return None

    monkeypatch.setattr("app.main.start_kb_workers", _noop)
    monkeypatch.setattr("app.workers.start_kb_workers", _noop)
    monkeypatch.setattr("app.workers.kb_ingest_worker.start_ingest_worker", _noop)


@pytest.fixture(autouse=True)
def _blank_anthropic_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")


@pytest.fixture
def migrated_db(_schema_at_head) -> Generator:
    from app.db import reset_engine
    from app.redis import reset_redis

    reset_engine()
    reset_redis()
    _flush_redis()
    _truncate_tables()
    yield
    reset_engine()
    reset_redis()


@pytest.fixture
def client(_schema_at_head) -> Generator:
    from fastapi.testclient import TestClient

    from app.chat.connection_manager import connection_manager
    from app.db import reset_engine
    from app.main import app
    from app.redis import reset_redis

    reset_engine()
    connection_manager.reset()
    reset_redis()
    _flush_redis()
    _truncate_tables()
    with TestClient(app) as test_client:
        yield test_client
    reset_engine()
    reset_redis()
    connection_manager.reset()
