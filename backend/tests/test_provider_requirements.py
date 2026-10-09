"""No fakes in production: a missing provider is a startup error, never a silent stand-in."""

import asyncio
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.kb_embedder import (
    EmbedderNotConfiguredError,
    OpenAIEmbedder,
    default_embedder,
)
from app.settings import Settings, require_provider_credentials, reset_settings_cache
from app.workers import entrypoint

APP_ROOT = Path(__file__).resolve().parents[1] / "app"
SECRETS = {
    "database_url": "postgresql://chat:chat@127.0.0.1:55432/support_chat_test",
    "jwt_secret": "a8f3c1e9b2d64750c4a1f8e3b6d9027c5e1a4b8f3c6d9e2a7b0c5d8e1f4a7b3c",
    "widget_token_secret": "d1e4a7b0c3f6d9e2a5b8c1d4e7f0a3b6c9d2e5f8a1b4c7d0e3f6a9b2c5d8e1f4",
    "rate_key_secret": "9c2f5a8d1e4b7c0f3a6d9e2b5c8f1a4d7e0b3c6f9a2d5e8b1c4f7a0d3e6b9c2f",
}


def _settings(app_env: str, anthropic: str | None, openai: str | None) -> Settings:
    return Settings(**SECRETS, app_env=app_env, anthropic_api_key=anthropic, openai_api_key=openai)


def test_a_test_run_may_start_without_provider_keys() -> None:
    require_provider_credentials(_settings("test", None, None))


def test_a_real_environment_refuses_to_start_without_both_keys() -> None:
    # Production additionally validates key shape in Settings itself (test_production_settings).
    with pytest.raises(RuntimeError) as error:
        require_provider_credentials(_settings("local", None, None))
    assert "ANTHROPIC_API_KEY" in str(error.value)
    assert "OPENAI_API_KEY" in str(error.value)


def test_the_error_names_only_what_is_missing() -> None:
    with pytest.raises(RuntimeError) as error:
        require_provider_credentials(_settings("local", "sk-ant-real-looking-key", None))
    assert "OPENAI_API_KEY" in str(error.value)
    assert "ANTHROPIC_API_KEY" not in str(error.value)


def test_a_blank_key_is_a_missing_key() -> None:
    with pytest.raises(RuntimeError):
        require_provider_credentials(_settings("local", "   ", "sk-real-looking-key"))


def test_both_keys_let_a_local_process_start() -> None:
    require_provider_credentials(_settings("local", "sk-ant-real-looking-key", "sk-real-key"))


def _without_keys(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    reset_settings_cache()


def test_the_api_refuses_to_start_without_provider_keys(monkeypatch) -> None:
    _without_keys(monkeypatch)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        with TestClient(create_app()):
            pass


def test_the_worker_refuses_to_start_without_provider_keys(monkeypatch) -> None:
    _without_keys(monkeypatch)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.run(entrypoint._run())


def test_the_default_embedder_is_never_a_fake(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    reset_settings_cache()
    with pytest.raises(EmbedderNotConfiguredError):
        default_embedder()
    with pytest.raises(EmbedderNotConfiguredError):
        OpenAIEmbedder("")


def test_production_code_contains_no_fake_embedder_or_test_detection() -> None:
    forbidden = re.compile(r"FakeEmbedder|PYTEST_CURRENT_TEST|axis_for_text|fake:axes")
    offenders = [
        str(path.relative_to(APP_ROOT))
        for path in APP_ROOT.rglob("*.py")
        if forbidden.search(path.read_text())
    ]
    assert offenders == []
