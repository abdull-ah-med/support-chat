"""The crawler identifies the product, not any one site, and is configurable."""

from app.services.kb_crawl import crawler_user_agent, robots_allows
from app.settings import reset_settings_cache


def test_default_user_agent_names_no_site(monkeypatch) -> None:
    monkeypatch.delenv("KB_CRAWLER_USER_AGENT", raising=False)
    monkeypatch.delenv("KB_CRAWLER_CONTACT_URL", raising=False)
    reset_settings_cache()
    assert crawler_user_agent() == "SupportChatBot/1.0"


def test_user_agent_and_contact_come_from_settings(monkeypatch) -> None:
    monkeypatch.setenv("KB_CRAWLER_USER_AGENT", "AcmeBot/2.0")
    monkeypatch.setenv("KB_CRAWLER_CONTACT_URL", "https://acme.example/bot")
    reset_settings_cache()
    assert crawler_user_agent() == "AcmeBot/2.0 (+https://acme.example/bot)"
    robots = "User-agent: AcmeBot\nDisallow: /private\n"
    assert robots_allows("https://x.example/private", robots) is False
    assert robots_allows("https://x.example/open", robots) is True
