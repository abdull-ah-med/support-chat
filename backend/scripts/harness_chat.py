"""TEMPORARY CLI over the local chatbot harness (/api/internal/dev/harness/*).

    uv run python -m scripts.harness_chat sites
    uv run python -m scripts.harness_chat new --site KEY [--name N --email E]
    uv run python -m scripts.harness_chat say CONV_ID "text" [--full]
    uv run python -m scripts.harness_chat show CONV_ID
    uv run python -m scripts.harness_chat stats

Server: HARNESS_API (default http://127.0.0.1:8010). Staff login from env
HARNESS_EMAIL/HARNESS_PASSWORD, else UI_AUDIT_EMAIL/UI_AUDIT_PASSWORD from
backend/.env.ui-audit. `say` prints a compact turn; --full adds the raw trace.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

API = os.environ.get("HARNESS_API", "http://127.0.0.1:8010")
ORIGIN = os.environ.get("STAFF_APP_ORIGIN", "http://localhost:3000")
ENV_FILE = Path(__file__).resolve().parents[1] / ".env.ui-audit"
BASE = "/api/internal/dev/harness"


def _credentials() -> tuple[str, str]:
    email = os.environ.get("HARNESS_EMAIL") or os.environ.get("UI_AUDIT_EMAIL")
    password = os.environ.get("HARNESS_PASSWORD") or os.environ.get("UI_AUDIT_PASSWORD")
    if (not email or not password) and ENV_FILE.exists():
        values: dict[str, str] = {}
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
        email = email or values.get("UI_AUDIT_EMAIL")
        password = password or values.get("UI_AUDIT_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "no staff credentials: set HARNESS_EMAIL/HARNESS_PASSWORD or fill .env.ui-audit"
        )
    return email, password


def _client() -> httpx.Client:
    client = httpx.Client(base_url=API, timeout=150.0)
    email, password = _credentials()
    login = client.post(
        "/auth/login", headers={"Origin": ORIGIN}, json={"email": email, "password": password}
    )
    if login.status_code != 200:
        raise SystemExit(f"login failed: HTTP {login.status_code}")
    client.headers.update(
        {"Authorization": f"Bearer {login.json()['access_token']}", "Origin": ORIGIN}
    )
    return client


def _out(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=None, default=str))


def _call(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, BASE + path, **kwargs)
    if response.status_code >= 400:
        raise SystemExit(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def _compact(turn: dict[str, Any]) -> dict[str, Any]:
    tel = turn["telemetry"]
    return {
        "reply": turn["reply_text"],
        "state": turn["state"],
        "intent": turn["intent"],
        "outcome": turn["outcome"],
        "reason_code": turn["reason_code"],
        "citations": [
            {k: c[k] for k in ("chunk_id", "source_url", "cited_text", "belongs_to_site")}
            for c in turn["citations"]
        ],
        "handoff_opened": turn["handoff_opened"],
        "knowledge_gap_recorded": turn["knowledge_gap_recorded"],
        "latency_ms": turn["latency_ms"],
        "stage_timings": tel["stage_timings"],
        "llm_calls": [
            {
                k: c.get(k)
                for k in (
                    "stage",
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "elapsed_ms",
                    "error",
                )
            }
            for c in tel["llm_calls"]
        ],
        "retrieval": {k: tel["retrieval"].get(k) for k in ("mode", "hit_count", "best_cosine")},
        "retrieval_hits": [
            {k: h.get(k) for k in ("id", "cosine", "score", "heading")}
            for h in tel["retrieval"].get("hits", [])
        ],
        "understanding": tel["understanding"],
        "validation": tel["validation"],
        "errors": tel["errors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new")
    new.add_argument("--site", required=True)
    new.add_argument("--name")
    new.add_argument("--email")
    say = sub.add_parser("say")
    say.add_argument("conversation_id")
    say.add_argument("text")
    say.add_argument("--full", action="store_true", help="print the whole turn incl. raw trace")
    show = sub.add_parser("show")
    show.add_argument("conversation_id")
    sub.add_parser("stats")
    sub.add_parser("sites")
    args = parser.parse_args()

    with contextlib.nullcontext(_client()) as client:
        if args.cmd == "new":
            body = {"site_key": args.site, "visitor_name": args.name, "visitor_email": args.email}
            _out(_call(client, "POST", "/conversations", json={k: v for k, v in body.items() if v}))
        elif args.cmd == "say":
            turn = _call(
                client,
                "POST",
                f"/conversations/{args.conversation_id}/messages",
                json={"text": args.text},
            )
            _out(turn if args.full else _compact(turn))
        elif args.cmd == "show":
            data = _call(client, "GET", f"/conversations/{args.conversation_id}")
            data["turns"] = [_compact(t) for t in data["turns"]]
            _out(data)
        elif args.cmd == "stats":
            _out(_call(client, "GET", "/stats"))
        else:
            _out(_call(client, "GET", "/sites"))


if __name__ == "__main__":
    sys.exit(main())
