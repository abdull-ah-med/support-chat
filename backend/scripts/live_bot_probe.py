"""Drive real multi-turn chats against the local /api/internal/dev workbench for one site.

Local only. Talks to the running API (started with INTERNAL_EVAL_ENABLED on a local
environment) with a seeded admin. The probe logic here is site-agnostic: the questions and
what to expect of each live in a per-site data file, `tests/evals/probes/<site_key>.json`.

    LIVE_PROBE_PASSWORD=... uv run python -m scripts.live_bot_probe --site <site_key> \
        [--probes PATH] [--skip-gaps]

Environment: LIVE_PROBE_API (default http://127.0.0.1:8000), LIVE_PROBE_EMAIL,
LIVE_PROBE_PASSWORD (15+ characters), STAFF_APP_ORIGIN, LIVE_PROBE_SITE (default for --site).

Step schema (in "steps"):
    id, name, message, optional "site" (another site key), optional "after" (id of an earlier
    step whose conversation to continue), "kind": "chat" (default) or "canned_search",
    "expect": a dict of conditions that must all hold. Chat conditions: reason, reason_not,
    locator, locator_in, locator_not_in, locator_contains, body_startswith, body_contains,
    body_not_contains, body_min_length, state, state_not, outcome, outcome_not.
    canned_search conditions: winner.
Optional "gaps": {"answer": {question, seeds, shortcut, body}, "dismiss": {question, seeds}}
seeds the question in that many separate chats, then answers or dismisses the suggested FAQ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

API = os.environ.get("LIVE_PROBE_API", "http://127.0.0.1:8000")
ORIGIN = os.environ.get("STAFF_APP_ORIGIN", "http://localhost:3000")
EMAIL = os.environ.get("LIVE_PROBE_EMAIL", "live-probe@example.local")
PASSWORD = os.environ.get("LIVE_PROBE_PASSWORD", "")
PROBE_DIR = Path(__file__).resolve().parents[1] / "tests" / "evals" / "probes"


def probes_path_for(site_key: str) -> Path:
    if not site_key or "/" in site_key or "\\" in site_key or site_key.startswith("."):
        raise ValueError(f"invalid site key: {site_key!r}")
    return PROBE_DIR / f"{site_key}.json"


def _reply(turn: dict[str, Any]) -> dict[str, Any]:
    return turn.get("reply") or {}


def _body(turn: dict[str, Any]) -> str:
    return str(_reply(turn).get("body") or "")


def _locator(turn: dict[str, Any]) -> str:
    return str(_reply(turn).get("display_locator") or "")


def _reason(turn: dict[str, Any]) -> str:
    reply = _reply(turn)
    return str(reply.get("system_reason") or reply.get("reason") or "")


def _outcome(turn: dict[str, Any]) -> str:
    return str(_reply(turn).get("outcome") or "")


# condition -> (observed field, test). Each test takes (observed, wanted).
_CONDITIONS: dict[str, tuple[str, Callable[[str, Any], bool]]] = {
    "reason": ("reason", lambda seen, want: seen == want),
    "reason_not": ("reason", lambda seen, want: seen != want),
    "locator": ("locator", lambda seen, want: seen == want),
    "locator_in": ("locator", lambda seen, want: seen in want),
    "locator_not_in": ("locator", lambda seen, want: seen not in want),
    "locator_contains": ("locator", lambda seen, want: want in seen),
    "body_startswith": ("body", lambda seen, want: seen.startswith(want)),
    "body_min_length": ("body", lambda seen, want: len(seen) >= want),
    "state": ("state", lambda seen, want: seen == want),
    "state_not": ("state", lambda seen, want: seen != want),
    "outcome": ("outcome", lambda seen, want: seen == want),
    "outcome_not": ("outcome", lambda seen, want: seen != want),
}


def check_chat(expect: dict[str, Any], turn: dict[str, Any]) -> list[str]:
    """Names of every condition in `expect` that the observed turn does not meet."""
    seen = {
        "reason": _reason(turn),
        "locator": _locator(turn),
        "body": _body(turn),
        "state": str(turn.get("state") or ""),
        "outcome": _outcome(turn),
    }
    unmet = []
    for name, (field, test) in _CONDITIONS.items():
        if name in expect and not test(seen[field], expect[name]):
            unmet.append(f"{name} {expect[name]!r}")
    lowered = seen["body"].casefold()
    unmet += [
        f"body_contains {n!r}" for n in expect.get("body_contains", []) if n not in seen["body"]
    ]
    unmet += [
        f"body_not_contains {n!r}"
        for n in expect.get("body_not_contains", [])
        if n.casefold() in lowered
    ]
    return unmet


class Probe:
    def __init__(self, client: httpx.Client, token: str, default_site: str) -> None:
        self.client = client
        self.default_site = default_site
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Origin": ORIGIN,
            "Content-Type": "application/json",
        }
        self.results: list[dict[str, Any]] = []
        self.conversations: dict[str, str] = {}

    def record(
        self,
        name: str,
        ok: bool,
        *,
        expected: str,
        actual: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {"name": name, "ok": ok, "expected": expected, "actual": actual}
        if extra:
            row["extra"] = extra
        self.results.append(row)
        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] {name}")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
        for key, value in (extra or {}).items():
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"  {key}: {text}")

    def chat(
        self,
        site_key: str,
        message: str,
        conversation_id: str | None = None,
        *,
        diagnostics: bool = False,
    ) -> dict[str, Any]:
        path = "/api/internal/dev/trace" if diagnostics else "/api/internal/dev/chat"
        payload: dict[str, Any] = {
            "site_key": site_key,
            "message": message,
            "client_message_id": str(uuid4()),
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        response = self.client.post(path, headers=self.headers, json=payload, timeout=90.0)
        if response.status_code >= 400:
            raise RuntimeError(f"{path} {response.status_code} {response.text[:400]}")
        return response.json()

    def canned_search(self, site_key: str, message: str) -> dict[str, Any]:
        response = self.client.post(
            "/api/internal/dev/canned-search",
            headers=self.headers,
            json={"site_key": site_key, "message": message},
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"canned-search {response.status_code} {response.text[:400]}")
        return response.json()

    def gaps(self, site_id: str | None, status: str = "open") -> dict[str, Any]:
        params = {"status": status}
        if site_id:
            params["site_id"] = site_id
        response = self.client.get(
            "/api/knowledge-gaps", headers=self.headers, params=params, timeout=30.0
        )
        if response.status_code >= 400:
            raise RuntimeError(f"gaps {response.status_code} {response.text[:400]}")
        return response.json()

    def site_id(self, site_key: str) -> str:
        response = self.client.get("/api/sites", headers=self.headers, timeout=30.0)
        if response.status_code >= 400:
            raise RuntimeError(f"sites {response.status_code} {response.text[:400]}")
        for item in response.json().get("items", []):
            if item.get("key") == site_key:
                return str(item["id"])
        raise RuntimeError(f"site {site_key!r} not found")

    def post_gap(self, gap_id: str, path: str, json_body: dict[str, Any] | None = None) -> int:
        response = self.client.post(
            f"/api/knowledge-gaps/{gap_id}/{path}",
            headers=self.headers,
            json=json_body,
            timeout=60.0,
        )
        return response.status_code

    def similar(self, gap_id: str) -> dict[str, Any]:
        response = self.client.get(
            f"/api/knowledge-gaps/{gap_id}/similar", headers=self.headers, timeout=30.0
        )
        if response.status_code >= 400:
            raise RuntimeError(f"similar {response.status_code} {response.text[:400]}")
        return response.json()


def login(client: httpx.Client) -> str:
    response = client.post(
        "/auth/login",
        headers={"Origin": ORIGIN, "Content-Type": "application/json"},
        json={"email": EMAIL, "password": PASSWORD},
        timeout=20.0,
    )
    if response.status_code != 200:
        raise SystemExit(f"login failed {response.status_code} {response.text[:300]}")
    return response.json()["access_token"]


def wait_for_gap(
    probe: Probe, site_id: str, needle: str, *, timeout: float = 90.0
) -> dict[str, Any] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for item in probe.gaps(site_id, "open").get("items") or []:
            if needle.casefold() in str(item.get("question") or "").casefold():
                return item
        time.sleep(3)
    return None


def run_steps(probe: Probe, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        site = step.get("site") or probe.default_site
        expect = step.get("expect") or {}
        if step.get("kind") == "canned_search":
            search = probe.canned_search(site, step["message"])
            winner = expect.get("winner")
            probe.record(
                step["name"],
                search.get("winner") == winner,
                expected=f"winner={winner}",
                actual=json.dumps({"winner": search.get("winner"), "hits": search.get("hits")[:4]}),
            )
            continue
        previous = probe.conversations.get(step["after"]) if step.get("after") else None
        turn = probe.chat(site, step["message"], previous)
        probe.conversations[step["id"]] = str(turn.get("conversation_id"))
        failed = check_chat(expect, turn)
        probe.record(
            step["name"],
            not failed,
            expected=json.dumps(expect),
            actual=(
                f"reason={_reason(turn)} locator={_locator(turn)} state={turn.get('state')} "
                f"outcome={_outcome(turn)}" + (f" unmet={failed}" if failed else "")
            ),
            extra={"body": _body(turn), "conversation_id": turn.get("conversation_id")},
        )


def seed_gap_chats(probe: Probe, site_key: str, question: str, n: int) -> None:
    for i in range(n):
        turn = probe.chat(site_key, question)
        print(
            f"  gap seed {i + 1}/{n} reason={_reason(turn)} locator={_locator(turn)} "
            f"outcome={_outcome(turn)} state={turn.get('state')}"
        )
        time.sleep(0.4)


def run_gaps(probe: Probe, plan: dict[str, Any]) -> None:
    site_key = probe.default_site
    site_id = probe.site_id(site_key)
    answer, dismiss = plan.get("answer"), plan.get("dismiss")
    for item in (answer, dismiss):
        if item:
            print(f"\n--- seeding {item['seeds']} chats: {item['question'][:60]} ---")
            seed_gap_chats(probe, site_key, item["question"], item["seeds"])
    found = {}
    for label, item in (("answer", answer), ("dismiss", dismiss)):
        if not item:
            continue
        found[label] = wait_for_gap(probe, site_id, item["question"])
        gap = found[label]
        probe.record(
            f"{label} question appears on Suggested FAQs after {item['seeds']} chats",
            gap is not None and int((gap or {}).get("conversations") or 0) >= item["seeds"],
            expected=f"open queue item with conversations>={item['seeds']}",
            actual=json.dumps(gap, default=str) if gap else "not in open queue",
        )
    if answer and found.get("answer"):
        gap = found["answer"]
        similar = probe.similar(gap["id"])
        probe.record(
            "Similar endpoint returns a payload for an open gap",
            "canned" in similar and "knowledge" in similar,
            expected="keys canned and knowledge",
            actual=json.dumps(similar, default=str)[:400],
        )
        status = probe.post_gap(
            gap["id"],
            "answer",
            {"kind": "canned", "shortcut": answer["shortcut"], "body": answer["body"]},
        )
        answered = {i["id"] for i in probe.gaps(site_id, "answered").get("items") or []}
        probe.record(
            "Answer writes a canned reply and moves the gap to answered",
            status == 204 and gap["id"] in answered,
            expected="204 and gap in answered view",
            actual=f"status={status} in_answered={gap['id'] in answered}",
        )
        replay = probe.chat(site_key, answer["question"])
        probe.record(
            "Bot uses the new canned after answering the suggested FAQ",
            _reason(replay) == "canned"
            and _locator(replay) == f"#{answer['shortcut']}"
            and _body(replay) == answer["body"],
            expected=f"canned #{answer['shortcut']} verbatim answer body",
            actual=f"reason={_reason(replay)} locator={_locator(replay)}",
            extra={"body": _body(replay)},
        )
    if dismiss and found.get("dismiss"):
        gap = found["dismiss"]
        status = probe.post_gap(gap["id"], "dismiss")
        dismissed = {i["id"] for i in probe.gaps(site_id, "dismissed").get("items") or []}
        still_open = {i["id"] for i in probe.gaps(site_id, "open").get("items") or []}
        probe.record(
            "Dismiss hides the question from the open queue",
            status == 204 and gap["id"] in dismissed and gap["id"] not in still_open,
            expected="204, in dismissed, not in open",
            actual=(
                f"status={status} dismissed={gap['id'] in dismissed} "
                f"still_open={gap['id'] in still_open}"
            ),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=os.environ.get("LIVE_PROBE_SITE"), help="site key")
    parser.add_argument("--probes", type=Path, help="probe file instead of the site default")
    parser.add_argument("--skip-gaps", action="store_true", help="do not seed or answer FAQs")
    args = parser.parse_args(argv)
    if not args.site and not args.probes:
        parser.error("pass --site (or set LIVE_PROBE_SITE) or --probes")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(PASSWORD) < 15:
        print("LIVE_PROBE_PASSWORD must be at least 15 characters", file=sys.stderr)
        return 2
    path = args.probes or probes_path_for(args.site)
    plan = json.loads(path.read_text())
    site_key = args.site or plan["site_key"]
    if plan.get("site_key") and plan["site_key"] != site_key:
        print(f"{path.name} is for site {plan['site_key']!r}, not {site_key!r}", file=sys.stderr)
        return 2
    with httpx.Client(base_url=API) as client:
        probe = Probe(client, login(client), site_key)
        ping = client.post(
            "/api/internal/dev/canned-search",
            headers=probe.headers,
            json={"site_key": site_key, "message": "hello"},
            timeout=30.0,
        )
        probe.record(
            "internal canned-search workbench is mounted",
            ping.status_code == 200,
            expected="200 from POST /api/internal/dev/canned-search",
            actual=f"{ping.status_code} {ping.text[:200]}",
        )
        run_steps(probe, plan.get("steps", []))
        if plan.get("gaps") and not args.skip_gaps:
            run_gaps(probe, plan["gaps"])

    failed = [row for row in probe.results if not row["ok"]]
    print("\n======== SUMMARY ========")
    print(f"passed {len(probe.results) - len(failed)} / {len(probe.results)}")
    for row in failed:
        print(f"FAIL {row['name']}: {row['actual']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
