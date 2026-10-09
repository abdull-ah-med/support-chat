"""Run one site's bot eval dataset in-process against the configured providers.

    uv run python -m scripts.run_site_eval --site <site_key> [--dataset PATH] [--family PREFIX]
                                           [--json OUT.json]

Datasets live in `tests/evals/dataset/<site_key>.json`. The site must exist in the database
the app is configured for, with its knowledge already published. This uses the real embedder
and model: it exits instead of falling back to a stand-in when a provider key is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.db import session_maker
from app.repositories.site_repo import SiteRepository
from app.services.bot_eval import (
    EvalVerdict,
    ObservedReply,
    dataset_path_for,
    format_summary,
    load_dataset,
    run_case,
    summarize,
)
from app.settings import get_settings

FALLBACK_ORIGIN = "http://localhost:3000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, help="site key (selects tests/evals/dataset/KEY)")
    parser.add_argument("--dataset", type=Path, help="dataset file instead of the site default")
    parser.add_argument("--family", help="only cases whose family starts with this prefix")
    parser.add_argument("--json", dest="json_out", type=Path, help="write per-case results here")
    return parser.parse_args(argv)


def require_providers() -> None:
    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Real providers are required for a live eval; set {', '.join(missing)}.")


def _row(case_id: str, verdict: EvalVerdict, observed: ObservedReply) -> str:
    mark = "PASS" if verdict.passed else "FAIL"
    preview = (observed.body or "").replace("\n", " ")[:100]
    latency = "-" if verdict.latency_ms is None else f"{verdict.latency_ms}ms"
    return (
        f"{mark}  {case_id:36}  state={observed.state:6}  outcome={observed.outcome or '-':16}  "
        f"reason={observed.system_reason or '-':14}  cited={len(observed.citation_chunk_ids)}  "
        f"{latency:>7}  {preview}"
    )


async def run(args: argparse.Namespace) -> int:
    path = args.dataset or dataset_path_for(args.site)
    _key, cases = load_dataset(args.site, path)
    if args.family:
        cases = tuple(case for case in cases if case.family.startswith(args.family))
    if not cases:
        raise SystemExit("no cases selected")
    require_providers()
    verdicts: list[EvalVerdict] = []
    records: list[dict] = []
    async with session_maker()() as session:
        site = await SiteRepository(session).get_by_key(args.site)
        if site is None:
            raise SystemExit(f"site {args.site!r} not found")
        origin = site.allowed_origins[0] if site.allowed_origins else FALLBACK_ORIGIN
        for case in cases:
            observed, verdict = await run_case(session, site, case, parent_origin=origin)
            verdicts.append(verdict)
            print(_row(case.id, verdict, observed))
            for failure in verdict.failures:
                print(f"      {failure}")
            if not verdict.passed:
                print(f"      body={observed.body!r}")
            records.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "passed": verdict.passed,
                    "failures": list(verdict.failures),
                    "state": observed.state,
                    "outcome": observed.outcome,
                    "reason": observed.system_reason,
                    "cited": len(observed.citation_chunk_ids),
                    "foreign_citations": verdict.foreign_citations,
                    "abstained": verdict.abstained,
                    "turn_latencies_ms": list(observed.turn_latencies_ms),
                    "body": observed.body,
                }
            )
    print()
    print(format_summary(summarize(verdicts)))
    passed = sum(verdict.passed for verdict in verdicts)
    print(f"\n{passed}/{len(verdicts)} passed")
    if args.json_out:
        args.json_out.write_text(json.dumps({"site": args.site, "cases": records}, indent=2))
    return 0 if passed == len(verdicts) else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
