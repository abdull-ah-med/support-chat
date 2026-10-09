import asyncio

from app.logging import configure_logging
from app.settings import require_provider_credentials
from app.workers import start_kb_workers, stop_kb_workers


async def _run() -> None:
    configure_logging()
    require_provider_credentials()
    await start_kb_workers()
    try:
        await asyncio.Event().wait()
    finally:
        await stop_kb_workers()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
