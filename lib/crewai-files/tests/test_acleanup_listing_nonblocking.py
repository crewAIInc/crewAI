"""acleanup_provider_files must not run the provider's synchronous listing on the event loop."""

import asyncio
import time
from typing import Any

import crewai_files.cache.cleanup as cleanup_module
from crewai_files.cache.cleanup import acleanup_provider_files


class BlockingListUploader:
    """Uploader whose sync listing stalls like a real network call."""

    def list_files(self) -> list[dict[str, Any]]:
        time.sleep(0.5)
        return [{"id": "file-1"}]

    async def adelete(self, file_id: str) -> bool:
        return True


def test_acleanup_delete_all_listing_does_not_block_the_event_loop(monkeypatch):
    monkeypatch.setattr(
        cleanup_module, "_uploader_or_none", lambda provider: BlockingListUploader()
    )

    ticks: list[float] = []

    async def ticker() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    async def scenario() -> int:
        ticker_task = asyncio.create_task(ticker())
        try:
            await asyncio.sleep(0.05)
            return await acleanup_provider_files(
                "openai", delete_all_from_provider=True
            )
        finally:
            ticker_task.cancel()

    deleted = asyncio.run(scenario())
    assert deleted == 1
    # The listing takes 0.5s and the ticker fires every 0.05s. If the listing
    # runs on the loop the ticker freezes for its whole duration (~3 ticks
    # total); off the loop it keeps ticking (~12 in the same window).
    assert len(ticks) >= 6, f"event loop was blocked during listing: {len(ticks)} ticks"
