import asyncio
import logging
import time
from typing import Any

from fastapi.responses import StreamingResponse


logger = logging.getLogger(__name__)

_END = object()


class SpeculativeStream:
    """
    Start consuming an existing StreamingResponse in the background.

    Chunks remain in a bounded queue until release_response() exposes them.
    cancel() closes the wrapped async iterator and therefore its backend HTTP
    stream. A separate release gate may be used by the wrapped generator to
    prevent completion side effects, such as memory writes, before NO_TOOL.
    """

    def __init__(
        self,
        source_response: StreamingResponse,
        *,
        release_gate: asyncio.Event,
        max_chunks: int = 32,
    ) -> None:
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")

        self.source_response = source_response
        self.release_gate = release_gate
        self.queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=max_chunks
        )

        self.error: BaseException | None = None
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

        self._released = False
        self._iterator_closed = False
        self._producer = asyncio.create_task(
            self._pump(),
            name="kven-speculative-main-stream",
        )

    async def wait_started(self) -> None:
        await self.started.wait()

    async def _close_source_iterator(self) -> None:
        if self._iterator_closed:
            return

        self._iterator_closed = True
        close = getattr(
            self.source_response.body_iterator,
            "aclose",
            None,
        )

        if callable(close):
            try:
                await close()
            except RuntimeError as exc:
                # An async generator may already be closing as cancellation
                # propagates through its current __anext__ call.
                if "already running" not in str(exc):
                    raise

    async def _pump(self) -> None:
        cancelled = False
        self.started.set()

        try:
            async for chunk in self.source_response.body_iterator:
                await self.queue.put(chunk)

        except asyncio.CancelledError:
            cancelled = True
            logger.info(
                "[SPECULATIVE_STREAM] producer_cancelled=True"
            )
            raise

        except BaseException as exc:
            self.error = exc
            logger.error(
                "[SPECULATIVE_STREAM] producer_failed error=%s",
                exc,
                exc_info=True,
            )

        finally:
            try:
                await self._close_source_iterator()
            finally:
                self.finished.set()

            # No consumer exists on the TOOL/cancel path, so never block
            # trying to enqueue a sentinel after cancellation.
            if not cancelled:
                await self.queue.put(_END)

    async def cancel(self, *, reason: str) -> None:
        started = time.monotonic()

        if not self._producer.done():
            self._producer.cancel()

        try:
            await self._producer
        except asyncio.CancelledError:
            pass

        elapsed_ms = (
            time.monotonic() - started
        ) * 1000.0

        logger.info(
            "[SPECULATIVE_STREAM] cancelled reason=%s elapsed_ms=%.1f",
            reason,
            elapsed_ms,
        )

    async def _consume(self):
        try:
            while True:
                item = await self.queue.get()

                if item is _END:
                    break

                yield item

            if self.error is not None:
                raise self.error

        except asyncio.CancelledError:
            logger.info(
                "[SPECULATIVE_STREAM] client_disconnected=True"
            )
            raise

        finally:
            if not self._producer.done():
                await self.cancel(reason="client_disconnect")

    def release_response(self) -> StreamingResponse:
        if self._released:
            raise RuntimeError(
                "Speculative stream has already been released"
            )

        self._released = True
        self.release_gate.set()

        headers = {
            key: value
            for key, value in self.source_response.headers.items()
            if key.lower() != "content-length"
        }

        logger.info(
            "[SPECULATIVE_STREAM] released=True "
            "buffered_chunks=%s producer_done=%s",
            self.queue.qsize(),
            self._producer.done(),
        )

        return StreamingResponse(
            self._consume(),
            status_code=self.source_response.status_code,
            headers=headers,
            background=self.source_response.background,
        )
