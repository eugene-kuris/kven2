import asyncio
import unittest

from fastapi.responses import StreamingResponse

from speculative_stream import SpeculativeStream


class SpeculativeStreamTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_release_exposes_buffered_stream(self):
        release_gate = asyncio.Event()
        source_closed = asyncio.Event()

        async def source():
            try:
                yield b"data: first\n\n"
                await release_gate.wait()
                yield b"data: second\n\n"
                yield b"data: [DONE]\n\n"
            finally:
                source_closed.set()

        speculative = SpeculativeStream(
            StreamingResponse(source()),
            release_gate=release_gate,
            max_chunks=4,
        )

        await speculative.wait_started()
        await asyncio.sleep(0)

        self.assertFalse(release_gate.is_set())

        response = speculative.release_response()
        chunks = []

        async for chunk in response.body_iterator:
            chunks.append(chunk)

        self.assertTrue(release_gate.is_set())
        self.assertTrue(source_closed.is_set())
        self.assertEqual(
            chunks,
            [
                b"data: first\n\n",
                b"data: second\n\n",
                b"data: [DONE]\n\n",
            ],
        )

    async def test_cancel_closes_source_without_release(self):
        release_gate = asyncio.Event()
        source_closed = asyncio.Event()
        source_entered = asyncio.Event()

        async def source():
            try:
                source_entered.set()
                yield b"data: partial\n\n"
                await release_gate.wait()
                yield b"data: forbidden\n\n"
            finally:
                source_closed.set()

        speculative = SpeculativeStream(
            StreamingResponse(source()),
            release_gate=release_gate,
            max_chunks=4,
        )

        await speculative.wait_started()
        await source_entered.wait()
        await asyncio.sleep(0)

        await speculative.cancel(reason="tool_selected")

        self.assertFalse(release_gate.is_set())
        self.assertTrue(source_closed.is_set())
        self.assertTrue(speculative.finished.is_set())

    async def test_client_disconnect_cancels_producer(self):
        release_gate = asyncio.Event()
        source_closed = asyncio.Event()

        async def source():
            try:
                yield b"data: first\n\n"
                await asyncio.Event().wait()
            finally:
                source_closed.set()

        speculative = SpeculativeStream(
            StreamingResponse(source()),
            release_gate=release_gate,
            max_chunks=4,
        )

        response = speculative.release_response()
        iterator = response.body_iterator

        first = await anext(iterator)
        self.assertEqual(first, b"data: first\n\n")

        await iterator.aclose()
        await asyncio.wait_for(
            source_closed.wait(),
            timeout=2.0,
        )

        self.assertTrue(source_closed.is_set())

    async def test_queue_is_bounded(self):
        release_gate = asyncio.Event()

        async def source():
            yield b"one"
            yield b"two"
            await release_gate.wait()

        speculative = SpeculativeStream(
            StreamingResponse(source()),
            release_gate=release_gate,
            max_chunks=1,
        )

        await speculative.wait_started()
        await asyncio.sleep(0.02)

        self.assertEqual(speculative.queue.maxsize, 1)
        self.assertLessEqual(speculative.queue.qsize(), 1)

        await speculative.cancel(reason="test_cleanup")


if __name__ == "__main__":
    unittest.main()
