import atexit
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import hnsw
import routes


# Importing routes registers the production HNSW shutdown writer.
# Unit tests must never persist their process-local HNSW state.
atexit.unregister(hnsw.save_hnsw)


class GateProbe:
    """Event-compatible gate that records entry into wait()."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.wait_entered = asyncio.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    async def wait(self) -> bool:
        self.wait_entered.set()
        await self._event.wait()
        return True


class FakeBackendResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, finished: asyncio.Event) -> None:
        self.finished = finished

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        first = {
            "id": "chatcmpl-gate-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": (
                            "Deterministic completion gate answer."
                        ),
                    },
                    "finish_reason": None,
                }
            ],
        }
        terminal = {
            "id": "chatcmpl-gate-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        try:
            yield "data: " + json.dumps(first)
            yield ""
            yield "data: " + json.dumps(terminal)
            yield ""
            yield "data: [DONE]"
        finally:
            self.finished.set()


class FakeStreamContext:
    def __init__(
        self,
        response: FakeBackendResponse,
        closed: asyncio.Event,
    ) -> None:
        self.response = response
        self.closed = closed

    async def __aenter__(self) -> FakeBackendResponse:
        return self.response

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.closed.set()
        return False


class CompletionGateTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_cancel_at_closed_gate_skips_write_path(self):
        backend_finished = asyncio.Event()
        backend_closed = asyncio.Event()
        gate = GateProbe()

        fake_response = FakeBackendResponse(
            backend_finished
        )

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(
                self,
                exc_type,
                exc,
                traceback,
            ) -> bool:
                return False

            def stream(self, *args, **kwargs):
                return FakeStreamContext(
                    fake_response,
                    backend_closed,
                )

        process_episodic = AsyncMock()
        schedule_hybrid_write_path = Mock()

        with patch.object(
            routes.httpx,
            "AsyncClient",
            FakeAsyncClient,
        ), patch.object(
            routes,
            "process_episodic",
            process_episodic,
        ), patch.object(
            routes,
            "_schedule_hybrid_write_path",
            schedule_hybrid_write_path,
        ):
            response = routes._stream_main_chat_response(
                {
                    "model": "test-model",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Gate test",
                        }
                    ],
                },
                "http://fake-backend.invalid/chat",
                write_path_messages=[
                    {
                        "role": "user",
                        "content": "Gate test",
                    }
                ],
                active_state={},
                skip_write_path=False,
                timeout_seconds=10.0,
                completion_gate=gate,
            )

            async def drain_response():
                chunks = []

                async for chunk in response.body_iterator:
                    chunks.append(chunk)

                return chunks

            consumer = asyncio.create_task(
                drain_response()
            )

            await asyncio.wait_for(
                backend_finished.wait(),
                timeout=1.0,
            )
            await asyncio.wait_for(
                backend_closed.wait(),
                timeout=1.0,
            )
            await asyncio.wait_for(
                gate.wait_entered.wait(),
                timeout=1.0,
            )

            self.assertFalse(gate.is_set())
            self.assertFalse(consumer.done())

            process_episodic.assert_not_called()
            schedule_hybrid_write_path.assert_not_called()

            consumer.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await consumer

            process_episodic.assert_not_called()
            schedule_hybrid_write_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
