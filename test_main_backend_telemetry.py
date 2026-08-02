import json
import unittest
from unittest.mock import patch

import routes


class _FakeStreamingResponse:
    status_code = 200
    headers = {
        "content-type": "text/event-stream",
    }

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield "data: " + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": (
                    "chat.completion.chunk"
                ),
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

        yield "data: " + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": (
                    "chat.completion.chunk"
                ),
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": (
                                "<think>\n\n"
                                "</think>\n\n"
                            )
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

        yield "data: " + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": (
                    "chat.completion.chunk"
                ),
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "OK"
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

        yield "data: " + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": (
                    "chat.completion.chunk"
                ),
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "prompt_tokens_details": {
                        "cached_tokens": 8
                    },
                },
                "timings": {
                    "cache_n": 8,
                    "prompt_n": 12,
                    "prompt_ms": 20.5,
                    "prompt_per_second": 585.4,
                    "predicted_n": 3,
                    "predicted_ms": 10.0,
                    "predicted_per_second": 300.0,
                },
            }
        )

        yield "data: [DONE]"


class _FakeAsyncClient:
    last_json = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def stream(
        self,
        *args,
        **kwargs,
    ):
        type(self).last_json = kwargs.get(
            "json"
        )
        return _FakeStreamingResponse()


class MainBackendTelemetryTests(
    unittest.IsolatedAsyncioTestCase
):
    def test_normalizes_llama_cpp_usage(self):
        summary = (
            routes
            ._summarize_main_backend_usage(
                {
                    "prompt_n": 100,
                    "cache_n": 80,
                    "predicted_n": 20,
                    "prompt_ms": 1200.5,
                    "predicted_ms": 500.0,
                    "prompt_per_second": 83.3,
                    "predicted_per_second": 40.0,
                }
            )
        )

        self.assertEqual(
            summary["input_tokens"],
            100,
        )
        self.assertEqual(
            summary["cached_tokens"],
            80,
        )
        self.assertEqual(
            summary["output_tokens"],
            20,
        )
        self.assertEqual(
            summary["total_tokens"],
            120,
        )
        self.assertEqual(
            summary["prompt_ms"],
            1200.5,
        )
        self.assertEqual(
            summary["generation_ms"],
            500.0,
        )

    def test_normalizes_openai_usage(self):
        summary = (
            routes
            ._summarize_main_backend_usage(
                {
                    "prompt_tokens": 50,
                    "completion_tokens": 7,
                    "total_tokens": 57,
                }
            )
        )

        self.assertEqual(
            summary["input_tokens"],
            50,
        )
        self.assertEqual(
            summary["output_tokens"],
            7,
        )
        self.assertEqual(
            summary["total_tokens"],
            57,
        )

    async def test_stream_logs_and_preserves_usage(
        self,
    ):
        payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "test",
                },
            ],
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            "max_tokens": 128,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        with patch.object(
            routes.httpx,
            "AsyncClient",
            _FakeAsyncClient,
        ):
            with self.assertLogs(
                routes.logger,
                level="INFO",
            ) as captured:
                response = (
                    routes
                    ._stream_main_chat_response(
                        payload,
                        (
                            "http://backend.invalid/"
                            "v1/chat/completions"
                        ),
                        write_path_messages=(
                            payload["messages"]
                        ),
                        active_state={},
                        skip_write_path=True,
                    )
                )

                chunks = []

                async for chunk in (
                    response.body_iterator
                ):
                    if isinstance(
                        chunk,
                        bytes,
                    ):
                        chunk = chunk.decode(
                            "utf-8"
                        )

                    chunks.append(chunk)

        output = "".join(chunks)
        logs = "\n".join(
            captured.output
        )

        self.assertIn(
            '"prompt_tokens": 12',
            output,
        )
        self.assertIn(
            '"cached_tokens": 8',
            output,
        )
        self.assertIn(
            "[MAIN_BACKEND_TELEMETRY]",
            logs,
        )
        self.assertIn(
            "outcome=completed",
            logs,
        )
        self.assertIn(
            "input_tokens=12",
            logs,
        )
        self.assertIn(
            "cached_tokens=8",
            logs,
        )
        self.assertIn(
            "output_tokens=3",
            logs,
        )
        self.assertIn(
            "total_tokens=15",
            logs,
        )
        self.assertIn(
            "prompt_ms=20.5",
            logs,
        )
        self.assertIn(
            "generation_ms=10.0",
            logs,
        )
        self.assertIn(
            "prompt_tps=585.4",
            logs,
        )
        self.assertIn(
            "generation_tps=300.0",
            logs,
        )
        self.assertIn(
            "max_tokens=128",
            logs,
        )
        self.assertTrue(
            _FakeAsyncClient.last_json[
                "stream_options"
            ]["include_usage"]
        )
        self.assertNotIn(
            "<think>",
            output,
        )
        self.assertRegex(
            logs,
            r"first_backend_output_ms=\d",
        )
        self.assertIn(
            "first_reasoning_ms=None",
            logs,
        )
        self.assertRegex(
            logs,
            r"first_answer_ms=\d",
        )
        self.assertRegex(
            logs,
            r"stream_total_ms=\d",
        )

    async def test_internal_usage_is_exposed_without_client_opt_in(
        self,
    ):
        payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "test",
                },
            ],
            "stream": True,
            "max_tokens": 64,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        with patch.object(
            routes.httpx,
            "AsyncClient",
            _FakeAsyncClient,
        ):
            with self.assertLogs(
                routes.logger,
                level="INFO",
            ) as captured:
                response = (
                    routes
                    ._stream_main_chat_response(
                        payload,
                        (
                            "http://backend.invalid/"
                            "v1/chat/completions"
                        ),
                        write_path_messages=(
                            payload["messages"]
                        ),
                        active_state={},
                        skip_write_path=True,
                    )
                )

                chunks = []

                async for chunk in (
                    response.body_iterator
                ):
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode(
                            "utf-8"
                        )

                    chunks.append(chunk)

        output = "".join(chunks)
        logs = "\n".join(captured.output)

        self.assertTrue(
            _FakeAsyncClient.last_json[
                "stream_options"
            ]["include_usage"]
        )
        self.assertIn(
            '"usage"',
            output,
        )
        self.assertIn(
            '"prompt_tokens": 12',
            output,
        )
        self.assertIn(
            '"completion_tokens": 3',
            output,
        )
        self.assertIn(
            '"total_tokens": 15',
            output,
        )
        self.assertNotIn(
            '"timings"',
            output,
        )
        self.assertIn(
            "input_tokens=12",
            logs,
        )
        self.assertIn(
            "cached_tokens=8",
            logs,
        )
        self.assertIn(
            "prompt_ms=20.5",
            logs,
        )
        self.assertIn(
            "generation_ms=10.0",
            logs,
        )


if __name__ == "__main__":
    unittest.main()
