import atexit
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.responses import Response

import hnsw
import routes


# Importing routes imports hnsw, which registers a production persistence
# handler. Unit tests must never persist process-local HNSW state on exit.
atexit.unregister(hnsw.save_hnsw)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search current information on the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def make_payload(*, stream: bool) -> dict:
    return {
        "model": "test-model",
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": "Extract the explicitly stated port 8080.",
            }
        ],
        "tools": TOOLS,
    }


async def call_hybrid(payload: dict):
    return await routes._proxy_hybrid_native_openai_tool_protocol(
        payload,
        "http://main-backend.invalid/v1/chat/completions",
        model_adapter=None,
        write_path_messages=[],
        active_state={},
        owui_rag_meta={},
        skip_write_path=True,
        timeout_seconds=10.0,
    )


class FakeSpeculativeStream:
    def __init__(self, response):
        self.wait_started = AsyncMock()
        self.cancel = AsyncMock()
        self.release_response = Mock(return_value=response)


class HybridConditionalThinkingTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_fast_mode_releases_fast_speculation(self):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "mode": "FAST",
                "meta": {},
            }
        )
        expected = Response(content=b"fast-speculative")
        speculative = FakeSpeculativeStream(expected)
        final_response = AsyncMock()

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_start_speculative_no_tool_stream",
            return_value=speculative,
        ) as start_speculative, patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(stream=True)
            )

        self.assertIs(result, expected)
        self.assertFalse(
            start_speculative.call_args.kwargs[
                "enable_thinking"
            ]
        )
        speculative.release_response.assert_called_once()
        speculative.cancel.assert_not_awaited()
        final_response.assert_not_awaited()

    async def test_think_mode_restarts_fast_speculation(self):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "mode": "THINK",
                "meta": {},
            }
        )
        speculative = FakeSpeculativeStream(
            Response(content=b"unused-fast")
        )
        expected = Response(content=b"think-final")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_start_speculative_no_tool_stream",
            return_value=speculative,
        ) as start_speculative, patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(stream=True)
            )

        self.assertIs(result, expected)
        self.assertFalse(
            start_speculative.call_args.kwargs[
                "enable_thinking"
            ]
        )
        speculative.cancel.assert_awaited_once_with(
            reason="answer_mode_changed"
        )
        speculative.release_response.assert_not_called()
        final_response.assert_awaited_once()
        self.assertEqual(
            final_response.call_args.kwargs["answer_mode"],
            "THINK",
        )

    async def test_conditional_disabled_preserves_thinking_speculation(
        self,
    ):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "mode": "FAST",
                "meta": {},
            }
        )
        expected = Response(content=b"legacy-thinking")
        speculative = FakeSpeculativeStream(expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "0",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_start_speculative_no_tool_stream",
            return_value=speculative,
        ) as start_speculative:
            result = await call_hybrid(
                make_payload(stream=True)
            )

        self.assertIs(result, expected)
        self.assertTrue(
            start_speculative.call_args.kwargs[
                "enable_thinking"
            ]
        )
        speculative.release_response.assert_called_once()
        speculative.cancel.assert_not_awaited()

    async def test_non_stream_forwards_planner_answer_mode(self):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "mode": "FAST",
                "meta": {},
            }
        )
        expected = Response(content=b"non-stream-fast")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(stream=False)
            )

        self.assertIs(result, expected)
        final_response.assert_awaited_once()
        self.assertEqual(
            final_response.call_args.kwargs["answer_mode"],
            "FAST",
        )

    def test_missing_mode_uses_safe_thinking_fallback(self):
        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ):
            selected = (
                routes._resolve_hybrid_no_tool_thinking(
                    answer_mode=None,
                    rag_detected=False,
                )
            )

        self.assertTrue(selected)


    async def test_tool_continuation_stream_disables_thinking_by_default(
        self,
    ):
        payload = {
            "model": "test-model",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the supplied tool result.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": '{"query":"example"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_test",
                    "content": "Example result",
                },
            ],
        }
        expected = Response(content=b"continuation-fast")

        with patch.dict(
            os.environ,
            {},
            clear=False,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected,
        ) as stream_response:
            os.environ.pop(
                "KVEN2_TOOL_CONTINUATION_ENABLE_THINKING",
                None,
            )

            result = await (
                routes._proxy_hybrid_continuation_final_response(
                    payload,
                    "http://main-backend.invalid/v1/chat/completions",
                    model_adapter=None,
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                    timeout_seconds=10.0,
                )
            )

        self.assertIs(result, expected)
        guarded_payload = stream_response.call_args.args[0]
        self.assertFalse(
            guarded_payload["chat_template_kwargs"][
                "enable_thinking"
            ]
        )
        self.assertEqual(
            guarded_payload["reasoning_format"],
            "none",
        )

    async def test_tool_continuation_stream_allows_explicit_thinking_opt_in(
        self,
    ):
        payload = {
            "model": "test-model",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the supplied tool result.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": '{"query":"example"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_test",
                    "content": "Example result",
                },
            ],
        }
        expected = Response(content=b"continuation-thinking")

        with patch.dict(
            os.environ,
            {
                "KVEN2_TOOL_CONTINUATION_ENABLE_THINKING": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected,
        ) as stream_response:
            result = await (
                routes._proxy_hybrid_continuation_final_response(
                    payload,
                    "http://main-backend.invalid/v1/chat/completions",
                    model_adapter=None,
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                    timeout_seconds=10.0,
                )
            )

        self.assertIs(result, expected)
        guarded_payload = stream_response.call_args.args[0]
        self.assertTrue(
            guarded_payload["chat_template_kwargs"][
                "enable_thinking"
            ]
        )
        self.assertEqual(
            guarded_payload["reasoning_format"],
            "deepseek",
        )


if __name__ == "__main__":
    unittest.main()
