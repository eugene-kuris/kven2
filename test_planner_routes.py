import atexit
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.responses import Response

import routes
import hnsw

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


def make_payload(
    *,
    messages=None,
    tool_choice=None,
    stream=False,
):
    payload = {
        "model": "test-model",
        "stream": stream,
        "messages": messages
        or [
            {
                "role": "user",
                "content": "Найди текущий курс доллара.",
            }
        ],
        "tools": TOOLS,
    }

    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return payload


async def call_hybrid(payload):
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


class PlannerRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_flag_uses_legacy_decision(self):
        planner = AsyncMock()
        legacy_decision = AsyncMock(
            return_value=(
                {
                    "model": "test-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "KVEN_NO_TOOL",
                            },
                        }
                    ],
                },
                None,
            )
        )
        expected = Response(content=b"legacy-final")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {"KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "0"},
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(make_payload())

        self.assertIs(result, expected)
        planner.assert_not_awaited()
        legacy_decision.assert_awaited_once()
        final_response.assert_awaited_once()

    async def test_no_tool_skips_legacy_main_decision(self):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "meta": {
                    "selection": {
                        "elapsed_seconds": 1.0,
                    }
                },
            }
        )
        legacy_decision = AsyncMock()
        expected = Response(content=b"planner-final")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {"KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1"},
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(make_payload())

        self.assertIs(result, expected)
        planner.assert_awaited_once()
        legacy_decision.assert_not_awaited()
        final_response.assert_awaited_once()

    async def test_tool_decision_returns_openai_tool_call(self):
        planner = AsyncMock(
            return_value={
                "decision": "TOOL",
                "tool_call": {
                    "id": "call_planner_test",
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "arguments": json.dumps(
                            {
                                "query": "текущий курс доллара",
                            },
                            ensure_ascii=False,
                        ),
                    },
                },
                "meta": {},
            }
        )
        legacy_decision = AsyncMock()
        final_response = AsyncMock()

        with patch.dict(
            os.environ,
            {"KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1"},
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(make_payload())

        body = json.loads(result.body)
        choice = body["choices"][0]
        tool_call = choice["message"]["tool_calls"][0]

        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            tool_call["function"]["name"],
            "search_web",
        )
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {
                "query": "текущий курс доллара",
            },
        )
        legacy_decision.assert_not_awaited()
        final_response.assert_not_awaited()

    async def test_planner_error_routes_direct_fast_without_legacy(self):
        planner = AsyncMock(
            return_value={
                "decision": "ERROR",
                "error": "planner unavailable",
                "meta": {},
            }
        )
        legacy_decision = AsyncMock()
        expected = Response(content=b"fallback-final")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_PLANNER_ERROR_FALLBACK_MIN_TOKENS": "256",
                "KVEN2_PLANNER_ERROR_FALLBACK_MAX_TOKENS": "2048",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(make_payload())

        self.assertIs(result, expected)
        planner.assert_awaited_once()
        legacy_decision.assert_not_awaited()
        final_response.assert_awaited_once()
        self.assertEqual(
            final_response.await_args.kwargs["answer_mode"],
            "FAST",
        )
        self.assertEqual(
            final_response.await_args.kwargs[
                "final_min_tokens_override"
            ],
            256,
        )
        self.assertEqual(
            final_response.await_args.kwargs[
                "final_max_tokens_cap"
            ],
            2048,
        )

    async def test_tool_continuation_bypasses_planner(self):
        messages = [
            {
                "role": "user",
                "content": "Найди текущий курс.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"курс"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "USD/UAH test result",
            },
        ]

        planner = AsyncMock()
        expected = Response(content=b"continuation")
        continuation = AsyncMock(return_value=expected)
        legacy_decision = AsyncMock()

        with patch.dict(
            os.environ,
            {"KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1"},
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_proxy_hybrid_continuation_final_response",
            continuation,
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ):
            result = await call_hybrid(
                make_payload(messages=messages)
            )

        self.assertIs(result, expected)
        continuation.assert_awaited_once()
        planner.assert_not_awaited()
        legacy_decision.assert_not_awaited()

    async def test_speculative_no_tool_releases_main_stream(self):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "meta": {},
            }
        )
        legacy_decision = AsyncMock()
        final_response = AsyncMock()

        expected = Response(content=b"speculative-final")
        speculative = FakeSpeculativeStream(expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
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
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(stream=True)
            )

        self.assertIs(result, expected)
        start_speculative.assert_called_once()
        speculative.wait_started.assert_awaited_once()
        speculative.release_response.assert_called_once()
        speculative.cancel.assert_not_awaited()
        legacy_decision.assert_not_awaited()
        final_response.assert_not_awaited()

    async def test_speculative_tool_cancels_main_stream(self):
        planner = AsyncMock(
            return_value={
                "decision": "TOOL",
                "tool_call": {
                    "id": "call_speculative_tool",
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "arguments": json.dumps(
                            {"query": "current data"}
                        ),
                    },
                },
                "meta": {},
            }
        )

        expected = Response(content=b"unused")
        speculative = FakeSpeculativeStream(expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
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
        ):
            result = await call_hybrid(
                make_payload(stream=True)
            )

        if hasattr(result, "body_iterator"):
            chunks = [
                chunk
                async for chunk in result.body_iterator
            ]
            body = b"".join(
                chunk.encode("utf-8")
                if isinstance(chunk, str)
                else chunk
                for chunk in chunks
            ).decode("utf-8")
        else:
            raw_body = result.body
            body = (
                raw_body.decode("utf-8")
                if isinstance(raw_body, bytes)
                else str(raw_body)
            )

        events = []

        for block in body.split("\n\n"):
            line = block.strip()

            if not line.startswith("data: "):
                continue

            raw = line[6:].strip()

            if not raw or raw == "[DONE]":
                continue

            events.append(json.loads(raw))

        finish_reasons = []
        tool_names = []

        for event in events:
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason")

                if finish_reason:
                    finish_reasons.append(finish_reason)

                delta = choice.get("delta") or {}

                for call in delta.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = function.get("name")

                    if name:
                        tool_names.append(name)

        self.assertIn("tool_calls", finish_reasons)
        self.assertIn("search_web", tool_names)
        speculative.cancel.assert_awaited_once_with(
            reason="tool_selected"
        )
        speculative.release_response.assert_not_called()

    async def test_speculative_error_cancels_before_direct_fast(self):
        planner = AsyncMock(
            return_value={
                "decision": "ERROR",
                "error": "planner unavailable",
                "meta": {},
            }
        )
        legacy_decision = AsyncMock(
            return_value=(
                {
                    "model": "test-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "KVEN_NO_TOOL",
                            },
                        }
                    ],
                },
                None,
            )
        )

        expected = Response(content=b"direct-fast-after-cancel")
        final_response = AsyncMock(return_value=expected)
        speculative = FakeSpeculativeStream(
            Response(content=b"unused")
        )

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
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
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(stream=True)
            )

        self.assertIs(result, expected)
        speculative.cancel.assert_awaited_once_with(
            reason="planner_error"
        )
        legacy_decision.assert_not_awaited()
        final_response.assert_awaited_once()
        self.assertEqual(
            final_response.await_args.kwargs["answer_mode"],
            "FAST",
        )
        self.assertEqual(
            final_response.await_args.kwargs[
                "final_min_tokens_override"
            ],
            256,
        )
        self.assertEqual(
            final_response.await_args.kwargs[
                "final_max_tokens_cap"
            ],
            2048,
        )

    async def test_speculation_is_skipped_for_non_stream_request(self):
        planner = AsyncMock(
            return_value={
                "decision": "NO_TOOL",
                "meta": {},
            }
        )
        expected = Response(content=b"buffered-final")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {
                "KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1",
                "KVEN2_SPECULATIVE_PLANNER_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_start_speculative_no_tool_stream",
        ) as start_speculative, patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(stream=False)
            )

        self.assertIs(result, expected)
        start_speculative.assert_not_called()
        final_response.assert_awaited_once()


    def test_planner_error_budget_disables_thinking_and_caps_tokens(self):
        guarded = routes._apply_final_answer_safeguards(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "complex request",
                    }
                ],
                "max_tokens": 24687,
            },
            route_label="planner_error_fallback",
            enable_thinking=False,
            min_tokens_override=256,
            max_tokens_cap=2048,
        )

        self.assertEqual(guarded["max_tokens"], 2048)
        self.assertFalse(
            guarded["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(guarded["reasoning_format"], "none")

    async def test_tool_choice_none_bypasses_all_decisions(self):
        planner = AsyncMock()
        legacy_decision = AsyncMock()
        expected = Response(content=b"none-final")
        final_response = AsyncMock(return_value=expected)

        with patch.dict(
            os.environ,
            {"KVEN2_PLANNER_TOOL_ROUTING_ENABLED": "1"},
            clear=False,
        ), patch.object(
            routes,
            "planner_route_tool_request",
            planner,
        ), patch.object(
            routes,
            "_post_native_decision_json",
            legacy_decision,
        ), patch.object(
            routes,
            "_proxy_hybrid_no_tool_final_response",
            final_response,
        ):
            result = await call_hybrid(
                make_payload(tool_choice="none")
            )

        self.assertIs(result, expected)
        planner.assert_not_awaited()
        legacy_decision.assert_not_awaited()
        final_response.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
