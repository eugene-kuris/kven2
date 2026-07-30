import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.responses import Response

import routes


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


def make_payload(*, messages=None, tool_choice=None):
    payload = {
        "model": "test-model",
        "stream": False,
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

    async def test_planner_error_falls_back_to_legacy_decision(self):
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
        expected = Response(content=b"fallback-final")
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
        legacy_decision.assert_awaited_once()
        final_response.assert_awaited_once()

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
