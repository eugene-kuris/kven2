import json
import unittest
from unittest.mock import AsyncMock, patch

import planner_router


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
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a known URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


class PlannerRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_tool(self):
        mocked = AsyncMock(
            return_value=(
                "THINK",
                {"elapsed_seconds": 0.5},
            )
        )

        with patch.object(
            planner_router,
            "_post_planner_text",
            mocked,
        ):
            result = await planner_router.route_tool_request(
                [
                    {
                        "role": "user",
                        "content": "Explain how TCP works.",
                    }
                ],
                TOOLS,
            )

        self.assertEqual(result["decision"], "NO_TOOL")
        self.assertEqual(result["mode"], "THINK")
        self.assertEqual(mocked.await_count, 1)

    async def test_no_tool_requires_valid_answer_mode(self):
        mocked = AsyncMock(
            return_value=(
                "MAYBE",
                {
                    "elapsed_seconds": 0.2,
                },
            )
        )

        with patch.object(
            planner_router,
            "_post_planner_text",
            mocked,
        ):
            result = await planner_router.route_tool_request(
                [
                    {
                        "role": "user",
                        "content": "Explain how TCP works.",
                    }
                ],
                TOOLS,
            )

        self.assertEqual(result["decision"], "ERROR")
        self.assertIn(
            "unknown planner protocol response",
            result["error"],
        )

    async def test_tool_selection_and_arguments(self):
        selection = AsyncMock(
            return_value=(
                "TOOL 1",
                {"elapsed_seconds": 1.0},
            )
        )
        arguments = AsyncMock(
            return_value=(
                {
                    "query": "current USD exchange rate",
                },
                {"elapsed_seconds": 1.2},
            )
        )

        with (
            patch.object(
                planner_router,
                "_post_planner_text",
                selection,
            ),
            patch.object(
                planner_router,
                "_post_planner_tool_call",
                arguments,
            ),
        ):
            result = await planner_router.route_tool_request(
                [
                    {
                        "role": "user",
                        "content": "Find the current USD exchange rate.",
                    }
                ],
                TOOLS,
            )

        self.assertEqual(result["decision"], "TOOL")
        self.assertEqual(selection.await_count, 1)
        self.assertEqual(arguments.await_count, 1)

        function = result["tool_call"]["function"]
        self.assertEqual(function["name"], "search_web")
        self.assertEqual(
            json.loads(function["arguments"]),
            {
                "query": "current USD exchange rate",
            },
        )

    async def test_missing_required_argument_is_error(self):
        selection = AsyncMock(return_value=("TOOL 1", {}))
        arguments = AsyncMock(
            return_value=(
                {},
                {},
            )
        )

        with (
            patch.object(
                planner_router,
                "_post_planner_text",
                selection,
            ),
            patch.object(
                planner_router,
                "_post_planner_tool_call",
                arguments,
            ),
        ):
            result = await planner_router.route_tool_request(
                [
                    {
                        "role": "user",
                        "content": "Find current information.",
                    }
                ],
                TOOLS,
            )

        self.assertEqual(result["decision"], "ERROR")
        self.assertIn(
            "required argument is missing",
            result["error"],
        )


    def test_selection_prompt_keeps_dynamic_context_last(self):
        prompt = planner_router._selection_prompt(
            "DYNAMIC-CONTEXT",
            [
                {
                    "id": 0,
                    "name": "stable_tool",
                    "description": "Stable description.",
                }
            ],
        )

        self.assertLess(
            prompt.index("Available tools:"),
            prompt.index("Conversation context:"),
        )
        self.assertTrue(prompt.endswith("DYNAMIC-CONTEXT"))
        self.assertIn("FAST", prompt)
        self.assertIn("THINK", prompt)
        self.assertIn("TOOL <numeric_id>", prompt)
        self.assertIn(
            "0 stable_tool: Stable description.",
            prompt,
        )
        self.assertIn(
            "information or an action that is not already available",
            prompt,
        )
        self.assertIn(
            "Choose the most direct source",
            prompt,
        )
        self.assertIn(
            "a file tool for a known local path",
            prompt,
        )
        self.assertIn(
            "a URL fetch tool for an explicit URL",
            prompt,
        )
        self.assertIn(
            "web search for public information",
            prompt,
        )
        self.assertIn(
            "Do not choose web search when an explicit URL",
            prompt,
        )

        catalog = planner_router._compact_tool_catalog(
            planner_router._normalize_tools(TOOLS)
        )
        self.assertEqual(catalog[1]["id"], 1)
        self.assertEqual(
            set(catalog[1]),
            {
                "id",
                "name",
                "description",
            },
        )
        self.assertEqual(
            catalog[1]["description"],
            "Search current information on the web.",
        )
        self.assertNotIn("argument_names", prompt)

    def test_get_time_description_explains_temporal_anchor(
        self,
    ):
        from tool_registry import export_openai_tools

        tools = export_openai_tools()

        descriptions = {
            item["function"]["name"]:
                item["function"]["description"]
            for item in tools
        }

        description = descriptions["get_time"]

        self.assertIn(
            "current server date, time, timezone, and weekday",
            description,
        )
        self.assertIn(
            "current temporal state",
            description,
        )

    def test_selection_prompt_limits_dynamic_context(self):
        context = "OLD-" + ("x" * 5000) + "-LATEST"
        prompt = planner_router._selection_prompt(
            context,
            [{"id": 0, "name": "stable_tool"}],
        )

        self.assertNotIn("OLD-", prompt)
        self.assertTrue(prompt.endswith("-LATEST"))

    def test_selection_protocol_rejects_out_of_range_tool(self):
        with self.assertRaisesRegex(
            planner_router.PlannerRouterError,
            "out of range",
        ):
            planner_router._parse_selection_protocol(
                "TOOL 99",
                [{"name": "search_web"}],
            )

    def test_selection_protocol_rejects_extra_text(self):
        with self.assertRaisesRegex(
            planner_router.PlannerRouterError,
            "exactly one protocol line",
        ):
            planner_router._parse_selection_protocol(
                "THINK\nBecause this is difficult",
                [],
            )

    def test_arguments_prompt_keeps_dynamic_context_last(self):
        prompt = planner_router._arguments_prompt(
            "DYNAMIC-CONTEXT",
        )

        self.assertIn(
            "Call the provided tool exactly once",
            prompt,
        )
        self.assertTrue(prompt.endswith("DYNAMIC-CONTEXT"))


    async def test_native_tool_request_uses_required_tool_choice(self):
        captured: dict = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "search_web",
                                            "arguments": (
                                                '{"query":"current weather"}'
                                            ),
                                        }
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                    },
                }

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(
                self,
                exc_type,
                exc,
                traceback,
            ):
                return False

            async def post(self, url, json):
                captured["url"] = url
                captured["payload"] = json
                return FakeResponse()

        selected_tool = planner_router._normalize_tools(TOOLS)[
            "search_web"
        ]

        with patch.object(
            planner_router.httpx,
            "AsyncClient",
            return_value=FakeClient(),
        ):
            arguments, _ = await planner_router._post_planner_tool_call(
                "DYNAMIC-CONTEXT",
                selected_tool,
                max_tokens=32,
                timeout_seconds=5.0,
            )

        payload = captured["payload"]
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(
            payload["tools"][0]["function"]["name"],
            "search_web",
        )
        self.assertEqual(
            arguments,
            {"query": "current weather"},
        )

    def test_native_tool_call_rejects_changed_tool(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "fetch_url",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }

        with self.assertRaisesRegex(
            planner_router.PlannerRouterError,
            "changed the selected tool",
        ):
            planner_router._extract_native_tool_arguments(
                response,
                "search_web",
            )

    def test_native_tool_call_rejects_non_object_arguments(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": "[]",
                                }
                            }
                        ]
                    }
                }
            ]
        }

        with self.assertRaisesRegex(
            planner_router.PlannerRouterError,
            "not an object",
        ):
            planner_router._extract_native_tool_arguments(
                response,
                "search_web",
            )


    async def test_parameterless_explicit_tool_skips_arguments(self):
        parameterless_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Return current time.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]

        with patch.object(
            planner_router,
            "_post_planner_tool_call",
            AsyncMock(),
        ) as mocked:
            result = await planner_router.route_tool_request(
                [
                    {
                        "role": "user",
                        "content": "What time is it?",
                    }
                ],
                parameterless_tools,
                explicit_tool_name="get_time",
            )

        self.assertEqual(result["decision"], "TOOL")
        self.assertEqual(mocked.await_count, 0)
        self.assertEqual(
            result["tool_call"]["function"]["name"],
            "get_time",
        )
        self.assertEqual(
            result["tool_call"]["function"]["arguments"],
            "{}",
        )
        self.assertEqual(
            result["meta"]["arguments"]["reason"],
            "strict_empty_object_schema",
        )

    async def test_explicit_tool_skips_selection(self):
        mocked = AsyncMock(
            return_value=(
                {
                    "url": "https://example.com",
                },
                {
                    "elapsed_seconds": 0.7,
                },
            )
        )

        with patch.object(
            planner_router,
            "_post_planner_tool_call",
            mocked,
        ):
            result = await planner_router.route_tool_request(
                [
                    {
                        "role": "user",
                        "content": "Open https://example.com",
                    }
                ],
                TOOLS,
                explicit_tool_name="fetch_url",
            )

        self.assertEqual(result["decision"], "TOOL")
        self.assertEqual(mocked.await_count, 1)
        self.assertEqual(
            result["tool_call"]["function"]["name"],
            "fetch_url",
        )


if __name__ == "__main__":
    unittest.main()
