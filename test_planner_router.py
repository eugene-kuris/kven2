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
            [{"name": "stable_tool"}],
        )

        self.assertLess(
            prompt.index("Available tools (use the numeric id"),
            prompt.index("Conversation context:"),
        )
        self.assertTrue(prompt.endswith("DYNAMIC-CONTEXT"))
        self.assertIn(
            "FAST",
            prompt,
        )
        self.assertIn(
            "THINK",
            prompt,
        )
        self.assertIn("TOOL <numeric_id>", prompt)
        self.assertEqual(
            planner_router._compact_tool_catalog(
                planner_router._normalize_tools(TOOLS)
            )[1]["id"],
            1,
        )

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
