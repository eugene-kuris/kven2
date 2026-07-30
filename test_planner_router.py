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
                {"decision": "NO_TOOL", "mode": "THINK"},
                {"elapsed_seconds": 0.5},
            )
        )

        with patch.object(
            planner_router,
            "_post_planner_json",
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
                {
                    "decision": "NO_TOOL",
                    "mode": "MAYBE",
                },
                {
                    "elapsed_seconds": 0.2,
                },
            )
        )

        with patch.object(
            planner_router,
            "_post_planner_json",
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
            "unknown NO_TOOL answer mode",
            result["error"],
        )

    async def test_tool_selection_and_arguments(self):
        mocked = AsyncMock(
            side_effect=[
                (
                    {
                        "decision": "TOOL",
                        "name": "search_web",
                    },
                    {
                        "elapsed_seconds": 1.0,
                    },
                ),
                (
                    {
                        "name": "search_web",
                        "arguments": {
                            "query": "current USD exchange rate",
                        },
                    },
                    {
                        "elapsed_seconds": 1.2,
                    },
                ),
            ]
        )

        with patch.object(
            planner_router,
            "_post_planner_json",
            mocked,
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
        self.assertEqual(mocked.await_count, 2)

        function = result["tool_call"]["function"]
        self.assertEqual(function["name"], "search_web")
        self.assertEqual(
            json.loads(function["arguments"]),
            {
                "query": "current USD exchange rate",
            },
        )

    async def test_missing_required_argument_is_error(self):
        mocked = AsyncMock(
            side_effect=[
                (
                    {
                        "decision": "TOOL",
                        "name": "search_web",
                    },
                    {},
                ),
                (
                    {
                        "name": "search_web",
                        "arguments": {},
                    },
                    {},
                ),
            ]
        )

        with patch.object(
            planner_router,
            "_post_planner_json",
            mocked,
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
            prompt.index("Available tools:"),
            prompt.index("Conversation context:"),
        )
        self.assertTrue(prompt.endswith("DYNAMIC-CONTEXT"))
        self.assertIn(
            '{"decision":"NO_TOOL","mode":"FAST"}',
            prompt,
        )
        self.assertIn(
            '{"decision":"NO_TOOL","mode":"THINK"}',
            prompt,
        )

    def test_arguments_prompt_keeps_dynamic_context_last(self):
        prompt = planner_router._arguments_prompt(
            "DYNAMIC-CONTEXT",
            {
                "name": "stable_tool",
                "description": "Stable description",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        )

        self.assertLess(
            prompt.index("Selected tool:"),
            prompt.index("Conversation context:"),
        )
        self.assertTrue(prompt.endswith("DYNAMIC-CONTEXT"))


    async def test_explicit_tool_skips_selection(self):
        mocked = AsyncMock(
            return_value=(
                {
                    "name": "fetch_url",
                    "arguments": {
                        "url": "https://example.com",
                    },
                },
                {
                    "elapsed_seconds": 0.7,
                },
            )
        )

        with patch.object(
            planner_router,
            "_post_planner_json",
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
