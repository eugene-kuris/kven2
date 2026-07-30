import unittest
from unittest.mock import AsyncMock, patch

import planner_router


class PlannerThinkingTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_mode(self):
        planner_meta = {
            "elapsed_seconds": 0.25,
            "prompt_tokens": 120,
            "cached_tokens": 80,
        }

        with patch.object(
            planner_router,
            "_post_planner_json",
            AsyncMock(
                return_value=(
                    {"mode": "FAST"},
                    planner_meta,
                )
            ),
        ) as post:
            result = await planner_router.classify_main_thinking(
                [
                    {
                        "role": "user",
                        "content": "Say hello.",
                    }
                ],
                timeout_seconds=7.5,
            )

        self.assertEqual(result["mode"], "FAST")
        self.assertEqual(
            result["meta"]["selection"],
            planner_meta,
        )

        self.assertEqual(
            post.await_args.kwargs["max_tokens"],
            planner_router.THINKING_MAX_TOKENS,
        )
        self.assertEqual(
            post.await_args.kwargs["timeout_seconds"],
            7.5,
        )

    async def test_think_mode(self):
        with patch.object(
            planner_router,
            "_post_planner_json",
            AsyncMock(
                return_value=(
                    {"mode": "think"},
                    {"elapsed_seconds": 0.4},
                )
            ),
        ):
            result = await planner_router.classify_main_thinking(
                [
                    {
                        "role": "user",
                        "content": "Calculate RAID5 usable capacity.",
                    }
                ]
            )

        self.assertEqual(result["mode"], "THINK")

    async def test_invalid_mode_returns_error(self):
        with patch.object(
            planner_router,
            "_post_planner_json",
            AsyncMock(
                return_value=(
                    {"mode": "MAYBE"},
                    {"elapsed_seconds": 0.2},
                )
            ),
        ):
            result = await planner_router.classify_main_thinking(
                [
                    {
                        "role": "user",
                        "content": "Check the configuration.",
                    }
                ]
            )

        self.assertEqual(result["mode"], "ERROR")
        self.assertIn(
            "unknown thinking mode",
            result["error"],
        )

    def test_prompt_keeps_dynamic_context_last(self):
        context = "DYNAMIC-CONTEXT-MARKER"
        prompt = planner_router._thinking_prompt(context)

        self.assertTrue(
            prompt.endswith(
                "Conversation context:\n"
                "DYNAMIC-CONTEXT-MARKER"
            )
        )
        self.assertIn(
            '{"mode":"FAST"}',
            prompt,
        )
        self.assertIn(
            '{"mode":"THINK"}',
            prompt,
        )
        self.assertIn(
            "Applying any rule",
            prompt,
        )
        self.assertIn(
            "git merge --ff-only",
            prompt,
        )
        self.assertIn(
            "UNIX rwx",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
