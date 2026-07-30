import atexit
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

import hnsw
import routes

# Importing routes registers the production HNSW persistence handler.
# Unit tests must never persist process-local HNSW state on exit.
atexit.unregister(hnsw.save_hnsw)


class ConditionalMainThinkingTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_disabled_flag_preserves_thinking(self):
        planner = AsyncMock()

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "0",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_classify_main_thinking",
            planner,
        ):
            selected = await routes._resolve_main_chat_thinking(
                [{"role": "user", "content": "Say hello."}],
                rag_detected=False,
                gateway_tool_loop_enabled=False,
            )

        self.assertIs(selected, True)
        planner.assert_not_awaited()

    async def test_fast_disables_thinking(self):
        planner = AsyncMock(
            return_value={
                "mode": "FAST",
                "meta": {
                    "selection": {
                        "elapsed_seconds": 0.7,
                    }
                },
            }
        )

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_TIMEOUT": "7.5",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_classify_main_thinking",
            planner,
        ):
            messages = [
                {
                    "role": "user",
                    "content": "Extract the stated port number.",
                }
            ]

            selected = await routes._resolve_main_chat_thinking(
                messages,
                rag_detected=False,
                gateway_tool_loop_enabled=False,
            )

        self.assertIs(selected, False)
        self.assertEqual(
            planner.await_args.args[0],
            messages,
        )
        self.assertEqual(
            planner.await_args.kwargs["timeout_seconds"],
            7.5,
        )

    async def test_think_enables_thinking(self):
        planner = AsyncMock(
            return_value={
                "mode": "THINK",
                "meta": {},
            }
        )

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_classify_main_thinking",
            planner,
        ):
            selected = await routes._resolve_main_chat_thinking(
                [
                    {
                        "role": "user",
                        "content": "Calculate RAID5 capacity.",
                    }
                ],
                rag_detected=False,
                gateway_tool_loop_enabled=False,
            )

        self.assertIs(selected, True)
        planner.assert_awaited_once()

    async def test_planner_error_falls_back_to_thinking(self):
        planner = AsyncMock(
            return_value={
                "mode": "ERROR",
                "error": "planner unavailable",
                "meta": {},
            }
        )

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_classify_main_thinking",
            planner,
        ):
            selected = await routes._resolve_main_chat_thinking(
                [{"role": "user", "content": "Diagnose this issue."}],
                rag_detected=False,
                gateway_tool_loop_enabled=False,
            )

        self.assertIs(selected, True)

    async def test_timeout_falls_back_to_thinking(self):
        planner = AsyncMock(
            side_effect=asyncio.TimeoutError(),
        )

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "planner_classify_main_thinking",
            planner,
        ):
            selected = await routes._resolve_main_chat_thinking(
                [{"role": "user", "content": "Diagnose this issue."}],
                rag_detected=False,
                gateway_tool_loop_enabled=False,
            )

        self.assertIs(selected, True)

    async def test_base_policy_bypasses_planner(self):
        planner = AsyncMock()

        cases = [
            {
                "main_enabled": "0",
                "rag_detected": False,
                "gateway_tool_loop_enabled": False,
            },
            {
                "main_enabled": "1",
                "rag_detected": True,
                "gateway_tool_loop_enabled": False,
            },
            {
                "main_enabled": "1",
                "rag_detected": False,
                "gateway_tool_loop_enabled": True,
            },
        ]

        for case in cases:
            with self.subTest(case=case), patch.dict(
                os.environ,
                {
                    "KVEN2_MAIN_ENABLE_THINKING": case[
                        "main_enabled"
                    ],
                    "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
                },
                clear=False,
            ), patch.object(
                routes,
                "planner_classify_main_thinking",
                planner,
            ):
                selected = await routes._resolve_main_chat_thinking(
                    [{"role": "user", "content": "Test request."}],
                    rag_detected=case["rag_detected"],
                    gateway_tool_loop_enabled=case[
                        "gateway_tool_loop_enabled"
                    ],
                )

            self.assertIs(selected, False)

        planner.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
