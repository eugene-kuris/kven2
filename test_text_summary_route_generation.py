import copy
import os
import unittest
from unittest import mock

import context_window
import text_summary_checkpoint_manager as manager


class TextSummaryRouteGenerationTests(
    unittest.IsolatedAsyncioTestCase
):
    def _messages(self) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "runtime system",
            },
            {
                "role": "user",
                "content": "question one " * 45,
            },
            {
                "role": "assistant",
                "content": "answer one " * 45,
            },
            {
                "role": "user",
                "content": "question two " * 45,
            },
            {
                "role": "assistant",
                "content": "answer two " * 45,
            },
            {
                "role": "user",
                "content": "current question",
            },
            {
                "role": "assistant",
                "content": "current draft",
            },
        ]

    def _enabled_env(self) -> dict[str, str]:
        return {
            "KVEN2_TEXT_SUMMARY_GENERATION_ENABLED": "1",
            "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
            "KVEN2_TEXT_SUMMARY_GENERATION_MIN_MESSAGES": "1",
            "KVEN2_TEXT_SUMMARY_GENERATION_MIN_CHARS": "1000",
            "KVEN2_TEXT_SUMMARY_REFRESH_MIN_MESSAGES": "1",
            "KVEN2_TEXT_SUMMARY_REFRESH_MIN_CHARS": "1000",
            "KVEN2_TEXT_SUMMARY_GENERATION_TIMEOUT": "30",
            "KVEN2_TEXT_SUMMARY_GENERATION_MAX_TOKENS": "512",
            "KVEN2_TEXT_SUMMARY_GENERATION_MAX_INPUT_CHARS": "20000",
            "KVEN2_TEXT_SUMMARY_CHECKPOINT_LOAD_LIMIT": "11",
            "KVEN2_TEXT_SUMMARY_CHECKPOINT_STORE_LIMIT": "123",
        }

    async def test_disabled_feature_does_not_touch_dependencies(self):
        messages = self._messages()

        with mock.patch.dict(
            os.environ,
            {
                "KVEN2_TEXT_SUMMARY_GENERATION_ENABLED": "0",
            },
            clear=False,
        ), mock.patch.object(
            manager,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(),
        ) as load_mock, mock.patch.object(
            manager,
            "generate_text_summary",
            new=mock.AsyncMock(),
        ) as generate_mock:
            result = await (
                manager
                .maybe_generate_text_summary_checkpoint(
                    messages,
                    route_label="test",
                )
            )

        self.assertIsNone(result)
        load_mock.assert_not_awaited()
        generate_mock.assert_not_awaited()

    async def test_generates_and_saves_exact_safe_checkpoint(self):
        messages = self._messages()
        original = copy.deepcopy(messages)

        with mock.patch.dict(
            os.environ,
            self._enabled_env(),
            clear=False,
        ), mock.patch.object(
            manager,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[]),
        ) as load_mock, mock.patch.object(
            manager,
            "generate_text_summary",
            new=mock.AsyncMock(
                return_value=(
                    "Generated historical summary.",
                    {"elapsed_seconds": 1.25},
                )
            ),
        ) as generate_mock, mock.patch.object(
            manager,
            "save_text_summary_checkpoint",
            new=mock.AsyncMock(return_value=True),
        ) as save_mock:
            checkpoint = await (
                manager
                .maybe_generate_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertEqual(messages, original)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(
            checkpoint["summarized_message_count"],
            4,
        )
        self.assertEqual(
            checkpoint["summary_text"],
            "Generated historical summary.",
        )
        load_mock.assert_awaited_once_with(
            max_summarized_message_count=4,
            limit=11,
        )
        generate_mock.assert_awaited_once_with(
            messages[1:5],
            prior_summary=None,
            timeout_seconds=30.0,
            max_tokens=512,
            max_input_chars=20000,
        )
        save_mock.assert_awaited_once()
        saved_checkpoint = save_mock.await_args.args[0]
        self.assertEqual(
            saved_checkpoint["checkpoint_id"],
            checkpoint["checkpoint_id"],
        )
        self.assertEqual(
            save_mock.await_args.kwargs[
                "max_checkpoints"
            ],
            123,
        )

    async def test_fresh_matching_checkpoint_skips_refresh(self):
        messages = self._messages()
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="Existing summary.",
                summarized_prefix_end=3,
            )
        )
        env = self._enabled_env()
        env.update(
            {
                "KVEN2_TEXT_SUMMARY_REFRESH_MIN_MESSAGES": "3",
                "KVEN2_TEXT_SUMMARY_REFRESH_MIN_CHARS": "100000",
            }
        )

        with mock.patch.dict(
            os.environ,
            env,
            clear=False,
        ), mock.patch.object(
            manager,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[checkpoint]),
        ), mock.patch.object(
            manager,
            "generate_text_summary",
            new=mock.AsyncMock(),
        ) as generate_mock, mock.patch.object(
            manager,
            "save_text_summary_checkpoint",
            new=mock.AsyncMock(),
        ) as save_mock:
            result = await (
                manager
                .maybe_generate_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertEqual(
            result["checkpoint_id"],
            checkpoint["checkpoint_id"],
        )
        generate_mock.assert_not_awaited()
        save_mock.assert_not_awaited()

    async def test_refresh_uses_prior_summary_and_only_new_history(self):
        messages = self._messages()
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="Existing summary.",
                summarized_prefix_end=3,
            )
        )

        with mock.patch.dict(
            os.environ,
            self._enabled_env(),
            clear=False,
        ), mock.patch.object(
            manager,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[checkpoint]),
        ), mock.patch.object(
            manager,
            "generate_text_summary",
            new=mock.AsyncMock(
                return_value=(
                    "Refreshed summary.",
                    {"elapsed_seconds": 1.0},
                )
            ),
        ) as generate_mock, mock.patch.object(
            manager,
            "save_text_summary_checkpoint",
            new=mock.AsyncMock(return_value=True),
        ):
            result = await (
                manager
                .maybe_generate_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertEqual(
            result["summary_text"],
            "Refreshed summary.",
        )
        generate_mock.assert_awaited_once_with(
            messages[3:5],
            prior_summary="Existing summary.",
            timeout_seconds=30.0,
            max_tokens=512,
            max_input_chars=20000,
        )

    async def test_generation_failure_is_fail_open(self):
        messages = self._messages()

        with mock.patch.dict(
            os.environ,
            self._enabled_env(),
            clear=False,
        ), mock.patch.object(
            manager,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[]),
        ), mock.patch.object(
            manager,
            "generate_text_summary",
            new=mock.AsyncMock(
                side_effect=OSError("planner unavailable")
            ),
        ), mock.patch.object(
            manager,
            "save_text_summary_checkpoint",
            new=mock.AsyncMock(),
        ) as save_mock:
            result = await (
                manager
                .maybe_generate_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertIsNone(result)
        save_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
