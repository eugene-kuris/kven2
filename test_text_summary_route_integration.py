import copy
import os
import unittest
from unittest import mock

import context_window
import routes


class TextSummaryRouteIntegrationTests(
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
                "content": "question one " * 30,
            },
            {
                "role": "assistant",
                "content": "answer one " * 30,
            },
            {
                "role": "user",
                "content": "question two " * 30,
            },
            {
                "role": "assistant",
                "content": "answer two " * 30,
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

    async def test_disabled_feature_does_not_touch_store(self):
        messages = self._messages()

        with mock.patch.dict(
            os.environ,
            {
                "KVEN2_TEXT_SUMMARY_COMPACTION_ENABLED": "0",
            },
            clear=False,
        ), mock.patch.object(
            routes,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(),
        ) as load_mock:
            result = await (
                routes
                ._maybe_apply_text_summary_checkpoint(
                    messages,
                    route_label="test",
                )
            )

        self.assertIs(result, messages)
        load_mock.assert_not_awaited()

    async def test_matching_checkpoint_is_applied_at_exact_boundary(self):
        messages = self._messages()
        original = copy.deepcopy(messages)
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text=(
                    "The first two exchanges established "
                    "the historical context."
                ),
                summarized_prefix_end=5,
            )
        )

        with mock.patch.dict(
            os.environ,
            {
                "KVEN2_TEXT_SUMMARY_COMPACTION_ENABLED": "1",
                "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
                "KVEN2_TEXT_SUMMARY_CHECKPOINT_LOAD_LIMIT": "7",
            },
            clear=False,
        ), mock.patch.object(
            routes,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[checkpoint]),
        ) as load_mock, mock.patch.object(
            routes,
            "mark_text_summary_checkpoint_used",
            new=mock.AsyncMock(return_value=True),
        ) as mark_mock:
            result = await (
                routes
                ._maybe_apply_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertEqual(messages, original)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[2:], messages[5:])
        self.assertEqual(result[1]["role"], "assistant")
        self.assertIn(
            checkpoint["summary_text"],
            result[1]["content"],
        )
        load_mock.assert_awaited_once_with(
            max_summarized_message_count=4,
            limit=7,
        )
        mark_mock.assert_awaited_once_with(
            checkpoint["checkpoint_id"]
        )

    async def test_no_matching_checkpoint_preserves_messages(self):
        messages = self._messages()
        other_messages = copy.deepcopy(messages)
        other_messages[1]["content"] = "different history"
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                other_messages,
                summary_text="Different history.",
                summarized_prefix_end=5,
            )
        )

        with mock.patch.dict(
            os.environ,
            {
                "KVEN2_TEXT_SUMMARY_COMPACTION_ENABLED": "1",
                "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
            },
            clear=False,
        ), mock.patch.object(
            routes,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[checkpoint]),
        ), mock.patch.object(
            routes,
            "mark_text_summary_checkpoint_used",
            new=mock.AsyncMock(),
        ) as mark_mock:
            result = await (
                routes
                ._maybe_apply_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertIs(result, messages)
        mark_mock.assert_not_awaited()

    async def test_store_failure_is_fail_open(self):
        messages = self._messages()

        with mock.patch.dict(
            os.environ,
            {
                "KVEN2_TEXT_SUMMARY_COMPACTION_ENABLED": "1",
                "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
            },
            clear=False,
        ), mock.patch.object(
            routes,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(
                side_effect=OSError("database unavailable")
            ),
        ):
            result = await (
                routes
                ._maybe_apply_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertIs(result, messages)

    async def test_mark_used_failure_does_not_undo_compaction(self):
        messages = self._messages()
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="Historical context.",
                summarized_prefix_end=5,
            )
        )

        with mock.patch.dict(
            os.environ,
            {
                "KVEN2_TEXT_SUMMARY_COMPACTION_ENABLED": "1",
                "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
            },
            clear=False,
        ), mock.patch.object(
            routes,
            "load_text_summary_checkpoints",
            new=mock.AsyncMock(return_value=[checkpoint]),
        ), mock.patch.object(
            routes,
            "mark_text_summary_checkpoint_used",
            new=mock.AsyncMock(
                side_effect=OSError("database unavailable")
            ),
        ):
            result = await (
                routes
                ._maybe_apply_text_summary_checkpoint(
                    messages,
                    route_label="main",
                )
            )

        self.assertEqual(len(result), 4)
        self.assertIn(
            checkpoint["summary_text"],
            result[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
