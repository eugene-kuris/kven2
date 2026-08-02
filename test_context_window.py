import copy
import json
import unittest

import context_window


class ContextWindowReportTests(unittest.TestCase):
    def test_report_partitions_system_older_and_tail(self):
        messages = [
            {
                "role": "system",
                "content": "stable system",
            },
            {
                "role": "user",
                "content": "older user",
            },
            {
                "role": "assistant",
                "content": "recent assistant",
            },
            {
                "role": "user",
                "content": "latest user",
            },
        ]

        report = context_window.build_context_window_report(
            messages,
            tail_messages=2,
        )

        self.assertEqual(report["messages_total"], 4)
        self.assertEqual(
            report["system_prefix_messages"],
            1,
        )
        self.assertEqual(
            report["older_candidate_messages"],
            1,
        )
        self.assertEqual(
            report["verbatim_tail_start"],
            2,
        )
        self.assertEqual(
            report["verbatim_tail_roles"],
            ["assistant", "user"],
        )
        self.assertFalse(
            report["active_tool_continuation"]
        )

    def test_active_tool_continuation_is_kept_whole(self):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "request",
            },
            {
                "role": "assistant",
                "content": "ordinary answer",
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
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "first result",
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "second result",
            },
        ]

        report = context_window.build_context_window_report(
            messages,
            tail_messages=1,
        )

        self.assertTrue(
            report["active_tool_continuation"]
        )
        self.assertEqual(
            report[
                "active_tool_continuation_start"
            ],
            3,
        )
        self.assertEqual(
            report["verbatim_tail_start"],
            3,
        )
        self.assertEqual(
            report["verbatim_tail_roles"],
            ["assistant", "tool", "tool"],
        )
        self.assertEqual(
            report["older_tool_protocol_groups"],
            0,
        )
        self.assertEqual(
            report["older_tool_protocol_indices"],
            [],
        )

    def test_report_splits_text_and_media_payloads(self):
        data_uri = (
            "data:image/png;base64,"
            + ("A" * 100)
        )
        remote_url = "https://example.test/image.png"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": remote_url,
                        },
                    },
                ],
            }
        ]

        report = context_window.build_context_window_report(
            messages,
            tail_messages=12,
        )

        self.assertEqual(
            report["text_chars_total"],
            len("hello"),
        )
        self.assertEqual(
            report["media_data_uri_chars_total"],
            len(data_uri),
        )
        self.assertEqual(
            report["media_reference_chars_total"],
            len(remote_url),
        )
        self.assertEqual(
            report["media_data_uri_count"],
            1,
        )
        self.assertEqual(
            report["media_reference_count"],
            1,
        )

        manifest = report["message_manifest"][0]

        self.assertEqual(manifest["index"], 0)
        self.assertEqual(manifest["role"], "user")
        self.assertEqual(
            manifest["part_type_counts"],
            {
                "image_url": 2,
                "text": 1,
            },
        )
        self.assertEqual(
            manifest["media_payload_chars"],
            len(data_uri) + len(remote_url),
        )

    def test_report_identifies_older_media_and_tool_candidates(self):
        data_uri = (
            "data:image/png;base64,"
            + ("A" * 100)
        )
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "inspect image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
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
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "raw historical result",
            },
            {
                "role": "assistant",
                "content": "recent answer",
            },
            {
                "role": "user",
                "content": "latest request",
            },
        ]

        report = context_window.build_context_window_report(
            messages,
            tail_messages=2,
        )

        self.assertEqual(
            report["older_media_candidate_indices"],
            [1],
        )
        self.assertEqual(
            report["older_media_candidate_messages"],
            1,
        )
        self.assertEqual(
            report[
                "older_media_candidate_payload_chars"
            ],
            len(data_uri),
        )
        self.assertEqual(
            report["older_tool_protocol_groups"],
            1,
        )
        self.assertEqual(
            report["older_tool_protocol_indices"],
            [2, 3],
        )
        self.assertEqual(
            report["older_tool_protocol_messages"],
            2,
        )
        self.assertGreater(
            report["older_tool_protocol_json_chars"],
            0,
        )

    def test_media_compaction_preview_removes_previous_turn_media(self):
        old_data_uri = (
            "data:image/png;base64,"
            + ("A" * 100)
        )
        current_data_uri = (
            "data:image/png;base64,"
            + ("B" * 80)
        )
        messages = [
            {
                "role": "system",
                "content": "stable system",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "previous image question",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": old_data_uri,
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "previous image description",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "current image question",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": current_data_uri,
                        },
                    },
                ],
            },
        ]
        original = copy.deepcopy(messages)

        compacted, meta = (
            context_window.build_historical_media_compaction_preview(
                messages,
            )
        )

        self.assertEqual(messages, original)
        self.assertEqual(compacted[0], original[0])
        self.assertEqual(compacted[2:], original[2:])
        self.assertEqual(
            compacted[1]["content"][0],
            original[1]["content"][0],
        )
        self.assertNotIn(
            old_data_uri,
            json.dumps(compacted, ensure_ascii=False),
        )
        self.assertIn(
            current_data_uri,
            json.dumps(compacted, ensure_ascii=False),
        )
        self.assertEqual(meta["candidate_indices"], [1])
        self.assertEqual(meta["compacted_indices"], [1])
        self.assertEqual(meta["removed_media_parts"], 1)
        self.assertEqual(
            meta["protected_current_turn_start"],
            3,
        )
        self.assertEqual(
            meta["protected_current_turn_roles"],
            ["user"],
        )
        self.assertGreater(meta["saved_json_chars"], 0)

    def test_media_compaction_preview_preserves_current_tool_turn(self):
        data_uri = (
            "data:image/png;base64,"
            + ("A" * 100)
        )
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    }
                ],
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
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "current result",
            },
        ]
        original = copy.deepcopy(messages)

        compacted, meta = (
            context_window.build_historical_media_compaction_preview(
                messages,
            )
        )

        self.assertEqual(compacted, original)
        self.assertTrue(meta["active_tool_continuation"])
        self.assertEqual(
            meta["protected_current_turn_start"],
            1,
        )
        self.assertEqual(
            meta["protected_current_turn_roles"],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(meta["candidate_indices"], [])
        self.assertEqual(meta["compacted_indices"], [])

    def test_media_compaction_preview_fails_safe_without_user_message(self):
        data_uri = (
            "data:image/png;base64,"
            + ("A" * 100)
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    }
                ],
            }
        ]

        compacted, meta = (
            context_window.build_historical_media_compaction_preview(
                messages,
            )
        )

        self.assertEqual(compacted, messages)
        self.assertIsNone(
            meta["protected_current_turn_start"]
        )
        self.assertEqual(meta["compacted_indices"], [])

    def test_report_does_not_mutate_or_expose_content(self):
        messages = [
            {
                "role": "system",
                "content": "SECRET-SYSTEM-CONTENT",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "SECRET-USER-CONTENT",
                    }
                ],
            },
        ]
        original = copy.deepcopy(messages)

        report = context_window.build_context_window_report(
            messages,
            tail_messages=12,
        )

        self.assertEqual(messages, original)

        encoded_report = json.dumps(
            report,
            ensure_ascii=False,
        )
        self.assertNotIn(
            "SECRET-SYSTEM-CONTENT",
            encoded_report,
        )
        self.assertNotIn(
            "SECRET-USER-CONTENT",
            encoded_report,
        )
        self.assertGreater(
            report["message_json_chars_total"],
            0,
        )


class ContextBudgetReportTests(unittest.TestCase):
    def test_token_estimator_rounds_up(self):
        self.assertEqual(
            context_window.estimate_tokens_from_chars(
                0,
            ),
            0,
        )
        self.assertEqual(
            context_window.estimate_tokens_from_chars(
                1,
            ),
            1,
        )
        self.assertEqual(
            context_window.estimate_tokens_from_chars(
                8,
            ),
            2,
        )
        self.assertEqual(
            context_window.estimate_tokens_from_chars(
                9,
            ),
            3,
        )

    def test_short_history_fits_without_compaction(self):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "question",
            },
        ]

        report = (
            context_window
            .build_context_budget_report(
                messages,
                tail_messages=12,
                context_tokens=1024,
                reserved_completion_tokens=256,
                summary_target_tokens=128,
            )
        )

        self.assertFalse(
            report["over_budget_before"]
        )
        self.assertTrue(
            report["fits_after_summary"]
        )
        self.assertFalse(
            report[
                "compaction_candidate_available"
            ]
        )
        self.assertEqual(
            report[
                "effective_summary_target_tokens"
            ],
            0,
        )

    def test_long_history_reports_bounded_summary(self):
        messages = [
            {
                "role": "system",
                "content": "system policy",
            },
        ]

        for index in range(12):
            messages.append(
                {
                    "role": (
                        "user"
                        if index % 2 == 0
                        else "assistant"
                    ),
                    "content": (
                        f"historical-{index}-"
                        + ("X" * 120)
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": "current request",
            }
        )

        report = (
            context_window
            .build_context_budget_report(
                messages,
                tail_messages=2,
                context_tokens=180,
                reserved_completion_tokens=40,
                summary_target_tokens=30,
                chars_per_token=2.0,
            )
        )

        self.assertTrue(
            report["over_budget_before"]
        )
        self.assertTrue(
            report[
                "compaction_candidate_available"
            ]
        )
        self.assertGreater(
            report[
                "required_reduction_tokens"
            ],
            0,
        )
        self.assertLessEqual(
            report[
                "effective_summary_target_tokens"
            ],
            30,
        )
        self.assertEqual(
            report[
                "estimated_prompt_tokens_after_summary"
            ],
            (
                report[
                    "estimated_fixed_prompt_tokens"
                ]
                + report[
                    "effective_summary_target_tokens"
                ]
            ),
        )

    def test_report_is_content_free_and_non_mutating(self):
        import copy
        import json

        marker = "PRIVATE_CONTEXT_MARKER_92A7"
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": (
                                '{"path":"/tmp/example"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": marker,
            },
            {
                "role": "assistant",
                "content": "historical answer",
            },
            {
                "role": "user",
                "content": "latest request",
            },
        ]
        original = copy.deepcopy(messages)

        report = (
            context_window
            .build_context_budget_report(
                messages,
                tail_messages=2,
                context_tokens=256,
                reserved_completion_tokens=64,
                summary_target_tokens=32,
            )
        )

        self.assertEqual(messages, original)
        self.assertNotIn(
            marker,
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        self.assertGreater(
            report[
                "estimated_older_tool_protocol_tokens"
            ],
            0,
        )
        self.assertGreaterEqual(
            report[
                "estimated_older_candidate_tokens"
            ],
            report[
                "estimated_older_tool_protocol_tokens"
            ],
        )



class TextSummaryCheckpointTests(unittest.TestCase):
    def test_checkpoint_matches_extended_history_and_ignores_system_prefix(
        self,
    ):
        original_messages = [
            {
                "role": "system",
                "content": "runtime system A",
            },
            {
                "role": "user",
                "content": "first question",
            },
            {
                "role": "assistant",
                "content": "first answer",
            },
        ]
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                original_messages,
                summary_text=(
                    "The first exchange established "
                    "a stable result."
                ),
                summarized_prefix_end=3,
            )
        )
        current_messages = [
            {
                "role": "system",
                "content": "runtime system B",
            },
            {
                "role": "system",
                "content": "additional runtime policy",
            },
            {
                "role": "user",
                "content": "first question",
            },
            {
                "role": "assistant",
                "content": "first answer",
            },
            {
                "role": "user",
                "content": "second question",
            },
        ]
        original_current = copy.deepcopy(
            current_messages
        )
        original_checkpoint = copy.deepcopy(
            checkpoint
        )

        matched, report = (
            context_window
            .find_matching_text_summary_checkpoint(
                current_messages,
                [checkpoint],
            )
        )

        self.assertEqual(
            current_messages,
            original_current,
        )
        self.assertEqual(
            checkpoint,
            original_checkpoint,
        )
        self.assertIsNotNone(matched)
        self.assertEqual(
            matched["checkpoint_id"],
            checkpoint["checkpoint_id"],
        )
        self.assertTrue(report["selected"])
        self.assertEqual(
            report["matching_checkpoints"],
            1,
        )
        self.assertEqual(
            report[
                "selected_summarized_message_count"
            ],
            2,
        )

    def test_edit_before_boundary_invalidates_checkpoint(
        self,
    ):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "original question",
            },
            {
                "role": "assistant",
                "content": "original answer",
            },
        ]
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="Original exchange.",
                summarized_prefix_end=3,
            )
        )
        edited_messages = copy.deepcopy(messages)
        edited_messages[1]["content"] = (
            "edited question"
        )

        matched, report = (
            context_window
            .find_matching_text_summary_checkpoint(
                edited_messages,
                [checkpoint],
            )
        )

        self.assertIsNone(matched)
        self.assertFalse(report["selected"])
        self.assertEqual(
            report["prefix_hash_mismatches"],
            1,
        )

    def test_branch_after_boundary_keeps_checkpoint_valid(
        self,
    ):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "question one",
            },
            {
                "role": "assistant",
                "content": "answer one",
            },
        ]
        checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="First exchange.",
                summarized_prefix_end=3,
            )
        )
        branched_messages = [
            *messages,
            {
                "role": "user",
                "content": "different continuation",
            },
        ]

        matched, report = (
            context_window
            .find_matching_text_summary_checkpoint(
                branched_messages,
                [checkpoint],
            )
        )

        self.assertIsNotNone(matched)
        self.assertTrue(report["selected"])
        self.assertEqual(
            report["matching_checkpoints"],
            1,
        )

    def test_longest_matching_checkpoint_wins_and_latest_breaks_tie(
        self,
    ):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "question one",
            },
            {
                "role": "assistant",
                "content": "answer one",
            },
            {
                "role": "user",
                "content": "question two",
            },
            {
                "role": "assistant",
                "content": "answer two",
            },
            {
                "role": "user",
                "content": "question three",
            },
        ]
        short_checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="Short summary.",
                summarized_prefix_end=3,
            )
        )
        first_long_checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="First long summary.",
                summarized_prefix_end=5,
            )
        )
        second_long_checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                messages,
                summary_text="Second long summary.",
                summarized_prefix_end=5,
            )
        )

        matched, report = (
            context_window
            .find_matching_text_summary_checkpoint(
                messages,
                [
                    first_long_checkpoint,
                    short_checkpoint,
                    second_long_checkpoint,
                ],
            )
        )

        self.assertIsNotNone(matched)
        self.assertEqual(
            matched["summary_text"],
            "Second long summary.",
        )
        self.assertEqual(
            report["matching_checkpoints"],
            3,
        )
        self.assertEqual(
            report[
                "selected_summarized_message_count"
            ],
            4,
        )
        self.assertEqual(
            report["selected_checkpoint_index"],
            2,
        )

    def test_corrupt_and_insufficient_checkpoints_fail_closed(
        self,
    ):
        long_messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "question one",
            },
            {
                "role": "assistant",
                "content": "answer one",
            },
            {
                "role": "user",
                "content": "question two",
            },
            {
                "role": "assistant",
                "content": "answer two",
            },
        ]
        insufficient_checkpoint = (
            context_window
            .build_text_summary_checkpoint(
                long_messages,
                summary_text="Long history.",
                summarized_prefix_end=5,
            )
        )
        corrupt_checkpoint = copy.deepcopy(
            context_window
            .build_text_summary_checkpoint(
                long_messages[:3],
                summary_text="Short history.",
                summarized_prefix_end=3,
            )
        )
        corrupt_checkpoint["checkpoint_id"] = (
            "0" * 64
        )

        matched, report = (
            context_window
            .find_matching_text_summary_checkpoint(
                long_messages[:3],
                [
                    None,
                    corrupt_checkpoint,
                    insufficient_checkpoint,
                ],
            )
        )

        self.assertIsNone(matched)
        self.assertFalse(report["selected"])
        self.assertEqual(
            report["candidate_checkpoints"],
            3,
        )
        self.assertEqual(
            report["invalid_checkpoints"],
            2,
        )
        self.assertEqual(
            report[
                "insufficient_history_checkpoints"
            ],
            1,
        )

    def test_builder_rejects_invalid_inputs(self):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "question",
            },
        ]

        with self.assertRaises(ValueError):
            context_window.build_text_summary_checkpoint(
                messages,
                summary_text="summary",
                summarized_prefix_end=1,
            )

        with self.assertRaises(ValueError):
            context_window.build_text_summary_checkpoint(
                messages,
                summary_text="   ",
                summarized_prefix_end=2,
            )

        with self.assertRaises(ValueError):
            context_window.build_text_summary_checkpoint(
                messages,
                summary_text="summary",
                summarized_prefix_end=True,
            )


class TextSummaryCompactionPreviewTests(
    unittest.TestCase
):
    def test_historical_text_is_replaced_and_tail_is_preserved(
        self,
    ):
        messages = [
            {
                "role": "system",
                "content": "stable system",
            },
            {
                "role": "user",
                "content": (
                    "PRIVATE-OLD-USER "
                    + ("A" * 300)
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "PRIVATE-OLD-ASSISTANT "
                    + ("B" * 300)
                ),
            },
            {
                "role": "user",
                "content": "recent question",
            },
            {
                "role": "assistant",
                "content": "recent answer",
            },
        ]
        original = copy.deepcopy(messages)

        compacted, report = (
            context_window
            .build_text_summary_compaction_preview(
                messages,
                summary_text=(
                    "The older exchange established "
                    "one stable fact."
                ),
                tail_messages=2,
            )
        )

        self.assertEqual(messages, original)
        self.assertTrue(
            report["compaction_applied"]
        )
        self.assertEqual(
            report["reason"],
            "applied",
        )
        self.assertEqual(
            report["older_candidate_messages"],
            2,
        )
        self.assertEqual(
            compacted[0],
            messages[0],
        )
        self.assertEqual(
            compacted[-2:],
            messages[-2:],
        )
        self.assertEqual(
            compacted[1]["role"],
            "assistant",
        )
        self.assertTrue(
            compacted[1]["content"].startswith(
                context_window
                .TEXT_SUMMARY_MESSAGE_PREFIX
            )
        )
        self.assertGreater(
            report["saved_json_chars"],
            0,
        )
        self.assertEqual(
            report["before_json_chars"]
            - report["after_json_chars"],
            report["saved_json_chars"],
        )

        encoded_report = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertNotIn(
            "PRIVATE-OLD-USER",
            encoded_report,
        )
        self.assertNotIn(
            "PRIVATE-OLD-ASSISTANT",
            encoded_report,
        )

    def test_active_tool_turn_is_preserved_verbatim(
        self,
    ):
        current_turn = [
            {
                "role": "user",
                "content": "current tool request",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_active",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": (
                                '{"path":"/tmp/current"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_active",
                "content": (
                    "ACTIVE-TOOL-RESULT-SECRET"
                ),
            },
        ]
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "old " + ("X" * 400),
            },
            {
                "role": "assistant",
                "content": "old " + ("Y" * 400),
            },
            *current_turn,
        ]

        compacted, report = (
            context_window
            .build_text_summary_compaction_preview(
                messages,
                summary_text="Old exchange summary.",
                tail_messages=1,
            )
        )

        self.assertTrue(
            report["compaction_applied"]
        )
        self.assertTrue(
            report["active_tool_continuation"]
        )
        self.assertEqual(
            report[
                "active_tool_continuation_start"
            ],
            4,
        )
        self.assertEqual(
            report["active_tool_turn_start"],
            3,
        )
        self.assertEqual(
            report["protected_tail_start"],
            3,
        )
        self.assertEqual(
            compacted[-3:],
            current_turn,
        )

    def test_crossing_completed_tool_group_is_kept_whole(
        self,
    ):
        tool_group = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_completed",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_completed",
                "content": "bounded result",
            },
        ]
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "old " + ("A" * 500),
            },
            *tool_group,
            {
                "role": "assistant",
                "content": "final answer",
            },
        ]

        compacted, report = (
            context_window
            .build_text_summary_compaction_preview(
                messages,
                summary_text="Earlier context.",
                tail_messages=2,
            )
        )

        self.assertTrue(
            report["compaction_applied"]
        )
        self.assertEqual(
            report["base_tail_start"],
            3,
        )
        self.assertEqual(
            report["protected_tail_start"],
            2,
        )
        self.assertEqual(
            compacted[-3:-1],
            tool_group,
        )
        self.assertEqual(
            len(report["crossing_tool_groups"]),
            1,
        )

    def test_empty_or_oversized_summary_fails_closed(
        self,
    ):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "old question",
            },
            {
                "role": "assistant",
                "content": "old answer",
            },
            {
                "role": "user",
                "content": "current request",
            },
        ]
        original = copy.deepcopy(messages)

        empty_result, empty_report = (
            context_window
            .build_text_summary_compaction_preview(
                messages,
                summary_text="   ",
                tail_messages=1,
            )
        )

        self.assertEqual(empty_result, original)
        self.assertFalse(
            empty_report["compaction_applied"]
        )
        self.assertEqual(
            empty_report["reason"],
            "empty_summary",
        )

        large_result, large_report = (
            context_window
            .build_text_summary_compaction_preview(
                messages,
                summary_text=("Z" * 5000),
                tail_messages=1,
            )
        )

        self.assertEqual(large_result, original)
        self.assertFalse(
            large_report["compaction_applied"]
        )
        self.assertEqual(
            large_report["reason"],
            "not_smaller",
        )

    def test_second_preview_is_stable(self):
        messages = [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "old " + ("A" * 300),
            },
            {
                "role": "assistant",
                "content": "old " + ("B" * 300),
            },
            {
                "role": "user",
                "content": "recent question",
            },
            {
                "role": "assistant",
                "content": "recent answer",
            },
        ]
        summary = "Stable historical summary."

        first, first_report = (
            context_window
            .build_text_summary_compaction_preview(
                messages,
                summary_text=summary,
                tail_messages=2,
            )
        )
        second, second_report = (
            context_window
            .build_text_summary_compaction_preview(
                first,
                summary_text=summary,
                tail_messages=2,
            )
        )

        self.assertTrue(
            first_report["compaction_applied"]
        )
        self.assertEqual(second, first)
        self.assertFalse(
            second_report["compaction_applied"]
        )
        self.assertEqual(
            second_report["reason"],
            "not_smaller",
        )


class HistoricalToolProtocolCompactionTests(
    unittest.TestCase
):
    @staticmethod
    def _tool_call(
        call_id,
        *,
        arguments,
    ):
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": arguments,
            },
        }

    def test_completed_historical_tool_group_is_compacted(
        self,
    ):
        import copy
        import json

        old_arguments = (
            '{"path":"/tmp/private","padding":"'
            + ("A" * 500)
            + '"}'
        )
        old_result = (
            "PRIVATE-HISTORICAL-RESULT-"
            + ("B" * 700)
        )

        messages = [
            {
                "role": "system",
                "content": "System policy.",
            },
            {
                "role": "user",
                "content": "Read the historical file.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_old",
                        arguments=old_arguments,
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": old_result,
            },
            {
                "role": "assistant",
                "content": (
                    "Historical final answer."
                ),
            },
            {
                "role": "user",
                "content": "Current question.",
            },
        ]
        original = copy.deepcopy(messages)

        compacted, meta = (
            context_window
            .build_historical_tool_protocol_compaction_preview(
                messages,
                tail_messages=2,
            )
        )

        self.assertEqual(messages, original)
        self.assertNotEqual(compacted, original)

        self.assertEqual(
            compacted[2]["tool_calls"][0]["id"],
            "call_old",
        )
        self.assertEqual(
            compacted[2]["tool_calls"][0][
                "function"
            ]["name"],
            "read_file",
        )
        self.assertEqual(
            compacted[2]["tool_calls"][0][
                "function"
            ]["arguments"],
            "{}",
        )
        self.assertEqual(
            compacted[3]["tool_call_id"],
            "call_old",
        )
        self.assertEqual(
            compacted[3]["content"],
            context_window
            .HISTORICAL_TOOL_RESULT_PLACEHOLDER,
        )

        self.assertEqual(
            compacted[4],
            original[4],
        )
        self.assertEqual(
            compacted[5],
            original[5],
        )

        encoded = json.dumps(
            compacted,
            ensure_ascii=False,
        )

        self.assertNotIn(
            "PRIVATE-HISTORICAL-RESULT",
            encoded,
        )
        self.assertNotIn(
            '"padding"',
            encoded,
        )

        self.assertEqual(
            meta["candidate_groups"],
            1,
        )
        self.assertEqual(
            meta["validated_candidate_groups"],
            1,
        )
        self.assertEqual(
            meta["compacted_groups"],
            1,
        )
        self.assertEqual(
            meta["compacted_indices"],
            [2, 3],
        )
        self.assertGreater(
            meta["saved_json_chars"],
            0,
        )

    def test_active_current_tool_group_is_preserved(
        self,
    ):
        import copy

        active_result = (
            "PRIVATE-ACTIVE-RESULT-"
            + ("C" * 700)
        )

        messages = [
            {
                "role": "system",
                "content": "System policy.",
            },
            {
                "role": "user",
                "content": "Old question.",
            },
            {
                "role": "assistant",
                "content": "Old answer.",
            },
            {
                "role": "user",
                "content": "Read current file.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_current",
                        arguments=(
                            '{"path":"/tmp/current"}'
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_current",
                "content": active_result,
            },
        ]
        original = copy.deepcopy(messages)

        compacted, meta = (
            context_window
            .build_historical_tool_protocol_compaction_preview(
                messages,
                tail_messages=2,
            )
        )

        self.assertEqual(messages, original)
        self.assertEqual(compacted, original)
        self.assertTrue(
            meta["active_tool_continuation"]
        )
        self.assertEqual(
            meta["active_tool_continuation_start"],
            4,
        )
        self.assertEqual(
            meta["candidate_groups"],
            0,
        )
        self.assertEqual(
            meta["compacted_groups"],
            0,
        )
        self.assertEqual(
            meta["saved_json_chars"],
            0,
        )

    def test_mixed_history_compacts_only_old_group(
        self,
    ):
        import copy
        import json

        old_result = (
            "PRIVATE-OLD-RESULT-"
            + ("D" * 700)
        )
        current_result = (
            "PRIVATE-CURRENT-RESULT-"
            + ("E" * 700)
        )

        messages = [
            {
                "role": "system",
                "content": "System policy.",
            },
            {
                "role": "user",
                "content": "Old tool request.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_old",
                        arguments=(
                            '{"data":"'
                            + ("F" * 500)
                            + '"}'
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": old_result,
            },
            {
                "role": "assistant",
                "content": "Old final answer.",
            },
            {
                "role": "user",
                "content": "Current tool request.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_current",
                        arguments=(
                            '{"path":"/tmp/current"}'
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_current",
                "content": current_result,
            },
        ]
        original = copy.deepcopy(messages)

        compacted, meta = (
            context_window
            .build_historical_tool_protocol_compaction_preview(
                messages,
                tail_messages=2,
            )
        )

        self.assertEqual(messages, original)
        self.assertEqual(
            compacted[2]["tool_calls"][0][
                "function"
            ]["arguments"],
            "{}",
        )
        self.assertEqual(
            compacted[3]["content"],
            context_window
            .HISTORICAL_TOOL_RESULT_PLACEHOLDER,
        )

        self.assertEqual(
            compacted[6],
            original[6],
        )
        self.assertEqual(
            compacted[7],
            original[7],
        )

        encoded = json.dumps(
            compacted,
            ensure_ascii=False,
        )

        self.assertNotIn(
            "PRIVATE-OLD-RESULT",
            encoded,
        )
        self.assertIn(
            "PRIVATE-CURRENT-RESULT",
            encoded,
        )

        self.assertEqual(
            meta["compacted_indices"],
            [2, 3],
        )
        self.assertTrue(
            meta["active_tool_continuation"]
        )
        self.assertEqual(
            meta["active_tool_continuation_start"],
            6,
        )

    def test_short_group_is_not_expanded(
        self,
    ):
        import copy

        messages = [
            {
                "role": "system",
                "content": "System.",
            },
            {
                "role": "user",
                "content": "Old request.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_short",
                        arguments="{}",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_short",
                "content": "OK",
            },
            {
                "role": "assistant",
                "content": "Old answer.",
            },
            {
                "role": "user",
                "content": "Current request.",
            },
        ]
        original = copy.deepcopy(messages)

        compacted, meta = (
            context_window
            .build_historical_tool_protocol_compaction_preview(
                messages,
                tail_messages=2,
            )
        )

        self.assertEqual(messages, original)
        self.assertEqual(compacted, original)
        self.assertEqual(
            meta["compacted_groups"],
            0,
        )
        self.assertEqual(
            meta["skipped_non_shrinking_groups"],
            1,
        )
        self.assertEqual(
            meta["saved_json_chars"],
            0,
        )

    def test_duplicate_tool_result_fails_closed(
        self,
    ):
        import copy
        from unittest.mock import patch

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_duplicate",
                        arguments=(
                            '{"secret":"'
                            + ("X" * 500)
                            + '"}'
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_duplicate",
                "content": (
                    "FIRST-PRIVATE-RESULT-"
                    + ("Y" * 700)
                ),
            },
            {
                "role": "tool",
                "tool_call_id": "call_duplicate",
                "content": (
                    "SECOND-PRIVATE-RESULT-"
                    + ("Z" * 700)
                ),
            },
        ]
        original = copy.deepcopy(messages)

        fake_report = {
            "configured_tail_messages": 1,
            "active_tool_continuation": False,
            "active_tool_continuation_start": None,
            "verbatim_tail_start": 3,
            "older_tool_protocol_groups": 1,
            "older_tool_protocol_indices": [
                0,
                1,
                2,
            ],
        }

        with patch.object(
            context_window,
            "build_context_window_report",
            return_value=fake_report,
        ):
            compacted, meta = (
                context_window
                .build_historical_tool_protocol_compaction_preview(
                    messages,
                    tail_messages=1,
                )
            )

        self.assertEqual(messages, original)
        self.assertEqual(compacted, original)
        self.assertEqual(
            meta["compacted_groups"],
            0,
        )
        self.assertEqual(
            meta["compacted_indices"],
            [],
        )
        self.assertEqual(
            meta["saved_json_chars"],
            0,
        )
        self.assertEqual(
            meta["invalid_candidate_indices"],
            [0, 1, 2],
        )

    def test_metadata_is_content_free(
        self,
    ):
        import json

        secret = (
            "PRIVATE-METADATA-SECRET-"
            + ("G" * 700)
        )

        messages = [
            {
                "role": "system",
                "content": "System.",
            },
            {
                "role": "user",
                "content": "Old request.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call(
                        "call_secret",
                        arguments=secret,
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_secret",
                "content": secret,
            },
            {
                "role": "assistant",
                "content": "Old answer.",
            },
            {
                "role": "user",
                "content": "Current request.",
            },
        ]

        _, meta = (
            context_window
            .build_historical_tool_protocol_compaction_preview(
                messages,
                tail_messages=2,
            )
        )

        encoded_meta = json.dumps(
            meta,
            ensure_ascii=False,
            sort_keys=True,
        )

        self.assertNotIn(
            "PRIVATE-METADATA-SECRET",
            encoded_meta,
        )
        self.assertEqual(
            meta["compaction_version"],
            (
                "kven2-historical-tool-"
                "protocol-compaction-v1"
            ),
        )


if __name__ == "__main__":
    unittest.main()
