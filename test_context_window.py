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


if __name__ == "__main__":
    unittest.main()
