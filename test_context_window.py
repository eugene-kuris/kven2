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
