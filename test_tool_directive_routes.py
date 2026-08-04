import unittest

import routes
from tool_registry import export_openai_tools


def make_payload(
    text: str,
    *,
    tool_choice="auto",
) -> dict:
    return {
        "model": "test-model",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ],
        "tools": export_openai_tools(),
        "tool_choice": tool_choice,
    }


class ToolDirectiveRouteTests(unittest.TestCase):
    def test_tools_returns_local_dynamic_help(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload("#tools")
        )

        self.assertEqual(result["kind"], "help")
        content = result["content"]

        for name in (
            "get_time",
            "read_file",
            "fetch_url",
            "web_search",
        ):
            self.assertIn(f"#{name}", content)

    def test_specific_help_returns_one_tool(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload("#tools fetch_url")
        )

        self.assertEqual(result["kind"], "help")
        self.assertIn("#fetch_url", result["content"])
        self.assertNotIn("#get_time", result["content"])

    def test_tool_directive_forces_choice_and_strips_token(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload(
                "#fetch_url https://example.com — summarize"
            )
        )

        self.assertEqual(result["kind"], "tool")
        self.assertEqual(result["tool_name"], "fetch_url")

        prepared = result["payload"]

        self.assertEqual(
            prepared["tool_choice"]["function"]["name"],
            "fetch_url",
        )
        self.assertEqual(
            prepared["messages"][0]["content"],
            "https://example.com — summarize",
        )

    def test_get_time_without_tail_gets_default_request(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload("#get_time")
        )

        self.assertEqual(result["kind"], "tool")
        self.assertEqual(
            result["payload"]["messages"][0]["content"],
            "Назови текущую дату и время.",
        )

    def test_non_time_tool_requires_tail(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload("#read_file")
        )

        self.assertEqual(result["kind"], "error")
        self.assertIn(
            "requires a request",
            result["content"],
        )

    def test_tool_choice_none_conflict_is_visible(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload(
                "#get_time now",
                tool_choice="none",
            )
        )

        self.assertEqual(result["kind"], "error")
        self.assertIn(
            "tool_choice=none",
            result["content"],
        )

    def test_different_explicit_api_choice_conflicts(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload(
                "#get_time now",
                tool_choice={
                    "type": "function",
                    "function": {
                        "name": "fetch_url",
                    },
                },
            )
        )

        self.assertEqual(result["kind"], "error")
        self.assertIn(
            "tool_choice=fetch_url",
            result["content"],
        )

    def test_same_explicit_api_choice_is_accepted(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload(
                "#get_time now",
                tool_choice={
                    "type": "function",
                    "function": {
                        "name": "get_time",
                    },
                },
            )
        )

        self.assertEqual(result["kind"], "tool")
        self.assertEqual(result["tool_name"], "get_time")

    def test_unknown_directive_is_visible_error(self):
        result = routes._prepare_explicit_tool_directive(
            make_payload("#missing do something")
        )

        self.assertEqual(result["kind"], "error")
        self.assertIn("#missing", result["content"])
        self.assertIn("#tools", result["content"])

    def test_continuation_strips_completed_directive(self):
        payload = make_payload(
            "#fetch_url https://example.com — summarize",
            tool_choice="none",
        )
        payload["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "fetch_url",
                                "arguments": (
                                    '{"url":"https://example.com"}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_test",
                    "name": "fetch_url",
                    "content": "{}",
                },
            ]
        )

        stripped = routes._strip_completed_tool_directive(
            payload
        )

        self.assertEqual(
            stripped["messages"][0]["content"],
            "https://example.com — summarize",
        )


if __name__ == "__main__":
    unittest.main()
