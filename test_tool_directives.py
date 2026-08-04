import unittest

from tool_directives import (
    ToolDirectiveError,
    parse_tool_directive,
    render_tools_help,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return current time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch one explicit URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum returned characters.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


class ToolDirectiveTests(unittest.TestCase):
    def test_plain_text_has_no_directive(self):
        self.assertIsNone(
            parse_tool_directive(
                "Explain #get_time as plain text.",
                TOOLS,
            )
        )

    def test_tool_directive_is_case_insensitive(self):
        result = parse_tool_directive(
            "  #GET_TIME   What time is it?  ",
            TOOLS,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "tool")
        self.assertEqual(result.tool_name, "get_time")
        self.assertEqual(
            result.remaining_text,
            "What time is it?",
        )

    def test_tool_directive_must_be_allowed(self):
        with self.assertRaisesRegex(
            ToolDirectiveError,
            "unknown or unavailable",
        ):
            parse_tool_directive(
                "#fetch_url https://example.com",
                TOOLS,
                allowed_names={"get_time"},
            )

    def test_unknown_directive_is_error(self):
        with self.assertRaisesRegex(
            ToolDirectiveError,
            "#unknown",
        ):
            parse_tool_directive(
                "#unknown do something",
                TOOLS,
            )

    def test_multiple_directives_are_error(self):
        with self.assertRaisesRegex(
            ToolDirectiveError,
            "multiple tool directives",
        ):
            parse_tool_directive(
                "#get_time #fetch_url https://example.com",
                TOOLS,
            )

    def test_tools_without_name_requests_full_help(self):
        result = parse_tool_directive(
            "#tools",
            TOOLS,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "help")
        self.assertIsNone(result.tool_name)
        self.assertEqual(result.remaining_text, "")

    def test_tools_accepts_one_specific_name(self):
        result = parse_tool_directive(
            "#tools #FETCH_URL",
            TOOLS,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "help")
        self.assertEqual(result.tool_name, "fetch_url")

    def test_tools_rejects_extra_text(self):
        with self.assertRaisesRegex(
            ToolDirectiveError,
            "zero or one tool name",
        ):
            parse_tool_directive(
                "#tools fetch_url extra",
                TOOLS,
            )

    def test_help_lists_only_allowed_tools(self):
        text = render_tools_help(
            TOOLS,
            allowed_names={"get_time"},
        )

        self.assertIn("#get_time <запрос>", text)
        self.assertNotIn("#fetch_url", text)
        self.assertIn(
            "Подробная справка: #tools <имя>",
            text,
        )

    def test_specific_help_includes_schema(self):
        text = render_tools_help(
            TOOLS,
            tool_name="fetch_url",
        )

        self.assertIn("Инструмент:", text)
        self.assertIn("#fetch_url <запрос>", text)
        self.assertIn(
            "Обязательные аргументы: url.",
            text,
        )
        self.assertIn(
            "- url (string, обязательный): URL to fetch.",
            text,
        )
        self.assertIn(
            "- max_chars (integer, необязательный)",
            text,
        )

    def test_latest_user_text_supports_multimodal_content(self):
        from tool_directives import latest_user_message_text

        text = latest_user_message_text(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "#fetch_url ",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AA==",
                            },
                        },
                        {
                            "type": "text",
                            "text": "https://example.com",
                        },
                    ],
                }
            ]
        )

        self.assertEqual(
            text,
            "#fetch_url \nhttps://example.com",
        )

    def test_replace_latest_user_text_preserves_non_text_parts(self):
        from tool_directives import (
            replace_latest_user_message_text,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "#fetch_url https://example.com",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AA==",
                        },
                    },
                ],
            }
        ]

        rewritten = replace_latest_user_message_text(
            messages,
            "https://example.com",
        )

        self.assertEqual(
            rewritten[0]["content"][0]["text"],
            "https://example.com",
        )
        self.assertEqual(
            rewritten[0]["content"][1]["type"],
            "image_url",
        )
        self.assertEqual(
            messages[0]["content"][0]["text"],
            "#fetch_url https://example.com",
        )


if __name__ == "__main__":
    unittest.main()
