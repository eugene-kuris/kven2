import atexit
import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import hnsw
import routes

# Importing routes imports hnsw and registers its production persistence
# handler. Unit tests must never persist process-local HNSW state on exit.
atexit.unregister(hnsw.save_hnsw)


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return copy.deepcopy(self._payload)


class PromptCacheContextTests(unittest.IsolatedAsyncioTestCase):
    def test_context_window_report_is_disabled_by_default(self):
        messages = [
            {
                "role": "user",
                "content": "do not inspect this content",
            }
        ]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                "KVEN2_CONTEXT_WINDOW_REPORT_ENABLED",
                None,
            )
            with patch.object(
                routes,
                "build_context_window_report",
            ) as reporter:
                routes._maybe_log_context_window_report(
                    messages,
                    route_label="main",
                )

        reporter.assert_not_called()

    def test_context_window_report_is_content_free(self):
        messages = [
            {
                "role": "system",
                "content": "SECRET-SYSTEM-CONTENT",
            },
            {
                "role": "user",
                "content": "SECRET-USER-CONTENT",
            },
            {
                "role": "assistant",
                "content": "recent reply",
            },
        ]
        original = copy.deepcopy(messages)

        with patch.dict(
            os.environ,
            {
                "KVEN2_CONTEXT_WINDOW_REPORT_ENABLED": "1",
                "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
            },
            clear=False,
        ):
            with self.assertLogs(
                routes.logger,
                level="INFO",
            ) as captured:
                routes._maybe_log_context_window_report(
                    messages,
                    route_label="main",
                )

        self.assertEqual(messages, original)

        output = "\n".join(captured.output)
        self.assertIn(
            "[CONTEXT_WINDOW_REPORT]",
            output,
        )
        self.assertIn(
            '"configured_tail_messages":2',
            output,
        )
        self.assertIn(
            '"route_label":"main"',
            output,
        )
        self.assertNotIn(
            "SECRET-SYSTEM-CONTENT",
            output,
        )
        self.assertNotIn(
            "SECRET-USER-CONTENT",
            output,
        )

    def test_historical_media_compaction_is_disabled_by_default(self):
        messages = [
            {
                "role": "user",
                "content": "unchanged",
            }
        ]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                "KVEN2_HISTORICAL_MEDIA_COMPACTION_ENABLED",
                None,
            )
            with patch.object(
                routes,
                "build_historical_media_compaction_preview",
            ) as compactor:
                result = routes._maybe_compact_historical_media(
                    messages,
                    route_label="main",
                )

        self.assertIs(result, messages)
        compactor.assert_not_called()

    async def test_enabled_historical_media_compaction_reaches_backend(self):
        captured_payloads = []

        async def capture_backend(payload, chat_url, *, route_label):
            captured_payloads.append(copy.deepcopy(payload))
            return (
                [
                    "data: "
                    + json.dumps(
                        {"content": "OK"},
                        ensure_ascii=False,
                    )
                ],
                {"detected": False},
            )

        data_uri = (
            "data:image/png;base64,"
            + ("A" * 100)
        )
        request_payload = {
            "model": "test-model",
            "stream": False,
            "messages": [
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
                    "content": "image answer",
                },
                {
                    "role": "user",
                    "content": "latest request",
                },
            ],
        }
        original_payload = copy.deepcopy(request_payload)
        profile = {
            "agent_name": "Kven II",
            "agent_role": "Research assistant",
            "project_history": "Stable test profile",
            "owner": "Test Owner",
            "mission": "Test media compaction",
        }

        with patch.dict(
            os.environ,
            {
                "KVEN2_HISTORICAL_MEDIA_COMPACTION_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "load_active_state",
            AsyncMock(return_value={}),
        ), patch.object(
            routes,
            "save_history_snapshot",
            AsyncMock(),
        ), patch.object(
            routes,
            "load_agent_profile",
            Mock(return_value=profile),
        ), patch.object(
            routes,
            "get_semantic_context",
            AsyncMock(return_value=""),
        ), patch.object(
            routes,
            "get_project_context",
            AsyncMock(return_value=""),
        ), patch.object(
            routes,
            "retrieve_context",
            AsyncMock(return_value=[]),
        ), patch.object(
            routes,
            "resolve_model_adapter",
            Mock(return_value=SimpleNamespace(adapter_id="test-adapter")),
        ), patch.object(
            routes,
            "_forward_to_backend_and_collect",
            side_effect=capture_backend,
        ):
            response = await routes.handle_chat(
                FakeRequest(request_payload)
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(request_payload, original_payload)
        self.assertEqual(len(captured_payloads), 1)

        encoded_messages = json.dumps(
            captured_payloads[0]["messages"],
            ensure_ascii=False,
        )
        self.assertNotIn(data_uri, encoded_messages)
        self.assertIn(
            "Historical media omitted from active model context",
            encoded_messages,
        )
        self.assertIn("latest request", encoded_messages)

    def test_native_tool_payload_enables_prompt_cache(self):
        source_body = {
            "model": "owui-visible-model",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Найди актуальные сведения.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the web.",
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
                }
            ],
        }

        native_payload = routes._prepare_native_tool_payload(
            source_body,
            source_body["messages"],
            "backend-model",
        )

        self.assertIs(native_payload.get("cache_prompt"), True)
        self.assertEqual(native_payload.get("model"), "backend-model")
        self.assertNotIn("cache_prompt", source_body)

    async def test_stable_time_policy_and_cache_prompt_reach_backend(self):
        captured_payloads = []

        async def capture_backend(payload, chat_url, *, route_label):
            captured_payloads.append(copy.deepcopy(payload))
            return (
                [
                    "data: "
                    + json.dumps(
                        {"content": "OK"},
                        ensure_ascii=False,
                    )
                ],
                {"detected": False},
            )

        request_payload = {
            "model": "test-model",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Ответь кратко.",
                }
            ],
        }

        profile = {
            "agent_name": "Kven II",
            "agent_role": "Research assistant",
            "project_history": "Stable test profile",
            "owner": "Test Owner",
            "mission": "Test prompt cache behavior",
        }

        with patch.object(
            routes,
            "load_active_state",
            AsyncMock(return_value={}),
        ), patch.object(
            routes,
            "save_history_snapshot",
            AsyncMock(),
        ), patch.object(
            routes,
            "load_agent_profile",
            Mock(return_value=profile),
        ), patch.object(
            routes,
            "get_semantic_context",
            AsyncMock(return_value=""),
        ), patch.object(
            routes,
            "get_project_context",
            AsyncMock(return_value=""),
        ), patch.object(
            routes,
            "retrieve_context",
            AsyncMock(return_value=[]),
        ), patch.object(
            routes,
            "resolve_model_adapter",
            Mock(return_value=SimpleNamespace(adapter_id="test-adapter")),
        ), patch.object(
            routes,
            "_forward_to_backend_and_collect",
            side_effect=capture_backend,
        ):
            first_response = await routes.handle_chat(
                FakeRequest(request_payload)
            )
            second_response = await routes.handle_chat(
                FakeRequest(request_payload)
            )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(len(captured_payloads), 2)

        system_prompts = []

        for payload in captured_payloads:
            self.assertIs(payload.get("cache_prompt"), True)

            messages = payload.get("messages")
            self.assertIsInstance(messages, list)
            self.assertGreaterEqual(len(messages), 2)
            self.assertEqual(messages[0].get("role"), "system")

            system_text = messages[0].get("content", "")
            system_prompts.append(system_text)

            self.assertIn(
                "CURRENT DATE AND TIME POLICY:",
                system_text,
            )
            self.assertIn(
                "use the get_time tool if available",
                system_text,
            )
            self.assertNotIn(
                "Current server datetime:",
                system_text,
            )

        self.assertEqual(system_prompts[0], system_prompts[1])


class HistoricalToolProtocolRouteTests(
    unittest.IsolatedAsyncioTestCase
):
    @staticmethod
    def _historical_messages():
        return [
            {
                "role": "user",
                "content": "Historical tool request.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": (
                                '{"path":"/tmp/private",'
                                '"padding":"'
                                + ("A" * 600)
                                + '"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": (
                    "PRIVATE-HISTORICAL-RESULT-"
                    + ("B" * 900)
                ),
            },
            {
                "role": "assistant",
                "content": "Historical final answer.",
            },
            {
                "role": "user",
                "content": "Current request.",
            },
        ]

    def test_tool_compaction_is_disabled_by_default(
        self,
    ):
        messages = self._historical_messages()

        with patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            os.environ.pop(
                (
                    "KVEN2_HISTORICAL_TOOL_"
                    "PROTOCOL_COMPACTION_ENABLED"
                ),
                None,
            )

            with patch.object(
                routes,
                (
                    "build_historical_tool_protocol_"
                    "compaction_preview"
                ),
            ) as compactor:
                result = (
                    routes
                    ._maybe_compact_historical_tool_protocol(
                        messages,
                        route_label="main",
                    )
                )

        self.assertIs(result, messages)
        compactor.assert_not_called()

    def test_enabled_helper_is_content_free_and_non_mutating(
        self,
    ):
        messages = self._historical_messages()
        original = copy.deepcopy(messages)

        with patch.dict(
            os.environ,
            {
                (
                    "KVEN2_HISTORICAL_TOOL_"
                    "PROTOCOL_COMPACTION_ENABLED"
                ): "1",
                (
                    "KVEN2_CONTEXT_WINDOW_"
                    "TAIL_MESSAGES"
                ): "2",
            },
            clear=False,
        ):
            with self.assertLogs(
                routes.logger,
                level="INFO",
            ) as captured:
                compacted = (
                    routes
                    ._maybe_compact_historical_tool_protocol(
                        messages,
                        route_label="main",
                    )
                )

        self.assertEqual(messages, original)
        self.assertNotEqual(compacted, original)

        output = "\n".join(captured.output)

        self.assertIn(
            (
                "[HISTORICAL_TOOL_PROTOCOL_"
                "COMPACTION]"
            ),
            output,
        )
        self.assertIn(
            '"route_label":"main"',
            output,
        )
        self.assertIn(
            '"compacted_groups":1',
            output,
        )
        self.assertNotIn(
            "PRIVATE-HISTORICAL-RESULT",
            output,
        )
        self.assertNotIn(
            '"/tmp/private"',
            output,
        )

    def test_tool_compaction_failure_is_fail_open(
        self,
    ):
        messages = self._historical_messages()

        with patch.dict(
            os.environ,
            {
                (
                    "KVEN2_HISTORICAL_TOOL_"
                    "PROTOCOL_COMPACTION_ENABLED"
                ): "1",
            },
            clear=False,
        ), patch.object(
            routes,
            (
                "build_historical_tool_protocol_"
                "compaction_preview"
            ),
            side_effect=RuntimeError(
                "compaction failure"
            ),
        ):
            with self.assertLogs(
                routes.logger,
                level="WARNING",
            ) as captured:
                result = (
                    routes
                    ._maybe_compact_historical_tool_protocol(
                        messages,
                        route_label="main",
                    )
                )

        self.assertIs(result, messages)
        self.assertIn(
            (
                "[HISTORICAL_TOOL_PROTOCOL_"
                "COMPACTION] failed"
            ),
            "\n".join(captured.output),
        )

    async def test_enabled_tool_compaction_reaches_backend(
        self,
    ):
        normal_backend = AsyncMock(
            side_effect=AssertionError(
                "normal backend path must not be used"
            )
        )
        hybrid_backend = AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
            )
        )

        request_payload = {
            "model": "test-model",
            "stream": False,
            "messages": (
                self._historical_messages()
            ),
        }
        original_payload = copy.deepcopy(
            request_payload
        )

        profile = {
            "agent_name": "Kven II",
            "agent_role": "Research assistant",
            "project_history": (
                "Stable test profile"
            ),
            "owner": "Test Owner",
            "mission": (
                "Test historical tool compaction"
            ),
        }

        environment = {
            (
                "KVEN2_HISTORICAL_TOOL_"
                "PROTOCOL_COMPACTION_ENABLED"
            ): "1",
            (
                "KVEN2_CONTEXT_WINDOW_"
                "TAIL_MESSAGES"
            ): "2",
            (
                "KVEN2_HISTORICAL_MEDIA_"
                "COMPACTION_ENABLED"
            ): "0",
            (
                "KVEN2_CONTEXT_BUDGET_"
                "REPORT_ENABLED"
            ): "0",
        }

        with patch.dict(
            os.environ,
            environment,
            clear=False,
        ), patch.object(
            routes,
            "load_active_state",
            AsyncMock(return_value={}),
        ), patch.object(
            routes,
            "save_history_snapshot",
            AsyncMock(),
        ), patch.object(
            routes,
            "load_agent_profile",
            Mock(return_value=profile),
        ), patch.object(
            routes,
            "get_project_context",
            AsyncMock(return_value=""),
        ), patch.object(
            routes,
            "retrieve_context",
            AsyncMock(return_value=[]),
        ), patch.object(
            routes,
            "resolve_model_adapter",
            Mock(
                return_value=SimpleNamespace(
                    adapter_id="test-adapter"
                )
            ),
        ), patch.object(
            routes,
            "_resolve_main_chat_thinking",
            AsyncMock(return_value=False),
        ), patch.object(
            routes,
            "_forward_to_backend_and_collect",
            normal_backend,
        ), patch.object(
            routes,
            "_proxy_hybrid_native_openai_tool_protocol",
            hybrid_backend,
        ):
            response = await routes.handle_chat(
                FakeRequest(request_payload)
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            request_payload,
            original_payload,
        )

        normal_backend.assert_not_awaited()
        hybrid_backend.assert_awaited_once()

        native_payload = (
            hybrid_backend.await_args.args[0]
        )

        self.assertIsInstance(
            native_payload,
            dict,
        )

        backend_messages = native_payload[
            "messages"
        ]

        tool_call_messages = [
            message
            for message in backend_messages
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
            )
        ]
        tool_result_messages = [
            message
            for message in backend_messages
            if message.get("role") == "tool"
        ]

        self.assertEqual(
            len(tool_call_messages),
            1,
        )
        self.assertEqual(
            len(tool_result_messages),
            1,
        )

        tool_call = tool_call_messages[0][
            "tool_calls"
        ][0]

        self.assertEqual(
            tool_call["id"],
            "call_old",
        )
        self.assertEqual(
            tool_call["function"]["name"],
            "read_file",
        )
        self.assertEqual(
            tool_call["function"]["arguments"],
            "{}",
        )
        self.assertEqual(
            tool_result_messages[0][
                "tool_call_id"
            ],
            "call_old",
        )
        self.assertEqual(
            tool_result_messages[0]["content"],
            (
                "[Historical tool result omitted "
                "from active context.]"
            ),
        )

        encoded_messages = json.dumps(
            backend_messages,
            ensure_ascii=False,
        )

        self.assertNotIn(
            "PRIVATE-HISTORICAL-RESULT",
            encoded_messages,
        )
        self.assertNotIn(
            '"/tmp/private"',
            encoded_messages,
        )
        self.assertIn(
            "Historical final answer.",
            encoded_messages,
        )
        self.assertIn(
            "Current request.",
            encoded_messages,
        )


if __name__ == "__main__":
    unittest.main()
