import atexit
import copy
import json
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


if __name__ == "__main__":
    unittest.main()
