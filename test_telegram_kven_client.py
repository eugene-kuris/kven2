import json
import unittest
from typing import Any

from telegram_kven_client import (
    KvenClientError,
    TelegramKvenClient,
)
from tool_registry import export_openai_tools


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
    ):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(
            payload,
            ensure_ascii=False,
        )

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(
                f"HTTP status {self.status_code}"
            )

    def json(self) -> Any:
        return self.payload


class FakeHttpClient:
    def __init__(self, *results: Any):
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        if not self.results:
            raise AssertionError(
                "Unexpected Kven API request"
            )

        result = self.results.pop(0)

        if isinstance(result, BaseException):
            raise result

        return result


class FakeToolExecutor:
    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        requested_call: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(requested_call)
        return self.result


def completion(
    *,
    content: Any = None,
    tool_calls: Any = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }


class ToolRegistryExportTests(unittest.TestCase):
    def test_export_contains_enabled_kven_tools(self):
        tools = export_openai_tools()

        names = [
            tool["function"]["name"]
            for tool in tools
        ]

        self.assertEqual(
            names,
            [
                "get_time",
                "read_file",
                "web_search",
                "fetch_url",
            ],
        )

        for tool in tools:
            self.assertEqual(
                tool["type"],
                "function",
            )
            function = tool["function"]
            self.assertIsInstance(
                function["description"],
                str,
            )
            self.assertIsInstance(
                function["parameters"],
                dict,
            )

    def test_export_is_a_defensive_copy(self):
        first = export_openai_tools()
        first[0]["function"]["parameters"][
            "properties"
        ]["corruption"] = {
            "type": "string",
        }

        second = export_openai_tools()

        self.assertNotIn(
            "corruption",
            second[0]["function"][
                "parameters"
            ]["properties"],
        )


class TelegramKvenClientTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_direct_answer_uses_native_tools(
        self,
    ):
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content="Direct answer",
                )
            )
        )
        executor = FakeToolExecutor(
            {
                "status": "ok",
            }
        )
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )
        messages = [
            {
                "role": "user",
                "content": "Hello",
            }
        ]

        answer = await client.generate_reply(messages)

        self.assertEqual(answer, "Direct answer")
        self.assertEqual(executor.calls, [])
        self.assertEqual(len(http.calls), 1)

        call = http.calls[0]

        self.assertEqual(
            call["url"],
            (
                "http://127.0.0.1:10000/"
                "v1/chat/completions"
            ),
        )
        self.assertEqual(call["timeout"], 1200.0)
        self.assertEqual(
            call["json"]["model"],
            "TEST_MODEL",
        )
        self.assertEqual(
            call["json"]["messages"],
            messages,
        )
        self.assertFalse(call["json"]["stream"])
        self.assertEqual(
            call["json"]["tool_choice"],
            "auto",
        )
        self.assertFalse(
            call["json"]["parallel_tool_calls"]
        )
        self.assertEqual(
            call["json"]["tools"],
            export_openai_tools(),
        )

    async def test_one_tool_call_is_executed_and_continued(
        self,
    ):
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": "{}",
            },
        }
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            ),
            FakeResponse(
                completion(
                    content="It is noon.",
                )
            ),
        )
        tool_result = {
            "tool_name": "get_time",
            "status": "ok",
            "result": {
                "time": "12:00",
            },
        }
        executor = FakeToolExecutor(tool_result)
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )
        messages = [
            {
                "role": "user",
                "content": "What time is it?",
            }
        ]

        answer = await client.generate_reply(messages)

        self.assertEqual(answer, "It is noon.")
        self.assertEqual(
            executor.calls,
            [
                {
                    "name": "get_time",
                    "arguments": {},
                }
            ],
        )
        self.assertEqual(len(http.calls), 2)

        continuation = http.calls[1]["json"]

        self.assertEqual(
            continuation["tool_choice"],
            "none",
        )
        self.assertFalse(
            continuation["parallel_tool_calls"]
        )
        self.assertEqual(
            continuation["messages"][:-2],
            messages,
        )
        self.assertEqual(
            continuation["messages"][-2],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            },
        )

        tool_message = continuation[
            "messages"
        ][-1]

        self.assertEqual(
            tool_message["role"],
            "tool",
        )
        self.assertEqual(
            tool_message["tool_call_id"],
            "call_123",
        )
        self.assertEqual(
            tool_message["name"],
            "get_time",
        )
        self.assertEqual(
            json.loads(tool_message["content"]),
            tool_result,
        )

    async def test_tool_error_is_returned_to_kven(
        self,
    ):
        tool_call = {
            "id": "call_error",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps(
                    {
                        "path": "/missing.txt",
                    }
                ),
            },
        }
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            ),
            FakeResponse(
                completion(
                    content="The file could not be read.",
                )
            ),
        )
        tool_result = {
            "tool_name": "read_file",
            "status": "error",
            "error": "File not found",
        }
        executor = FakeToolExecutor(tool_result)
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )

        answer = await client.generate_reply(
            [
                {
                    "role": "user",
                    "content": "Read the file",
                }
            ]
        )

        self.assertEqual(
            answer,
            "The file could not be read.",
        )
        self.assertEqual(len(http.calls), 2)
        self.assertEqual(
            json.loads(
                http.calls[1]["json"][
                    "messages"
                ][-1]["content"]
            ),
            tool_result,
        )

    async def test_unknown_tool_is_rejected(
        self,
    ):
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "delete_everything",
                                "arguments": "{}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            )
        )
        executor = FakeToolExecutor({})
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )

        with self.assertRaises(
            KvenClientError
        ):
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

        self.assertEqual(executor.calls, [])

    async def test_invalid_tool_arguments_are_rejected(
        self,
    ):
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": "not-json",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            )
        )
        executor = FakeToolExecutor({})
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )

        with self.assertRaises(
            KvenClientError
        ):
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

        self.assertEqual(executor.calls, [])

    async def test_multiple_tool_calls_are_rejected(
        self,
    ):
        calls = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": "{}",
                },
            }
            for index in range(2)
        ]
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=calls,
                    finish_reason="tool_calls",
                )
            )
        )
        executor = FakeToolExecutor({})
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )

        with self.assertRaises(
            KvenClientError
        ):
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

        self.assertEqual(executor.calls, [])

    async def test_second_tool_call_is_rejected(
        self,
    ):
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": "{}",
            },
        }
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            ),
            FakeResponse(
                completion(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            ),
        )
        executor = FakeToolExecutor(
            {
                "status": "ok",
            }
        )
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=executor,
        )

        with self.assertRaises(
            KvenClientError
        ):
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

        self.assertEqual(len(executor.calls), 1)

    async def test_empty_answer_is_rejected(
        self,
    ):
        http = FakeHttpClient(
            FakeResponse(
                completion(
                    content="   ",
                )
            )
        )
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=FakeToolExecutor({}),
        )

        with self.assertRaises(
            KvenClientError
        ):
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

    async def test_malformed_response_is_rejected(
        self,
    ):
        http = FakeHttpClient(
            FakeResponse(
                {
                    "unexpected": True,
                }
            )
        )
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=FakeToolExecutor({}),
        )

        with self.assertRaises(
            KvenClientError
        ):
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

    async def test_transport_failure_is_wrapped(
        self,
    ):
        http = FakeHttpClient(
            RuntimeError(
                "connection refused"
            )
        )
        client = TelegramKvenClient(
            model="TEST_MODEL",
            client=http,
            tool_executor=FakeToolExecutor({}),
        )

        with self.assertRaises(
            KvenClientError
        ) as context:
            await client.generate_reply(
                [
                    {
                        "role": "user",
                        "content": "test",
                    }
                ]
            )

        self.assertIn(
            "transport",
            str(context.exception).lower(),
        )


    async def test_compaction_marks_internal_request_and_disables_tools(
        self,
    ):
        class CapturingClient(TelegramKvenClient):
            async def _request_message(self, payload):
                self.captured_payload = payload
                return {
                    "role": "assistant",
                    "content": "{}",
                }

        client = CapturingClient(
            model="TEST_MODEL",
            client=object(),
            tool_executor=lambda request: None,
        )

        result = await client.generate_compaction(
            [
                {
                    "role": "system",
                    "content": "compaction instructions",
                },
                {
                    "role": "user",
                    "content": "#tools https://example.invalid memory retrieval",
                },
            ]
        )

        self.assertEqual(result, "{}")
        self.assertEqual(
            client.captured_payload.get("kven_internal_request"),
            "telegram_compaction",
        )
        self.assertEqual(client.captured_payload.get("tools"), [])
        self.assertEqual(
            client.captured_payload.get("tool_choice"),
            "none",
        )
        self.assertEqual(
            client.captured_payload.get("max_tokens"),
            4096,
        )
        self.assertEqual(
            client.captured_payload.get("temperature"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
