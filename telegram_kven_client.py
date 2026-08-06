from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Awaitable, Callable, Protocol

import httpx

from sandbox_client import execute_gateway_tool
from tool_registry import export_openai_tools


DEFAULT_KVEN_CHAT_URL = (
    "http://127.0.0.1:10000/v1/chat/completions"
)
DEFAULT_KVEN_TIMEOUT_SECONDS = 1200.0


class AsyncHttpClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> Any:
        ...


ToolExecutor = Callable[
    [dict[str, Any]],
    Awaitable[dict[str, Any]],
]


class KvenClientError(RuntimeError):
    pass


class TelegramKvenClient:
    def __init__(
        self,
        *,
        model: str,
        client: AsyncHttpClient | None = None,
        tool_executor: ToolExecutor = execute_gateway_tool,
        chat_url: str = DEFAULT_KVEN_CHAT_URL,
        timeout_seconds: float = (
            DEFAULT_KVEN_TIMEOUT_SECONDS
        ),
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "Kven model name must be a non-empty string"
            )

        if (
            not isinstance(chat_url, str)
            or not chat_url.strip()
        ):
            raise ValueError(
                "Kven chat URL must be a non-empty string"
            )

        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "Kven timeout must be positive"
            )

        if not callable(tool_executor):
            raise TypeError(
                "Kven tool executor must be callable"
            )

        self._model = model.strip()
        self._chat_url = chat_url.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._tool_executor = tool_executor
        self._owns_client = client is None
        self._client: AsyncHttpClient = (
            client
            if client is not None
            else httpx.AsyncClient()
        )

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close_method = getattr(
            self._client,
            "aclose",
            None,
        )

        if close_method is not None:
            await close_method()

    async def generate_reply(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        prepared_messages = self._validate_messages(
            messages
        )
        tools = export_openai_tools()

        first_payload = self._build_payload(
            messages=prepared_messages,
            tools=tools,
            tool_choice="auto",
        )
        first_message = await self._request_message(
            first_payload
        )

        first_tool_calls = first_message.get(
            "tool_calls"
        )

        if not first_tool_calls:
            return self._require_answer_content(
                first_message,
                phase="initial response",
            )

        tool_call, requested_call = (
            self._parse_single_tool_call(
                first_tool_calls,
                allowed_names={
                    tool["function"]["name"]
                    for tool in tools
                },
            )
        )

        try:
            tool_result = await self._tool_executor(
                requested_call
            )
        except Exception as exc:
            raise KvenClientError(
                "Kven tool execution failed: "
                f"{type(exc).__name__}: {exc}"
            ) from None

        if not isinstance(tool_result, dict):
            raise KvenClientError(
                "Kven tool executor returned "
                "a non-object result"
            )

        assistant_tool_message = {
            "role": "assistant",
            "content": first_message.get("content"),
            "tool_calls": [deepcopy(tool_call)],
        }
        tool_result_message = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": requested_call["name"],
            "content": json.dumps(
                tool_result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

        continuation_messages = (
            deepcopy(prepared_messages)
            + [
                assistant_tool_message,
                tool_result_message,
            ]
        )

        continuation_payload = self._build_payload(
            messages=continuation_messages,
            tools=tools,
            tool_choice="none",
        )
        final_message = await self._request_message(
            continuation_payload
        )

        if final_message.get("tool_calls"):
            raise KvenClientError(
                "Kven requested another tool after "
                "the single permitted tool call"
            )

        return self._require_answer_content(
            final_message,
            phase="tool continuation",
        )

    async def generate_compaction(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Generate derived JSON without exposing tools or tool side effects."""
        prepared_messages = self._validate_messages(messages)
        message = await self._request_message(self._build_payload(
            messages=prepared_messages, tools=[], tool_choice="none",
        ))
        if message.get("tool_calls"):
            raise KvenClientError("Kven requested a tool during compaction")
        return self._require_answer_content(message, phase="compaction response")

    def _build_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": deepcopy(messages),
            "stream": False,
            "tools": deepcopy(tools),
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }

    async def _request_message(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(
                self._chat_url,
                json=payload,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise KvenClientError(
                "Kven API transport failure: "
                f"{type(exc).__name__}: {exc}"
            ) from None

        try:
            response.raise_for_status()
        except Exception as exc:
            raise KvenClientError(
                "Kven API HTTP failure: "
                f"{type(exc).__name__}: {exc}"
            ) from None

        try:
            body = response.json()
        except Exception as exc:
            raise KvenClientError(
                "Kven API returned invalid JSON: "
                f"{type(exc).__name__}: {exc}"
            ) from None

        return self._extract_message(body)

    @staticmethod
    def _extract_message(
        body: object,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise KvenClientError(
                "Kven API response is not an object"
            )

        choices = body.get("choices")

        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise KvenClientError(
                "Kven API response has no single "
                "valid completion choice"
            )

        message = choices[0].get("message")

        if not isinstance(message, dict):
            raise KvenClientError(
                "Kven API completion has no "
                "assistant message"
            )

        role = message.get("role")

        if role not in (None, "assistant"):
            raise KvenClientError(
                "Kven API completion has an "
                "unexpected message role"
            )

        return deepcopy(message)

    @staticmethod
    def _validate_messages(
        messages: object,
    ) -> list[dict[str, Any]]:
        if (
            not isinstance(messages, list)
            or not messages
        ):
            raise ValueError(
                "Kven messages must be a non-empty list"
            )

        validated = []

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    "Kven message at index "
                    f"{index} is not an object"
                )

            role = message.get("role")
            content = message.get("content")

            if role not in {
                "system",
                "user",
                "assistant",
                "tool",
            }:
                raise ValueError(
                    "Kven message at index "
                    f"{index} has an invalid role"
                )

            if (
                content is not None
                and not isinstance(content, str)
            ):
                raise ValueError(
                    "Kven message at index "
                    f"{index} has invalid content"
                )

            validated.append(deepcopy(message))

        return validated

    @staticmethod
    def _parse_single_tool_call(
        tool_calls: object,
        *,
        allowed_names: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(tool_calls, list)
            or len(tool_calls) != 1
        ):
            raise KvenClientError(
                "Kven must return exactly one "
                "tool call"
            )

        tool_call = tool_calls[0]

        if not isinstance(tool_call, dict):
            raise KvenClientError(
                "Kven tool call is not an object"
            )

        call_id = tool_call.get("id")
        call_type = tool_call.get("type")
        function = tool_call.get("function")

        if (
            not isinstance(call_id, str)
            or not call_id
        ):
            raise KvenClientError(
                "Kven tool call has no valid ID"
            )

        if call_type != "function":
            raise KvenClientError(
                "Kven tool call has an invalid type"
            )

        if not isinstance(function, dict):
            raise KvenClientError(
                "Kven tool call has no function object"
            )

        name = function.get("name")
        raw_arguments = function.get("arguments")

        if (
            not isinstance(name, str)
            or name not in allowed_names
        ):
            raise KvenClientError(
                "Kven requested an unsupported tool"
            )

        if not isinstance(raw_arguments, str):
            raise KvenClientError(
                "Kven tool arguments are not JSON text"
            )

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise KvenClientError(
                "Kven tool arguments contain invalid JSON: "
                f"{exc.msg}"
            ) from None

        if not isinstance(arguments, dict):
            raise KvenClientError(
                "Kven tool arguments must decode "
                "to an object"
            )

        normalized_tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": raw_arguments,
            },
        }
        requested_call = {
            "name": name,
            "arguments": arguments,
        }

        return normalized_tool_call, requested_call

    @staticmethod
    def _require_answer_content(
        message: dict[str, Any],
        *,
        phase: str,
    ) -> str:
        content = message.get("content")

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise KvenClientError(
                f"Kven {phase} contained no usable "
                "assistant answer"
            )

        return content
