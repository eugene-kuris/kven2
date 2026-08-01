"""Compact tool routing through the dedicated planner model on Tesla P4."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import httpx

from config import settings


logger = logging.getLogger(__name__)

PLANNER_MODEL = os.getenv(
    "KVEN2_PLANNER_MODEL",
    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf",
)
PLANNER_CHAT_URL = (
    os.getenv("KVEN2_PLANNER_URL", settings.SMALL_MODEL_URL).rstrip("/")
    + "/chat/completions"
)

DEFAULT_TIMEOUT_SECONDS = 20.0
SELECTION_MAX_TOKENS = 24
ARGUMENTS_MAX_TOKENS = 192
THINKING_MAX_TOKENS = 8
MAX_CONTEXT_CHARS = 6000
MAX_SELECTION_CONTEXT_CHARS = 1200


class PlannerRouterError(RuntimeError):
    """Planner routing failed or returned an invalid result."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)

        return "\n".join(part for part in parts if part).strip()

    return ""


def _compact_conversation_context(
    messages: list[dict],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Keep only the recent non-system conversation needed for routing."""

    selected: list[str] = []

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "tool"}:
            continue

        text = _content_to_text(message.get("content"))
        if not text:
            continue

        selected.append(f"{role.upper()}: {text}")

        # Usually the last user request is enough. Keep a little nearby context
        # for references such as "check this URL" or "do the same for it".
        if role == "user" and len(selected) >= 3:
            break
        if len(selected) >= 5:
            break

    selected.reverse()
    context = "\n".join(selected).strip()

    if len(context) > max_chars:
        context = context[-max_chars:]

    return context


def _normalize_tools(
    tools: list[dict],
    allowed_names: set[str] | None = None,
) -> dict[str, dict]:
    normalized: dict[str, dict] = {}

    for item in tools:
        if not isinstance(item, dict):
            continue

        function = item.get("function")
        if not isinstance(function, dict):
            continue

        name = str(function.get("name") or "").strip()
        if not name:
            continue

        if allowed_names is not None and name not in allowed_names:
            continue

        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {
                "type": "object",
                "properties": {},
            }

        normalized[name] = {
            "name": name,
            "description": str(function.get("description") or "").strip(),
            "parameters": parameters,
        }

    return normalized



def _compact_tool_catalog(tools_by_name: dict[str, dict]) -> list[dict]:
    catalog: list[dict] = []

    for name in sorted(tools_by_name):
        catalog.append(
            {
                "id": len(catalog),
                "name": name,
            }
        )

    return catalog


def _extract_single_protocol_line(text: str) -> str:
    """Return one non-empty planner protocol line or fail closed."""

    lines = [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise PlannerRouterError(
            "planner did not return exactly one protocol line"
        )

    return lines[0]


def _parse_thinking_protocol(text: str) -> str:
    line = _extract_single_protocol_line(text).upper()

    if line not in {"FAST", "THINK"}:
        raise PlannerRouterError(
            f"unknown thinking mode: {line!r}"
        )

    return line


def _parse_selection_protocol(
    text: str,
    catalog: list[dict],
) -> tuple[str, str | None]:
    line = _extract_single_protocol_line(text).upper()

    if line in {"FAST", "THINK"}:
        return line, None

    parts = line.split()
    if len(parts) != 2 or parts[0] != "TOOL":
        raise PlannerRouterError(
            f"unknown planner protocol response: {line!r}"
        )

    try:
        tool_index = int(parts[1], 10)
    except ValueError as exc:
        raise PlannerRouterError(
            f"invalid planner tool index: {parts[1]!r}"
        ) from exc

    if tool_index < 0 or tool_index >= len(catalog):
        raise PlannerRouterError(
            f"planner tool index is out of range: {tool_index}"
        )

    return "TOOL", str(catalog[tool_index]["name"])


def _matches_json_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str) and bool(value.strip())
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "null":
        return value is None

    # Unknown or compound schemas are left to the tool implementation.
    return True


def _validate_arguments(tool: dict, arguments: Any) -> dict:
    if not isinstance(arguments, dict):
        raise PlannerRouterError("planner tool arguments are not an object")

    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        return arguments

    required = parameters.get("required")
    if not isinstance(required, list):
        required = []

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    for field in required:
        if not isinstance(field, str):
            continue

        if field not in arguments:
            raise PlannerRouterError(
                f"required argument is missing: {field}"
            )

        schema = properties.get(field)
        if not isinstance(schema, dict):
            continue

        expected_type = schema.get("type")

        if isinstance(expected_type, str):
            if not _matches_json_type(arguments[field], expected_type):
                raise PlannerRouterError(
                    f"argument {field!r} does not match type "
                    f"{expected_type!r}"
                )

        elif isinstance(expected_type, list):
            accepted = [
                item
                for item in expected_type
                if isinstance(item, str)
            ]
            if accepted and not any(
                _matches_json_type(arguments[field], item)
                for item in accepted
            ):
                raise PlannerRouterError(
                    f"argument {field!r} does not match types "
                    f"{accepted!r}"
                )

    return arguments


async def _post_planner_text(
    prompt: str,
    *,
    max_tokens: int,
    timeout_seconds: float,
) -> tuple[str, dict]:
    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "stop": ["\n"],
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "reasoning_format": "none",
    }

    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(PLANNER_CHAT_URL, json=payload)

    elapsed = time.perf_counter() - started
    response.raise_for_status()
    response_json = response.json()

    choice = (response_json.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")

    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    meta = {
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (
            usage.get("prompt_tokens_details") or {}
        ).get("cached_tokens"),
    }

    return content, meta


def _extract_native_tool_arguments(
    response_json: dict,
    selected_name: str,
) -> dict:
    """Extract one forced native tool call and fail closed."""

    choice = (response_json.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls")

    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise PlannerRouterError(
            "planner did not return exactly one native tool call"
        )

    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise PlannerRouterError("planner native tool call is invalid")

    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise PlannerRouterError(
            "planner native tool call has no function object"
        )

    returned_name = str(function.get("name") or "").strip()
    if returned_name != selected_name:
        raise PlannerRouterError(
            "planner changed the selected tool from "
            f"{selected_name!r} to {returned_name!r}"
        )

    raw_arguments = function.get("arguments")

    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise PlannerRouterError(
                "planner native tool arguments are not valid JSON"
            ) from exc
    else:
        raise PlannerRouterError(
            "planner native tool arguments are missing"
        )

    if not isinstance(arguments, dict):
        raise PlannerRouterError(
            "planner native tool arguments are not an object"
        )

    return arguments


async def _post_planner_tool_call(
    prompt: str,
    tool: dict,
    *,
    max_tokens: int,
    timeout_seconds: float,
) -> tuple[dict, dict]:
    """Ask llama.cpp for one forced native tool call."""

    selected_name = str(tool.get("name") or "").strip()
    if not selected_name:
        raise PlannerRouterError("selected tool has no name")

    planner_tool = {
        "type": "function",
        "function": {
            "name": selected_name,
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        },
    }

    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "tools": [planner_tool],
        "tool_choice": "required",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "reasoning_format": "none",
    }

    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(PLANNER_CHAT_URL, json=payload)

    elapsed = time.perf_counter() - started
    response.raise_for_status()
    response_json = response.json()

    arguments = _extract_native_tool_arguments(
        response_json,
        selected_name,
    )

    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    choice = (response_json.get("choices") or [{}])[0]
    meta = {
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (
            usage.get("prompt_tokens_details") or {}
        ).get("cached_tokens"),
    }

    return arguments, meta


def _thinking_prompt(context: str) -> str:
    return (
        "You are the Kven II answer-mode classifier.\n"
        "Decide whether the main model needs extended internal reasoning "
        "for the user's latest request.\n\n"
        "Rules:\n"
        "- Do not solve the user's task.\n"
        "- Select FAST only when the answer is explicitly present in the "
        "request or the task is a purely linguistic transformation with "
        "no inference.\n"
        "- FAST is suitable for greetings, acknowledgements, translation, "
        "simple paraphrasing, and extracting an explicitly stated value.\n"
        "- Applying any rule, even a single short or obvious rule, always "
        "requires THINK.\n"
        "- Questions asking what is allowed, what will happen, which route "
        "applies, or what capacity results from a technical state require "
        "THINK.\n"
        "- Select THINK for calculations, diagnosis, routing, permissions, "
        "configuration, code, storage, security, option comparison, and "
        "causal analysis.\n"
        "- THINK examples: applying UNIX rwx permissions and group "
        "membership; determining git merge --ff-only behavior; "
        "longest-prefix routing; RAID calculation; systemd dependency "
        "behavior.\n"
        "- When uncertain, select THINK.\n\n"
        "Return exactly one line containing one token:\n"
        "FAST\n"
        "or\n"
        "THINK\n\n"
        f"Conversation context:\n{context}"
    )


async def classify_main_thinking(
    messages: list[dict],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Classify the main-model answer as FAST or THINK.

    Result forms:
      {"mode": "FAST", "meta": {...}}
      {"mode": "THINK", "meta": {...}}
      {"mode": "ERROR", "error": "...", "meta": {...}}
    """

    context = _compact_conversation_context(messages)
    selection_meta: dict = {}

    try:
        response_text, selection_meta = await _post_planner_text(
            _thinking_prompt(context),
            max_tokens=THINKING_MAX_TOKENS,
            timeout_seconds=timeout_seconds,
        )

        mode = _parse_thinking_protocol(response_text)

        logger.info(
            "[PLANNER_THINKING] mode=%s elapsed=%s "
            "prompt_tokens=%s cached_tokens=%s",
            mode,
            selection_meta.get("elapsed_seconds"),
            selection_meta.get("prompt_tokens"),
            selection_meta.get("cached_tokens"),
        )

        return {
            "mode": mode,
            "meta": {
                "selection": selection_meta,
            },
        }

    except Exception as exc:
        logger.warning(
            "[PLANNER_THINKING] classification_failed error=%s",
            exc,
            exc_info=True,
        )

        return {
            "mode": "ERROR",
            "error": str(exc),
            "meta": {
                "selection": selection_meta,
            },
        }



def _selection_prompt(
    context: str,
    catalog: list[dict],
) -> str:
    selection_context = context[-MAX_SELECTION_CONTEXT_CHARS:]
    tool_lines = "\n".join(
        f"{item['id']} {item['name']}"
        for item in catalog
    )

    return (
        "Route the latest user request; do not solve it.\n"
        "Return exactly one line: FAST, THINK, or TOOL <numeric_id>.\n"
        "Use TOOL only for current, external, or private data; URL or "
        "file content; or an external action.\n"
        "Use FAST for greetings, acknowledgements, translation, "
        "paraphrasing, direct extraction, or trivial arithmetic.\n"
        "Use THINK for diagnosis, configuration, code, security, "
        "comparison, multi-step calculation, causal reasoning, or "
        "uncertainty.\n"
        "Never invent a tool id and select at most one tool.\n\n"
        "Available tools:\n"
        f"{tool_lines}"
        f"\n\nConversation context:\n{selection_context}"
    )


def _arguments_prompt(context: str) -> str:
    return (
        "Generate arguments for the provided Kven II tool.\n"
        "Call the provided tool exactly once.\n"
        "Do not select another tool and do not answer the user's task.\n"
        "Use only information grounded in the request or nearby "
        "conversation context.\n\n"
        f"Conversation context:\n{context}"
    )


async def route_tool_request(
    messages: list[dict],
    tools: list[dict],
    *,
    allowed_names: set[str] | None = None,
    explicit_tool_name: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Return a compact planner decision.

    Result forms:
      {"decision": "NO_TOOL", "mode": "FAST|THINK", "meta": {...}}
      {"decision": "TOOL", "tool_call": {...}, "meta": {...}}
      {"decision": "ERROR", "error": "...", "meta": {...}}
    """

    tools_by_name = _normalize_tools(tools, allowed_names)
    context = _compact_conversation_context(messages)

    if not tools_by_name:
        return {
            "decision": "NO_TOOL",
            "mode": "THINK",
            "meta": {
                "reason": "no_allowed_tools",
            },
        }

    if not context:
        return {
            "decision": "ERROR",
            "error": "no recent conversation context",
            "meta": {},
        }

    selection_meta: dict = {}
    arguments_meta: dict = {}

    try:
        selected_name = str(explicit_tool_name or "").strip()

        if selected_name:
            if selected_name not in tools_by_name:
                raise PlannerRouterError(
                    f"explicit tool is not allowed: {selected_name}"
                )
        else:
            catalog = _compact_tool_catalog(tools_by_name)
            response_text, selection_meta = await _post_planner_text(
                _selection_prompt(context, catalog),
                max_tokens=SELECTION_MAX_TOKENS,
                timeout_seconds=timeout_seconds,
            )

            decision, selected_name = _parse_selection_protocol(
                response_text,
                catalog,
            )

            if decision in {"FAST", "THINK"}:
                logger.info(
                    "[PLANNER_ROUTER] no_tool mode=%s elapsed=%s "
                    "prompt_tokens=%s cached_tokens=%s",
                    decision,
                    selection_meta.get("elapsed_seconds"),
                    selection_meta.get("prompt_tokens"),
                    selection_meta.get("cached_tokens"),
                )
                return {
                    "decision": "NO_TOOL",
                    "mode": decision,
                    "meta": {
                        "selection": selection_meta,
                    },
                }

            if not selected_name or selected_name not in tools_by_name:
                raise PlannerRouterError(
                    f"planner selected unknown tool: {selected_name!r}"
                )

        selected_tool = tools_by_name[selected_name]

        generated_arguments, arguments_meta = (
            await _post_planner_tool_call(
                _arguments_prompt(context),
                selected_tool,
                max_tokens=ARGUMENTS_MAX_TOKENS,
                timeout_seconds=timeout_seconds,
            )
        )

        arguments = _validate_arguments(
            selected_tool,
            generated_arguments,
        )

        tool_call = {
            "id": f"call_planner_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {
                "name": selected_name,
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        }

        logger.info(
            "[PLANNER_ROUTER] tool_selected name=%s "
            "selection_elapsed=%s arguments_elapsed=%s",
            selected_name,
            selection_meta.get("elapsed_seconds"),
            arguments_meta.get("elapsed_seconds"),
        )

        return {
            "decision": "TOOL",
            "tool_call": tool_call,
            "meta": {
                "selection": selection_meta,
                "arguments": arguments_meta,
            },
        }

    except Exception as exc:
        logger.warning(
            "[PLANNER_ROUTER] routing_failed error=%s",
            exc,
            exc_info=True,
        )
        return {
            "decision": "ERROR",
            "error": str(exc),
            "meta": {
                "selection": selection_meta,
                "arguments": arguments_meta,
            },
        }
