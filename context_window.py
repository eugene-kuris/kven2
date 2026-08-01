from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _json_char_count(value: Any) -> int:
    """Return a deterministic serialized character count."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        encoded = str(value)

    return len(encoded)


def _content_char_count(content: Any) -> int:
    """Count visible or structured message content without exposing it."""

    if content is None:
        return 0

    if isinstance(content, str):
        return len(content)

    return _json_char_count(content)


def _message_role(message: Any) -> str:
    if not isinstance(message, dict):
        return "invalid"

    role = str(message.get("role") or "").strip().lower()
    return role or "unknown"


def _leading_system_message_count(messages: list) -> int:
    count = 0

    for message in messages:
        if _message_role(message) != "system":
            break

        count += 1

    return count


def _active_tool_continuation_start(
    messages: list,
) -> int | None:
    """
    Return the assistant tool-call index for an active tool continuation.

    Active tail shape:

        assistant(tool_calls) -> tool [-> tool ...] -> end
    """

    if not messages:
        return None

    index = len(messages) - 1

    while index >= 0 and not isinstance(messages[index], dict):
        index -= 1

    saw_tool_result = False

    while index >= 0:
        message = messages[index]

        if not isinstance(message, dict):
            index -= 1
            continue

        if message.get("role") != "tool":
            break

        saw_tool_result = True
        index -= 1

    if not saw_tool_result:
        return None

    while index >= 0 and not isinstance(messages[index], dict):
        index -= 1

    if index < 0:
        return None

    assistant_message = messages[index]

    if (
        assistant_message.get("role") == "assistant"
        and isinstance(
            assistant_message.get("tool_calls"),
            list,
        )
        and assistant_message.get("tool_calls")
    ):
        return index

    return None


def build_context_window_report(
    messages: list,
    *,
    tail_messages: int = 12,
) -> dict:
    """
    Build a content-free dry-run report for a future sliding window.

    The function never changes the supplied messages. It separates:

    - leading system prefix;
    - older summarization candidate;
    - verbatim recent tail;
    - active tool continuation that must remain indivisible.
    """

    if not isinstance(messages, list):
        messages = []

    safe_tail_messages = max(1, int(tail_messages))
    message_count = len(messages)
    role_counts = Counter(
        _message_role(message)
        for message in messages
    )

    content_chars = [
        _content_char_count(
            message.get("content")
            if isinstance(message, dict)
            else message
        )
        for message in messages
    ]
    message_json_chars = [
        _json_char_count(message)
        for message in messages
    ]

    system_prefix_messages = (
        _leading_system_message_count(messages)
    )
    active_tool_start = (
        _active_tool_continuation_start(messages)
    )

    tail_start = max(
        system_prefix_messages,
        message_count - safe_tail_messages,
    )

    if (
        active_tool_start is not None
        and active_tool_start < tail_start
    ):
        tail_start = active_tool_start

    older_start = system_prefix_messages
    older_end = tail_start

    return {
        "messages_total": message_count,
        "role_counts": dict(sorted(role_counts.items())),
        "content_chars_total": sum(content_chars),
        "message_json_chars_total": sum(
            message_json_chars
        ),
        "system_prefix_messages": (
            system_prefix_messages
        ),
        "system_prefix_content_chars": sum(
            content_chars[:system_prefix_messages]
        ),
        "system_prefix_json_chars": sum(
            message_json_chars[:system_prefix_messages]
        ),
        "older_candidate_messages": max(
            0,
            older_end - older_start,
        ),
        "older_candidate_content_chars": sum(
            content_chars[older_start:older_end]
        ),
        "older_candidate_json_chars": sum(
            message_json_chars[older_start:older_end]
        ),
        "verbatim_tail_start": tail_start,
        "verbatim_tail_messages": max(
            0,
            message_count - tail_start,
        ),
        "verbatim_tail_content_chars": sum(
            content_chars[tail_start:]
        ),
        "verbatim_tail_json_chars": sum(
            message_json_chars[tail_start:]
        ),
        "verbatim_tail_roles": [
            _message_role(message)
            for message in messages[tail_start:]
        ],
        "active_tool_continuation": (
            active_tool_start is not None
        ),
        "active_tool_continuation_start": (
            active_tool_start
        ),
        "configured_tail_messages": safe_tail_messages,
    }
