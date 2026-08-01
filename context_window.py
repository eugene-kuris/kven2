from __future__ import annotations

import copy
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


_MEDIA_PART_TYPES = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "input_audio",
    "video",
    "file",
    "input_file",
}


def _media_string_values(value: Any):
    """Yield media payload strings without exposing them."""

    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() == "type":
                continue
            yield from _media_string_values(nested)
        return

    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _media_string_values(nested)


def _content_manifest(content: Any) -> dict:
    """Describe text and media sizes without retaining content."""

    content_json_chars = _json_char_count(content)
    text_chars = 0
    media_payload_chars = 0
    media_data_uri_chars = 0
    media_reference_chars = 0
    media_data_uri_count = 0
    media_reference_count = 0
    part_types: Counter = Counter()

    if content is None:
        content_kind = "none"
        parts = []
    elif isinstance(content, str):
        content_kind = "text"
        parts = [content]
    elif isinstance(content, list):
        content_kind = "list"
        parts = content
    elif isinstance(content, dict):
        content_kind = "dict"
        parts = [content]
    else:
        content_kind = type(content).__name__
        parts = [content]

    for part in parts:
        if isinstance(part, str):
            part_types["text"] += 1
            text_chars += len(part)
            continue

        if not isinstance(part, dict):
            part_types[type(part).__name__] += 1
            continue

        part_type = str(
            part.get("type") or "unknown"
        ).strip().lower()
        part_types[part_type] += 1

        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_chars += len(text)
            elif text is not None:
                text_chars += _json_char_count(text)
            continue

        if part_type not in _MEDIA_PART_TYPES:
            continue

        for value in _media_string_values(part):
            value_chars = len(value)
            media_payload_chars += value_chars

            if value.lstrip().lower().startswith("data:"):
                media_data_uri_chars += value_chars
                media_data_uri_count += 1
            else:
                media_reference_chars += value_chars
                media_reference_count += 1

    return {
        "content_kind": content_kind,
        "content_json_chars": content_json_chars,
        "text_chars": text_chars,
        "media_payload_chars": media_payload_chars,
        "media_data_uri_chars": media_data_uri_chars,
        "media_reference_chars": media_reference_chars,
        "media_data_uri_count": media_data_uri_count,
        "media_reference_count": media_reference_count,
        "other_or_structure_chars": max(
            0,
            content_json_chars
            - text_chars
            - media_payload_chars,
        ),
        "part_type_counts": dict(
            sorted(part_types.items())
        ),
    }


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


def _completed_tool_protocol_groups(
    messages: list,
) -> list[dict]:
    """Return completed assistant(tool_calls) -> tool groups."""

    groups = []
    index = 0

    while index < len(messages):
        message = messages[index]

        if not (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("tool_calls"), list)
            and message.get("tool_calls")
        ):
            index += 1
            continue

        tool_indices = []
        cursor = index + 1

        while cursor < len(messages):
            candidate = messages[cursor]

            if not isinstance(candidate, dict):
                cursor += 1
                continue

            if candidate.get("role") != "tool":
                break

            tool_indices.append(cursor)
            cursor += 1

        if tool_indices:
            groups.append(
                {
                    "assistant_index": index,
                    "tool_indices": tool_indices,
                    "message_indices": [
                        index,
                        *tool_indices,
                    ],
                }
            )
            index = cursor
            continue

        index += 1

    return groups


_HISTORICAL_MEDIA_PLACEHOLDER = (
    "[Historical media omitted from active model context after its turn "
    "completed. The original remains visible in the chat UI.]"
)


def _replace_media_content_with_placeholder(
    content: Any,
    *,
    placeholder: str,
) -> tuple[Any, int]:
    """Return copied content with media parts replaced by one text marker."""

    if isinstance(content, list):
        copied_parts = []
        removed_parts = 0

        for part in content:
            if isinstance(part, dict):
                part_type = str(
                    part.get("type") or ""
                ).strip().lower()
                if part_type in _MEDIA_PART_TYPES:
                    removed_parts += 1
                    continue

            copied_parts.append(copy.deepcopy(part))

        if removed_parts:
            copied_parts.append(
                {
                    "type": "text",
                    "text": placeholder,
                }
            )

        return copied_parts, removed_parts

    if isinstance(content, dict):
        part_type = str(
            content.get("type") or ""
        ).strip().lower()
        if part_type in _MEDIA_PART_TYPES:
            return (
                {
                    "type": "text",
                    "text": placeholder,
                },
                1,
            )

    return copy.deepcopy(content), 0


def _latest_user_message_index(
    messages: list,
) -> int | None:
    """Return the latest user-message index, or None when unavailable."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
        ):
            return index

    return None


def build_historical_media_compaction_preview(
    messages: list,
    *,
    placeholder: str = _HISTORICAL_MEDIA_PLACEHOLDER,
) -> tuple[list, dict]:
    """Remove media only from turns completed before the latest user turn.

    The latest user message and every message after it form the protected
    current turn. This keeps an attached image available throughout planner,
    tool-call, and tool-continuation passes. On the next user request, the
    previous turn becomes historical and its media payload can be replaced in
    the backend copy. The original OpenWebUI request is never mutated.
    """

    source_messages = messages if isinstance(messages, list) else []
    compacted_messages = copy.deepcopy(source_messages)
    before_report = build_context_window_report(
        source_messages,
        tail_messages=1,
    )
    current_turn_start = _latest_user_message_index(
        source_messages
    )
    system_prefix_messages = int(
        before_report.get("system_prefix_messages", 0)
        or 0
    )
    message_manifest = list(
        before_report.get("message_manifest", [])
        or []
    )

    candidate_indices = []
    if current_turn_start is not None:
        for index in range(
            system_prefix_messages,
            current_turn_start,
        ):
            if not (0 <= index < len(message_manifest)):
                continue
            manifest = message_manifest[index]
            if not isinstance(manifest, dict):
                continue
            if (
                manifest.get("media_data_uri_count")
                or manifest.get("media_reference_count")
            ):
                candidate_indices.append(index)

    compacted_indices = []
    removed_media_parts = 0

    for index in candidate_indices:
        if not (
            0 <= index < len(compacted_messages)
            and isinstance(compacted_messages[index], dict)
        ):
            continue

        message = compacted_messages[index]
        compacted_content, removed = (
            _replace_media_content_with_placeholder(
                message.get("content"),
                placeholder=str(placeholder),
            )
        )
        if not removed:
            continue

        message["content"] = compacted_content
        compacted_indices.append(index)
        removed_media_parts += removed

    before_json_chars = _json_char_count(source_messages)
    after_json_chars = _json_char_count(compacted_messages)
    after_report = build_context_window_report(
        compacted_messages,
        tail_messages=1,
    )

    protected_roles = []
    if current_turn_start is not None:
        protected_roles = [
            _message_role(message)
            for message in source_messages[current_turn_start:]
        ]

    return compacted_messages, {
        "policy": "before_latest_user_message",
        "candidate_indices": candidate_indices,
        "compacted_indices": compacted_indices,
        "compacted_messages": len(compacted_indices),
        "removed_media_parts": removed_media_parts,
        "before_json_chars": before_json_chars,
        "after_json_chars": after_json_chars,
        "saved_json_chars": max(
            0,
            before_json_chars - after_json_chars,
        ),
        "before_media_payload_chars": before_report.get(
            "media_payload_chars_total",
            0,
        ),
        "after_media_payload_chars": after_report.get(
            "media_payload_chars_total",
            0,
        ),
        "protected_current_turn_start": current_turn_start,
        "protected_current_turn_messages": (
            len(source_messages) - current_turn_start
            if current_turn_start is not None
            else len(source_messages)
        ),
        "protected_current_turn_roles": protected_roles,
        "active_tool_continuation": before_report.get(
            "active_tool_continuation",
            False,
        ),
    }


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
    content_manifests = [
        _content_manifest(
            message.get("content")
            if isinstance(message, dict)
            else message
        )
        for message in messages
    ]
    tool_calls_json_chars = [
        _json_char_count(message.get("tool_calls"))
        if (
            isinstance(message, dict)
            and message.get("tool_calls") is not None
        )
        else 0
        for message in messages
    ]
    message_manifest = [
        {
            "index": index,
            "role": _message_role(message),
            "message_json_chars": (
                message_json_chars[index]
            ),
            "tool_calls_json_chars": (
                tool_calls_json_chars[index]
            ),
            "has_tool_calls": bool(
                isinstance(message, dict)
                and message.get("tool_calls")
            ),
            "has_tool_call_id": bool(
                isinstance(message, dict)
                and message.get("tool_call_id")
            ),
            **content_manifests[index],
        }
        for index, message in enumerate(messages)
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

    older_media_candidate_indices = [
        index
        for index in range(older_start, older_end)
        if (
            content_manifests[index][
                "media_data_uri_count"
            ]
            or content_manifests[index][
                "media_reference_count"
            ]
        )
    ]

    completed_tool_groups = (
        _completed_tool_protocol_groups(messages)
    )
    older_tool_groups = [
        group
        for group in completed_tool_groups
        if (
            group["message_indices"]
            and min(group["message_indices"])
            >= older_start
            and max(group["message_indices"])
            < older_end
        )
    ]
    older_tool_protocol_indices = sorted(
        {
            index
            for group in older_tool_groups
            for index in group["message_indices"]
        }
    )

    return {
        "messages_total": message_count,
        "role_counts": dict(sorted(role_counts.items())),
        "content_chars_total": sum(content_chars),
        "message_json_chars_total": sum(
            message_json_chars
        ),
        "text_chars_total": sum(
            item["text_chars"]
            for item in content_manifests
        ),
        "media_payload_chars_total": sum(
            item["media_payload_chars"]
            for item in content_manifests
        ),
        "media_data_uri_chars_total": sum(
            item["media_data_uri_chars"]
            for item in content_manifests
        ),
        "media_reference_chars_total": sum(
            item["media_reference_chars"]
            for item in content_manifests
        ),
        "media_data_uri_count": sum(
            item["media_data_uri_count"]
            for item in content_manifests
        ),
        "media_reference_count": sum(
            item["media_reference_count"]
            for item in content_manifests
        ),
        "other_or_structure_chars_total": sum(
            item["other_or_structure_chars"]
            for item in content_manifests
        ),
        "tool_calls_json_chars_total": sum(
            tool_calls_json_chars
        ),
        "message_manifest": message_manifest,
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
        "older_media_candidate_messages": len(
            older_media_candidate_indices
        ),
        "older_media_candidate_indices": (
            older_media_candidate_indices
        ),
        "older_media_candidate_payload_chars": sum(
            content_manifests[index][
                "media_payload_chars"
            ]
            for index in older_media_candidate_indices
        ),
        "older_media_candidate_data_uri_count": sum(
            content_manifests[index][
                "media_data_uri_count"
            ]
            for index in older_media_candidate_indices
        ),
        "older_media_candidate_reference_count": sum(
            content_manifests[index][
                "media_reference_count"
            ]
            for index in older_media_candidate_indices
        ),
        "older_tool_protocol_groups": len(
            older_tool_groups
        ),
        "older_tool_protocol_messages": len(
            older_tool_protocol_indices
        ),
        "older_tool_protocol_indices": (
            older_tool_protocol_indices
        ),
        "older_tool_protocol_json_chars": sum(
            message_json_chars[index]
            for index in older_tool_protocol_indices
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
