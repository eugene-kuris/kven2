from __future__ import annotations

import copy
import hashlib
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

CONTEXT_BUDGET_REPORT_VERSION = (
    "kven2-context-budget-v1"
)


def estimate_tokens_from_chars(
    char_count: int,
    *,
    chars_per_token: float = 4.0,
) -> int:
    """
    Return a deterministic conservative token estimate.

    This estimator is intentionally tokenizer-independent.
    It is suitable for dry-run budgeting and telemetry, not
    exact backend token accounting.
    """
    import math

    try:
        safe_chars = max(0, int(char_count))
    except (TypeError, ValueError):
        safe_chars = 0

    try:
        safe_ratio = float(chars_per_token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "chars_per_token must be a positive number"
        ) from exc

    if safe_ratio <= 0:
        raise ValueError(
            "chars_per_token must be greater than zero"
        )

    if safe_chars == 0:
        return 0

    return int(
        math.ceil(safe_chars / safe_ratio)
    )


def build_context_budget_report(
    messages: list,
    *,
    tail_messages: int = 12,
    context_tokens: int = 49152,
    reserved_completion_tokens: int = 8192,
    summary_target_tokens: int = 2048,
    chars_per_token: float = 4.0,
) -> dict:
    """
    Build a deterministic content-free context budget report.

    The supplied messages are never changed. The report
    estimates whether the current prompt fits, how much old
    history is available for compaction, and whether a bounded
    summary could fit beside the protected prompt sections.
    """
    try:
        safe_context_tokens = max(
            1,
            int(context_tokens),
        )
    except (TypeError, ValueError):
        safe_context_tokens = 49152

    try:
        safe_completion_reserve = max(
            0,
            int(reserved_completion_tokens),
        )
    except (TypeError, ValueError):
        safe_completion_reserve = 8192

    safe_completion_reserve = min(
        safe_completion_reserve,
        max(0, safe_context_tokens - 1),
    )

    try:
        safe_summary_target = max(
            0,
            int(summary_target_tokens),
        )
    except (TypeError, ValueError):
        safe_summary_target = 2048

    base_report = build_context_window_report(
        messages,
        tail_messages=tail_messages,
    )

    system_tokens = estimate_tokens_from_chars(
        base_report.get(
            "system_prefix_json_chars",
            0,
        ),
        chars_per_token=chars_per_token,
    )
    older_tokens = estimate_tokens_from_chars(
        base_report.get(
            "older_candidate_json_chars",
            0,
        ),
        chars_per_token=chars_per_token,
    )
    tail_tokens = estimate_tokens_from_chars(
        base_report.get(
            "verbatim_tail_json_chars",
            0,
        ),
        chars_per_token=chars_per_token,
    )
    older_tool_tokens = estimate_tokens_from_chars(
        base_report.get(
            "older_tool_protocol_json_chars",
            0,
        ),
        chars_per_token=chars_per_token,
    )

    estimated_tokens_before = (
        system_tokens
        + older_tokens
        + tail_tokens
    )
    fixed_tokens = (
        system_tokens
        + tail_tokens
    )
    prompt_token_budget = max(
        1,
        safe_context_tokens
        - safe_completion_reserve,
    )
    available_summary_tokens = max(
        0,
        prompt_token_budget
        - fixed_tokens,
    )

    effective_summary_tokens = min(
        safe_summary_target,
        older_tokens,
        available_summary_tokens,
    )

    if older_tokens:
        estimated_tokens_after_summary = (
            fixed_tokens
            + effective_summary_tokens
        )
    else:
        estimated_tokens_after_summary = (
            fixed_tokens
        )

    required_reduction_tokens = max(
        0,
        estimated_tokens_before
        - prompt_token_budget,
    )
    predicted_reduction_tokens = max(
        0,
        estimated_tokens_before
        - estimated_tokens_after_summary,
    )

    result = dict(base_report)
    result.update(
        {
            "budget_report_version": (
                CONTEXT_BUDGET_REPORT_VERSION
            ),
            "chars_per_token": float(
                chars_per_token
            ),
            "context_token_budget": (
                safe_context_tokens
            ),
            "reserved_completion_tokens": (
                safe_completion_reserve
            ),
            "prompt_token_budget": (
                prompt_token_budget
            ),
            "configured_summary_target_tokens": (
                safe_summary_target
            ),
            "effective_summary_target_tokens": (
                effective_summary_tokens
            ),
            "estimated_system_prefix_tokens": (
                system_tokens
            ),
            "estimated_older_candidate_tokens": (
                older_tokens
            ),
            "estimated_older_tool_protocol_tokens": (
                older_tool_tokens
            ),
            "estimated_older_non_tool_tokens": max(
                0,
                older_tokens
                - older_tool_tokens,
            ),
            "estimated_verbatim_tail_tokens": (
                tail_tokens
            ),
            "estimated_fixed_prompt_tokens": (
                fixed_tokens
            ),
            "estimated_prompt_tokens_before": (
                estimated_tokens_before
            ),
            "estimated_prompt_tokens_after_summary": (
                estimated_tokens_after_summary
            ),
            "required_reduction_tokens": (
                required_reduction_tokens
            ),
            "predicted_reduction_tokens": (
                predicted_reduction_tokens
            ),
            "over_budget_before": (
                estimated_tokens_before
                > prompt_token_budget
            ),
            "fits_after_summary": (
                estimated_tokens_after_summary
                <= prompt_token_budget
            ),
            "compaction_candidate_available": (
                older_tokens > 0
            ),
        }
    )

    return result


TEXT_SUMMARY_CHECKPOINT_VERSION = (
    "kven2-text-summary-checkpoint-v1"
)

TEXT_SUMMARY_CHECKPOINT_HASH_SCOPE = (
    "conversation-prefix-after-leading-system-v1"
)


_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def _canonical_json_sha256(value: Any) -> str:
    """Return SHA-256 for deterministic compact UTF-8 JSON."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    """Return whether value is a lowercase SHA-256 hex digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX_DIGITS
    )


def build_text_summary_checkpoint(
    messages: list,
    *,
    summary_text: str,
    summarized_prefix_end: int,
) -> dict:
    """Build a deterministic checkpoint for one exact chat prefix.

    ``summarized_prefix_end`` is an absolute, exclusive message index in the
    supplied request. Leading system messages are excluded from the identity
    because OpenWebUI can regenerate that runtime prefix between otherwise
    identical requests.
    """

    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    if (
        not isinstance(summarized_prefix_end, int)
        or isinstance(summarized_prefix_end, bool)
    ):
        raise ValueError(
            "summarized_prefix_end must be an integer"
        )

    system_prefix_messages = (
        _leading_system_message_count(messages)
    )

    if summarized_prefix_end <= system_prefix_messages:
        raise ValueError(
            "summarized prefix must contain at least one "
            "non-system message"
        )

    if summarized_prefix_end > len(messages):
        raise ValueError(
            "summarized_prefix_end exceeds message count"
        )

    if not isinstance(summary_text, str):
        raise ValueError("summary_text must be a string")

    normalized_summary = summary_text.strip()

    if not normalized_summary:
        raise ValueError("summary_text must not be empty")

    summarized_messages = messages[
        system_prefix_messages:summarized_prefix_end
    ]
    summarized_message_count = len(
        summarized_messages
    )
    prefix_sha256 = _canonical_json_sha256(
        summarized_messages
    )
    summary_sha256 = hashlib.sha256(
        normalized_summary.encode("utf-8")
    ).hexdigest()

    checkpoint_identity = {
        "checkpoint_version": (
            TEXT_SUMMARY_CHECKPOINT_VERSION
        ),
        "hash_scope": (
            TEXT_SUMMARY_CHECKPOINT_HASH_SCOPE
        ),
        "summarized_message_count": (
            summarized_message_count
        ),
        "prefix_sha256": prefix_sha256,
        "summary_sha256": summary_sha256,
    }

    return {
        **checkpoint_identity,
        "checkpoint_id": _canonical_json_sha256(
            checkpoint_identity
        ),
        "summary_text": normalized_summary,
        "summary_chars": len(normalized_summary),
    }


def find_matching_text_summary_checkpoint(
    messages: list,
    checkpoints: list,
) -> tuple[dict | None, dict]:
    """Return the longest valid checkpoint matching the current prefix.

    Invalid, corrupt, edited, branched-before-boundary, or overlong
    checkpoints are ignored. A branch after the summarized boundary remains a
    valid continuation of that checkpoint.
    """

    source_messages = (
        messages
        if isinstance(messages, list)
        else []
    )
    source_checkpoints = (
        checkpoints
        if isinstance(checkpoints, list)
        else []
    )
    system_prefix_messages = (
        _leading_system_message_count(source_messages)
    )
    conversation_messages = source_messages[
        system_prefix_messages:
    ]

    report = {
        "checkpoint_version": (
            TEXT_SUMMARY_CHECKPOINT_VERSION
        ),
        "hash_scope": (
            TEXT_SUMMARY_CHECKPOINT_HASH_SCOPE
        ),
        "candidate_checkpoints": len(
            source_checkpoints
        ),
        "valid_checkpoints": 0,
        "invalid_checkpoints": 0,
        "insufficient_history_checkpoints": 0,
        "prefix_hash_mismatches": 0,
        "prefix_hash_errors": 0,
        "matching_checkpoints": 0,
        "system_prefix_messages": (
            system_prefix_messages
        ),
        "selected": False,
        "selected_checkpoint_index": None,
        "selected_summarized_message_count": 0,
        "selected_summarized_prefix_end": None,
        "selected_summary_chars": 0,
    }

    selected_checkpoint = None
    selected_index = None
    selected_message_count = -1

    for index, checkpoint in enumerate(
        source_checkpoints
    ):
        if not isinstance(checkpoint, dict):
            report["invalid_checkpoints"] += 1
            continue

        checkpoint_version = checkpoint.get(
            "checkpoint_version"
        )
        hash_scope = checkpoint.get("hash_scope")
        summarized_message_count = checkpoint.get(
            "summarized_message_count"
        )
        prefix_sha256 = checkpoint.get(
            "prefix_sha256"
        )
        summary_text = checkpoint.get(
            "summary_text"
        )
        summary_chars = checkpoint.get(
            "summary_chars"
        )
        summary_sha256 = checkpoint.get(
            "summary_sha256"
        )
        checkpoint_id = checkpoint.get(
            "checkpoint_id"
        )

        structurally_valid = (
            checkpoint_version
            == TEXT_SUMMARY_CHECKPOINT_VERSION
            and hash_scope
            == TEXT_SUMMARY_CHECKPOINT_HASH_SCOPE
            and isinstance(
                summarized_message_count,
                int,
            )
            and not isinstance(
                summarized_message_count,
                bool,
            )
            and summarized_message_count > 0
            and _is_sha256_hex(prefix_sha256)
            and isinstance(summary_text, str)
            and bool(summary_text)
            and summary_text == summary_text.strip()
            and isinstance(summary_chars, int)
            and not isinstance(summary_chars, bool)
            and summary_chars == len(summary_text)
            and _is_sha256_hex(summary_sha256)
            and summary_sha256
            == hashlib.sha256(
                summary_text.encode("utf-8")
            ).hexdigest()
            and _is_sha256_hex(checkpoint_id)
        )

        if structurally_valid:
            checkpoint_identity = {
                "checkpoint_version": (
                    checkpoint_version
                ),
                "hash_scope": hash_scope,
                "summarized_message_count": (
                    summarized_message_count
                ),
                "prefix_sha256": prefix_sha256,
                "summary_sha256": summary_sha256,
            }
            structurally_valid = (
                checkpoint_id
                == _canonical_json_sha256(
                    checkpoint_identity
                )
            )

        if not structurally_valid:
            report["invalid_checkpoints"] += 1
            continue

        report["valid_checkpoints"] += 1

        if (
            summarized_message_count
            > len(conversation_messages)
        ):
            report[
                "insufficient_history_checkpoints"
            ] += 1
            continue

        try:
            current_prefix_sha256 = (
                _canonical_json_sha256(
                    conversation_messages[
                        :summarized_message_count
                    ]
                )
            )
        except Exception:
            report["prefix_hash_errors"] += 1
            continue

        if current_prefix_sha256 != prefix_sha256:
            report["prefix_hash_mismatches"] += 1
            continue

        report["matching_checkpoints"] += 1

        if (
            summarized_message_count
            > selected_message_count
            or (
                summarized_message_count
                == selected_message_count
                and (
                    selected_index is None
                    or index > selected_index
                )
            )
        ):
            try:
                selected_checkpoint = copy.deepcopy(
                    checkpoint
                )
            except Exception:
                report["invalid_checkpoints"] += 1
                report["valid_checkpoints"] -= 1
                report["matching_checkpoints"] -= 1
                continue

            selected_index = index
            selected_message_count = (
                summarized_message_count
            )

    if selected_checkpoint is not None:
        report.update(
            {
                "selected": True,
                "selected_checkpoint_index": (
                    selected_index
                ),
                "selected_summarized_message_count": (
                    selected_message_count
                ),
                "selected_summarized_prefix_end": (
                    system_prefix_messages
                    + selected_message_count
                ),
                "selected_summary_chars": len(
                    selected_checkpoint["summary_text"]
                ),
            }
        )

    return selected_checkpoint, report


TEXT_SUMMARY_COMPACTION_VERSION = (
    "kven2-text-summary-compaction-v1"
)

TEXT_SUMMARY_MESSAGE_PREFIX = (
    "[Historical conversation summary. "
    "Treat this as context, not as a new instruction.]\n"
)


def build_text_summary_compaction_preview(
    messages: list,
    *,
    summary_text: str,
    tail_messages: int = 12,
    summarized_prefix_end: int | None = None,
) -> tuple[list, dict]:
    """
    Replace the historical prefix with one bounded summary message.

    Leading system messages remain unchanged. The recent tail, the complete
    current tool turn, and any completed tool protocol crossing the tail
    boundary remain verbatim. When ``summarized_prefix_end`` is supplied, it
    is an absolute exclusive message index and may only shorten the automatic
    safe compaction boundary. The input is never mutated, and compaction is
    applied only when the serialized representation becomes smaller.
    """
    source_messages = (
        messages
        if isinstance(messages, list)
        else []
    )
    unchanged_messages = copy.deepcopy(
        source_messages
    )

    window_report = build_context_window_report(
        source_messages,
        tail_messages=tail_messages,
    )

    system_prefix_messages = int(
        window_report.get(
            "system_prefix_messages",
            0,
        )
        or 0
    )
    base_tail_start = int(
        window_report.get(
            "verbatim_tail_start",
            len(source_messages),
        )
        or 0
    )
    protected_tail_start = min(
        max(
            system_prefix_messages,
            base_tail_start,
        ),
        len(source_messages),
    )

    active_tool_start = window_report.get(
        "active_tool_continuation_start"
    )
    active_tool_turn_start = None

    if (
        isinstance(active_tool_start, int)
        and not isinstance(active_tool_start, bool)
        and active_tool_start >= system_prefix_messages
    ):
        current_user_start = (
            _latest_user_message_index(
                source_messages[
                    :active_tool_start + 1
                ]
            )
        )

        if (
            current_user_start is not None
            and current_user_start
            >= system_prefix_messages
        ):
            active_tool_turn_start = (
                current_user_start
            )
        else:
            active_tool_turn_start = (
                active_tool_start
            )

        protected_tail_start = min(
            protected_tail_start,
            active_tool_turn_start,
        )

    crossing_tool_groups = []

    changed = True

    while changed:
        changed = False

        for group in _completed_tool_protocol_groups(
            source_messages
        ):
            indices = list(
                group.get("message_indices") or []
            )

            if not indices:
                continue

            group_start = min(indices)
            group_end = max(indices)

            if (
                group_start < protected_tail_start
                <= group_end
            ):
                crossing_tool_groups.append(
                    {
                        "start": group_start,
                        "end": group_end,
                    }
                )
                protected_tail_start = max(
                    system_prefix_messages,
                    group_start,
                )
                changed = True
                break

    safe_summary_boundary_end = (
        protected_tail_start
    )
    requested_summary_boundary_end = (
        summarized_prefix_end
    )
    selected_summary_boundary_end = (
        safe_summary_boundary_end
    )
    summary_boundary_reason = "automatic_safe_boundary"

    if summarized_prefix_end is not None:
        if (
            isinstance(summarized_prefix_end, bool)
            or not isinstance(
                summarized_prefix_end,
                int,
            )
        ):
            selected_summary_boundary_end = None
            summary_boundary_reason = (
                "invalid_summary_boundary"
            )
        elif (
            summarized_prefix_end
            <= system_prefix_messages
        ):
            selected_summary_boundary_end = None
            summary_boundary_reason = (
                "summary_boundary_before_history"
            )
        elif summarized_prefix_end > len(
            source_messages
        ):
            selected_summary_boundary_end = None
            summary_boundary_reason = (
                "summary_boundary_out_of_range"
            )
        elif (
            summarized_prefix_end
            > safe_summary_boundary_end
        ):
            selected_summary_boundary_end = None
            summary_boundary_reason = (
                "summary_boundary_crosses_protected_tail"
            )
        else:
            selected_summary_boundary_end = (
                summarized_prefix_end
            )
            summary_boundary_reason = (
                "explicit_safe_boundary"
            )

    older_candidate_messages = max(
        0,
        (
            selected_summary_boundary_end
            if isinstance(
                selected_summary_boundary_end,
                int,
            )
            else system_prefix_messages
        )
        - system_prefix_messages,
    )

    try:
        normalized_summary = str(
            summary_text
            if summary_text is not None
            else ""
        ).strip()
    except Exception:
        normalized_summary = ""

    before_json_chars = _json_char_count(
        source_messages
    )

    report = {
        "compaction_version": (
            TEXT_SUMMARY_COMPACTION_VERSION
        ),
        "configured_tail_messages": int(
            window_report.get(
                "configured_tail_messages",
                max(1, int(tail_messages)),
            )
        ),
        "system_prefix_messages": (
            system_prefix_messages
        ),
        "base_tail_start": base_tail_start,
        "protected_tail_start": (
            protected_tail_start
        ),
        "safe_summary_boundary_end": (
            safe_summary_boundary_end
        ),
        "requested_summarized_prefix_end": (
            requested_summary_boundary_end
        ),
        "selected_summarized_prefix_end": (
            selected_summary_boundary_end
        ),
        "summary_boundary_reason": (
            summary_boundary_reason
        ),
        "older_candidate_start": (
            system_prefix_messages
        ),
        "older_candidate_end": (
            selected_summary_boundary_end
        ),
        "older_candidate_messages": (
            older_candidate_messages
        ),
        "verbatim_tail_messages": max(
            0,
            len(source_messages)
            - (
                selected_summary_boundary_end
                if isinstance(
                    selected_summary_boundary_end,
                    int,
                )
                else len(source_messages)
            ),
        ),
        "active_tool_continuation": bool(
            window_report.get(
                "active_tool_continuation"
            )
        ),
        "active_tool_continuation_start": (
            active_tool_start
        ),
        "active_tool_turn_start": (
            active_tool_turn_start
        ),
        "crossing_tool_groups": (
            crossing_tool_groups
        ),
        "summary_input_chars": len(
            normalized_summary
        ),
        "summary_message_json_chars": 0,
        "before_messages": len(
            source_messages
        ),
        "after_messages": len(
            source_messages
        ),
        "removed_messages": 0,
        "before_json_chars": before_json_chars,
        "after_json_chars": before_json_chars,
        "saved_json_chars": 0,
        "compaction_applied": False,
        "reason": "not_evaluated",
    }

    if selected_summary_boundary_end is None:
        report["reason"] = summary_boundary_reason
        return unchanged_messages, report

    if older_candidate_messages <= 0:
        report["reason"] = "no_older_candidate"
        return unchanged_messages, report

    if not normalized_summary:
        report["reason"] = "empty_summary"
        return unchanged_messages, report

    summary_message = {
        "role": "assistant",
        "content": (
            TEXT_SUMMARY_MESSAGE_PREFIX
            + normalized_summary
        ),
    }

    report["summary_message_json_chars"] = (
        _json_char_count(summary_message)
    )

    proposed_messages = [
        *copy.deepcopy(
            source_messages[
                :system_prefix_messages
            ]
        ),
        summary_message,
        *copy.deepcopy(
            source_messages[
                selected_summary_boundary_end:
            ]
        ),
    ]

    proposed_json_chars = _json_char_count(
        proposed_messages
    )

    if proposed_json_chars >= before_json_chars:
        report["reason"] = "not_smaller"
        return unchanged_messages, report

    report.update(
        {
            "after_messages": len(
                proposed_messages
            ),
            "removed_messages": max(
                0,
                older_candidate_messages - 1,
            ),
            "after_json_chars": (
                proposed_json_chars
            ),
            "saved_json_chars": (
                before_json_chars
                - proposed_json_chars
            ),
            "compaction_applied": True,
            "reason": "applied",
        }
    )

    return proposed_messages, report


HISTORICAL_TOOL_PROTOCOL_COMPACTION_VERSION = (
    "kven2-historical-tool-protocol-compaction-v1"
)

HISTORICAL_TOOL_RESULT_PLACEHOLDER = (
    "[Historical tool result omitted from active context.]"
)


def build_historical_tool_protocol_compaction_preview(
    messages: list,
    *,
    tail_messages: int = 12,
) -> tuple[list, dict]:
    """
    Compact completed historical tool exchanges without mutating input.

    Only tool protocols already classified as historical by the
    context-window report are eligible. Active tool continuation and
    the protected verbatim tail are never rediscovered or modified.

    Tool call IDs and function names are preserved. Historical
    arguments are replaced with an empty JSON object and historical
    tool results with a bounded placeholder. A group is changed only
    when the serialized representation becomes smaller.
    """
    import copy
    import json

    source_messages = (
        messages
        if isinstance(messages, list)
        else []
    )
    compacted_messages = copy.deepcopy(
        source_messages
    )

    report = build_context_window_report(
        source_messages,
        tail_messages=tail_messages,
    )

    def json_chars(value) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    raw_indices = report.get(
        "older_tool_protocol_indices",
        [],
    )

    candidate_indices = sorted(
        {
            int(index)
            for index in raw_indices
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(source_messages)
            )
        }
    )

    validated_groups = []
    invalid_candidate_indices = []
    position = 0

    while position < len(candidate_indices):
        start_index = candidate_indices[position]
        start_message = source_messages[start_index]

        if not isinstance(start_message, dict):
            invalid_candidate_indices.append(
                start_index
            )
            position += 1
            continue

        if start_message.get("role") != "assistant":
            invalid_candidate_indices.append(
                start_index
            )
            position += 1
            continue

        tool_calls = start_message.get("tool_calls")

        if not isinstance(tool_calls, list) or not tool_calls:
            invalid_candidate_indices.append(
                start_index
            )
            position += 1
            continue

        call_ids = []
        calls_valid = True

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                calls_valid = False
                break

            call_id = str(
                tool_call.get("id") or ""
            ).strip()
            function = tool_call.get("function")

            if (
                not call_id
                or not isinstance(function, dict)
                or not str(
                    function.get("name") or ""
                ).strip()
                or call_id in call_ids
            ):
                calls_valid = False
                break

            call_ids.append(call_id)

        if not calls_valid:
            invalid_candidate_indices.append(
                start_index
            )
            position += 1
            continue

        group_indices = [start_index]
        result_ids = []
        cursor = position + 1

        while cursor < len(candidate_indices):
            candidate_index = candidate_indices[
                cursor
            ]

            if candidate_index != (
                group_indices[-1] + 1
            ):
                break

            candidate_message = source_messages[
                candidate_index
            ]

            if (
                not isinstance(candidate_message, dict)
                or candidate_message.get("role")
                != "tool"
            ):
                break

            tool_call_id = str(
                candidate_message.get(
                    "tool_call_id"
                )
                or ""
            ).strip()

            # Collect every contiguous historical tool result first.
            # Validate the complete group only after collection so an
            # extra, duplicate, or unknown result cannot be orphaned.
            group_indices.append(
                candidate_index
            )
            result_ids.append(tool_call_id)
            cursor += 1

        if (
            len(group_indices)
            != 1 + len(call_ids)
            or len(result_ids)
            != len(set(result_ids))
            or set(result_ids) != set(call_ids)
        ):
            invalid_candidate_indices.extend(
                group_indices
            )
            position = max(
                position + 1,
                cursor,
            )
            continue

        validated_groups.append(
            {
                "indices": group_indices,
                "call_ids": call_ids,
            }
        )
        position = cursor

    compacted_indices = []
    compacted_groups = 0
    skipped_non_shrinking_groups = 0

    for group in validated_groups:
        indices = group["indices"]
        assistant_index = indices[0]

        original_group = [
            source_messages[index]
            for index in indices
        ]
        replacement_group = copy.deepcopy(
            original_group
        )

        assistant_message = replacement_group[0]
        sanitized_calls = []

        for tool_call in assistant_message[
            "tool_calls"
        ]:
            sanitized_call = copy.deepcopy(
                tool_call
            )
            sanitized_function = copy.deepcopy(
                sanitized_call["function"]
            )
            sanitized_function["arguments"] = "{}"
            sanitized_call["function"] = (
                sanitized_function
            )
            sanitized_calls.append(
                sanitized_call
            )

        assistant_message["tool_calls"] = (
            sanitized_calls
        )

        for tool_message in replacement_group[1:]:
            tool_message["content"] = (
                HISTORICAL_TOOL_RESULT_PLACEHOLDER
            )

        before_group_chars = json_chars(
            original_group
        )
        after_group_chars = json_chars(
            replacement_group
        )

        if after_group_chars >= before_group_chars:
            skipped_non_shrinking_groups += 1
            continue

        for index, replacement in zip(
            indices,
            replacement_group,
        ):
            compacted_messages[index] = replacement
            compacted_indices.append(index)

        compacted_groups += 1

    before_json_chars = json_chars(
        source_messages
    )
    after_json_chars = json_chars(
        compacted_messages
    )

    meta = {
        "compaction_version": (
            HISTORICAL_TOOL_PROTOCOL_COMPACTION_VERSION
        ),
        "policy": (
            "sanitize_historical_tool_arguments_and_results"
        ),
        "configured_tail_messages": report.get(
            "configured_tail_messages"
        ),
        "active_tool_continuation": report.get(
            "active_tool_continuation",
            False,
        ),
        "active_tool_continuation_start": report.get(
            "active_tool_continuation_start"
        ),
        "verbatim_tail_start": report.get(
            "verbatim_tail_start"
        ),
        "candidate_groups": report.get(
            "older_tool_protocol_groups",
            0,
        ),
        "candidate_messages": len(
            candidate_indices
        ),
        "candidate_indices": candidate_indices,
        "validated_candidate_groups": len(
            validated_groups
        ),
        "invalid_candidate_indices": sorted(
            set(invalid_candidate_indices)
        ),
        "compacted_groups": compacted_groups,
        "compacted_messages": len(
            compacted_indices
        ),
        "compacted_indices": sorted(
            compacted_indices
        ),
        "skipped_non_shrinking_groups": (
            skipped_non_shrinking_groups
        ),
        "before_json_chars": before_json_chars,
        "after_json_chars": after_json_chars,
        "saved_json_chars": max(
            0,
            before_json_chars - after_json_chars,
        ),
    }

    return compacted_messages, meta
