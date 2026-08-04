from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


_DIRECTIVE_RE = re.compile(
    r"^\s*#(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?=\s|$)(?P<tail>.*)$",
    re.DOTALL,
)


class ToolDirectiveError(ValueError):
    """Raised when an explicit tool directive is invalid."""


@dataclass(frozen=True)
class ToolDirective:
    """Parsed explicit tool or help directive."""

    kind: str
    tool_name: str | None
    remaining_text: str


def latest_user_message_text(
    messages: Iterable[dict[str, Any]],
) -> str:
    """Return text from the latest user message."""

    message_list = list(messages)

    for message in reversed(message_list):
        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []

            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    parts.append(item["text"])

            return "\n".join(parts)

        return ""

    return ""


def replace_latest_user_message_text(
    messages: Iterable[dict[str, Any]],
    new_text: str,
) -> list[dict[str, Any]]:
    """Copy messages and replace text in the latest user message."""

    if not isinstance(new_text, str):
        raise TypeError("replacement user text must be a string")

    copied = [
        dict(message)
        if isinstance(message, dict)
        else message
        for message in messages
    ]

    for index in range(len(copied) - 1, -1, -1):
        message = copied[index]

        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            message["content"] = new_text
            return copied

        if isinstance(content, list):
            rewritten = []
            text_written = False

            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    if text_written:
                        continue

                    copied_item = dict(item)
                    copied_item["text"] = new_text
                    rewritten.append(copied_item)
                    text_written = True
                    continue

                rewritten.append(
                    dict(item)
                    if isinstance(item, dict)
                    else item
                )

            if not text_written:
                rewritten.insert(
                    0,
                    {
                        "type": "text",
                        "text": new_text,
                    },
                )

            message["content"] = rewritten
            return copied

        message["content"] = new_text
        return copied

    raise ToolDirectiveError(
        "no user message is available for directive removal"
    )


def _normalize_tools(
    tools: Iterable[dict[str, Any]],
    allowed_names: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}

    for item in tools:
        if not isinstance(item, dict):
            continue

        function = item.get("function")
        if not isinstance(function, dict):
            continue

        raw_name = function.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue

        name = raw_name.strip()

        if allowed_names is not None and name not in allowed_names:
            continue

        normalized[name] = function

    return normalized


def _canonical_name(
    requested_name: str,
    tools_by_name: dict[str, dict[str, Any]],
) -> str | None:
    lowered = requested_name.strip().lower()

    for name in tools_by_name:
        if name.lower() == lowered:
            return name

    return None


def _available_directives(
    tools_by_name: dict[str, dict[str, Any]],
) -> str:
    names = ", ".join(
        f"#{name}"
        for name in sorted(tools_by_name)
    )
    return names or "(none)"


def parse_tool_directive(
    text: str,
    tools: Iterable[dict[str, Any]],
    *,
    allowed_names: set[str] | None = None,
) -> ToolDirective | None:
    """
    Parse one leading #tool or #tools directive.

    Only the first non-empty token is treated as a directive. A second
    consecutive leading hash token is rejected explicitly.
    """

    if not isinstance(text, str):
        raise TypeError("tool directive text must be a string")

    match = _DIRECTIVE_RE.match(text)

    if match is None:
        return None

    tools_by_name = _normalize_tools(
        tools,
        allowed_names,
    )
    requested = match.group("name")
    tail = match.group("tail").strip()

    if requested.lower() == "tools":
        if not tail:
            return ToolDirective(
                kind="help",
                tool_name=None,
                remaining_text="",
            )

        parts = tail.split()

        if len(parts) != 1:
            raise ToolDirectiveError(
                "#tools accepts zero or one tool name"
            )

        detail_name = parts[0]

        if detail_name.startswith("#"):
            detail_name = detail_name[1:]

        canonical = _canonical_name(
            detail_name,
            tools_by_name,
        )

        if canonical is None:
            raise ToolDirectiveError(
                "unknown or unavailable tool for #tools: "
                f"{detail_name}; available: "
                f"{_available_directives(tools_by_name)}"
            )

        return ToolDirective(
            kind="help",
            tool_name=canonical,
            remaining_text="",
        )

    if tail.startswith("#"):
        second = _DIRECTIVE_RE.match(tail)

        if second is not None:
            raise ToolDirectiveError(
                "multiple tool directives are not allowed"
            )

    canonical = _canonical_name(
        requested,
        tools_by_name,
    )

    if canonical is None:
        raise ToolDirectiveError(
            "unknown or unavailable tool directive: "
            f"#{requested}; available: "
            f"{_available_directives(tools_by_name)}; "
            "help: #tools"
        )

    return ToolDirective(
        kind="tool",
        tool_name=canonical,
        remaining_text=tail,
    )


def _required_argument_names(
    function: dict[str, Any],
) -> list[str]:
    parameters = function.get("parameters")

    if not isinstance(parameters, dict):
        return []

    required = parameters.get("required")

    if not isinstance(required, list):
        return []

    return [
        item
        for item in required
        if isinstance(item, str) and item.strip()
    ]


def _parameter_lines(
    function: dict[str, Any],
) -> list[str]:
    parameters = function.get("parameters")

    if not isinstance(parameters, dict):
        return []

    properties = parameters.get("properties")

    if not isinstance(properties, dict):
        return []

    required = set(_required_argument_names(function))
    lines: list[str] = []

    for name, schema in properties.items():
        if not isinstance(name, str):
            continue

        description = ""
        json_type = ""

        if isinstance(schema, dict):
            description = str(
                schema.get("description") or ""
            ).strip()
            json_type = str(
                schema.get("type") or ""
            ).strip()

        suffix_parts = []

        if json_type:
            suffix_parts.append(json_type)

        suffix_parts.append(
            "обязательный"
            if name in required
            else "необязательный"
        )

        suffix = ", ".join(suffix_parts)
        line = f"- {name} ({suffix})"

        if description:
            line += f": {description}"

        lines.append(line)

    return lines


def render_tools_help(
    tools: Iterable[dict[str, Any]],
    *,
    allowed_names: set[str] | None = None,
    tool_name: str | None = None,
) -> str:
    """Render Russian user-visible help from the exposed tool schemas."""

    tools_by_name = _normalize_tools(
        tools,
        allowed_names,
    )

    if tool_name is not None:
        canonical = _canonical_name(
            tool_name,
            tools_by_name,
        )

        if canonical is None:
            raise ToolDirectiveError(
                "unknown or unavailable tool for help: "
                f"{tool_name}; available: "
                f"{_available_directives(tools_by_name)}"
            )

        selected_names = [canonical]
        heading = "Инструмент:"
    else:
        selected_names = sorted(tools_by_name)
        heading = "Доступные инструменты:"

    lines = [heading]

    if not selected_names:
        lines.append("")
        lines.append("Нет доступных инструментов.")
        return "\n".join(lines)

    for name in selected_names:
        function = tools_by_name[name]
        description = str(
            function.get("description") or ""
        ).strip()
        required = _required_argument_names(function)

        lines.extend(
            [
                "",
                f"#{name} <запрос>",
            ]
        )

        if description:
            lines.append(description)

        lines.append(
            "Обязательные аргументы: "
            + (
                ", ".join(required)
                if required
                else "нет"
            )
            + "."
        )

        parameter_lines = _parameter_lines(function)

        if parameter_lines:
            lines.append("Параметры:")
            lines.extend(parameter_lines)

    if tool_name is None:
        lines.extend(
            [
                "",
                "Подробная справка: #tools <имя>",
                "Директива должна быть первым непустым токеном запроса.",
            ]
        )

    return "\n".join(lines)
