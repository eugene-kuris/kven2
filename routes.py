#routes.py
import json
import os
import re
import httpx
import logging
import asyncio
import uuid
from contextvars import ContextVar
from fastapi import Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse, Response

# Исправлены импорты на прямые названия файлов
from sqlite import (
    load_active_state,
    save_active_state,
    save_history_snapshot,
    get_semantic_context,
    get_project_context
)
from retrieval import retrieve_context  # <-- ФАЗА 4: Векторный ретрив
# ИСПРАВЛЕНО: Импорт обновлен на новое имя файла kven2_profile
from kven2_profile import load_agent_profile
from write_path import process_episodic, strip_reasoning
from config import settings
from model_adapters import resolve_model_adapter
from planner_router import route_tool_request as planner_route_tool_request
from speculative_stream import SpeculativeStream
from sandbox_client import execute_gateway_tool as sandbox_execute_gateway_tool
from tool_loop import (
    extract_gateway_tool_call as tool_loop_extract_gateway_tool_call,
    extract_decision_text_from_response_json as tool_loop_extract_decision_text_from_response_json,
    redact_decision_response_for_log as tool_loop_redact_decision_response_for_log,
    format_tool_result_message as tool_loop_format_tool_result_message,
    format_empty_final_tool_fallback as tool_loop_format_empty_final_tool_fallback,
    forward_tool_decision_and_extract_text as tool_loop_forward_tool_decision_and_extract_text,
    _log_incoming_tool_request as tool_loop_log_incoming_tool_request,
    _format_gateway_tool_availability_context as tool_loop_format_gateway_tool_availability_context,
    _exposed_enabled_tool_names as tool_loop_exposed_enabled_tool_names,
    _tool_loop_enabled_for_registered_tools as tool_loop_enabled_for_registered_tools,
    _explicit_tool_choice_name as tool_loop_explicit_tool_choice_name,
    _format_hidden_tool_decision_prompt as tool_loop_format_hidden_tool_decision_prompt,
)
logger = logging.getLogger(__name__)

_REQUEST_ID: ContextVar[str] = ContextVar(
    "kven2_request_id",
    default="",
)


class _RequestIdLogFilter(logging.Filter):
    """Prefix routes.py logs with the current ASGI request id."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = _REQUEST_ID.get()

        if request_id:
            message = str(record.msg)

            if not message.startswith("[rid="):
                record.msg = f"[rid={request_id}] {message}"

        return True


logger.addFilter(_RequestIdLogFilter())

router = APIRouter()
BASE_BACKEND = settings.LLM_BACKEND_URL
ROUTES_TOOLS_PROBE_VERSION = "planner-tool-routing-2026-07-30-v12"


# -----------------------------------------------------------------------------
# Safe generation parameter pass-through
# -----------------------------------------------------------------------------
# The gateway owns Kven context/memory injection, but generation controls from
# OWUI/curl should still reach llama.cpp. Keep this list conservative: do not
# pass response_format/metadata here. Native OpenAI tools are handled by
# a separate transparent pass-through branch below.

PASSTHROUGH_PARAMS = (
    "max_tokens",
    "top_p",
    "min_p",
    "top_k",
    "stop",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
)

# -----------------------------------------------------------------------------
# Final-answer safeguards: disable visible thinking and stop repetition loops
# -----------------------------------------------------------------------------

FINAL_ANSWER_CONTROL_MARKER = "KVEN FINAL ANSWER CONTROL"
FINAL_ANSWER_CONTROL = f"""
{FINAL_ANSWER_CONTROL_MARKER}:
- Return only the completed answer to the user.
- Do not output internal analysis, plans, checklists, self-evaluation, hidden reasoning, or draft notes.
- Do not repeat a conclusion or restart the answer after it is complete.
- For ordinary operational questions, answer directly and stop immediately after the useful result.
""".strip()


def _env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(str(os.getenv(name, "")).strip())
    except Exception:
        value = default
    return max(min_value, min(value, max_value))


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment value with an explicit default."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)

    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _inject_final_answer_control(messages: list) -> list:
    """Append one compact anti-reasoning policy to the leading system message."""
    copied = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]

    if copied and isinstance(copied[0], dict) and copied[0].get("role") == "system":
        existing = _content_to_text(copied[0].get("content")).strip()
        if FINAL_ANSWER_CONTROL_MARKER not in existing:
            copied[0]["content"] = (existing + "\n\n--- FINAL ANSWER CONTROL ---\n\n" + FINAL_ANSWER_CONTROL).strip()
    else:
        copied.insert(0, {"role": "system", "content": FINAL_ANSWER_CONTROL})

    return copied


def _apply_final_answer_safeguards(
    payload: dict,
    *,
    route_label: str,
    enable_thinking: bool = False,
) -> dict:
    """Apply the final-answer budget, sampling, and explicit thinking policy."""
    safe = dict(payload or {})

    min_tokens = _env_int(
        "KVEN2_FINAL_MIN_TOKENS",
        4096,
        256,
        131072,
    )

    requested_value = safe.get("max_tokens")
    if requested_value is None:
        requested_value = safe.get("max_completion_tokens")

    try:
        requested_max_tokens = (
            int(requested_value)
            if requested_value is not None
            else min_tokens
        )
    except Exception:
        requested_max_tokens = min_tokens

    # Enforce only a lower bound. Larger client-supplied budgets are preserved.
    safe["max_tokens"] = max(min_tokens, requested_max_tokens)
    safe.pop("max_completion_tokens", None)

    # Qwen thinking/greedy generation is the main source of the observed
    # self-review loop. These values are intentionally gateway-owned for final
    # answers, while tool-decision passes retain their separate bounded policy.
    safe["temperature"] = _env_float("KVEN2_FINAL_TEMPERATURE", 0.7, 0.05, 2.0)
    safe["top_p"] = _env_float("KVEN2_FINAL_TOP_P", 0.8, 0.05, 1.0)
    safe["top_k"] = _env_int("KVEN2_FINAL_TOP_K", 20, 1, 200)
    safe["min_p"] = _env_float("KVEN2_FINAL_MIN_P", 0.0, 0.0, 1.0)
    safe["presence_penalty"] = _env_float("KVEN2_FINAL_PRESENCE_PENALTY", 0.5, -2.0, 2.0)
    safe["repeat_penalty"] = _env_float("KVEN2_FINAL_REPEAT_PENALTY", 1.08, 0.8, 2.0)
    safe["repeat_last_n"] = _env_int("KVEN2_FINAL_REPEAT_LAST_N", 512, 64, 4096)

    thinking_enabled = bool(enable_thinking)

    chat_template_kwargs = safe.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    else:
        chat_template_kwargs = dict(chat_template_kwargs)

    chat_template_kwargs["enable_thinking"] = thinking_enabled
    safe["chat_template_kwargs"] = chat_template_kwargs
    safe["reasoning_format"] = (
        "deepseek"
        if thinking_enabled
        else "none"
    )

    safe["messages"] = _inject_final_answer_control(safe.get("messages", []))

    logger.info(
        "[FINAL_GUARD] payload_applied route_label=%s max_tokens=%s min_tokens=%s "
        "temperature=%s top_p=%s top_k=%s repeat_penalty=%s "
        "repeat_last_n=%s thinking=%s reasoning_format=%s",
        route_label,
        safe.get("max_tokens"),
        min_tokens,
        safe.get("temperature"),
        safe.get("top_p"),
        safe.get("top_k"),
        safe.get("repeat_penalty"),
        safe.get("repeat_last_n"),
        chat_template_kwargs.get("enable_thinking"),
        safe.get("reasoning_format"),
    )
    return safe


def _extract_sse_content_piece(line: str) -> str:
    """Extract assistant text from one flat or OpenAI-style SSE data line."""
    if not isinstance(line, str) or not line.startswith("data: "):
        return ""
    raw = line[6:].strip()
    if not raw or raw == "[DONE]":
        return ""
    try:
        obj = json.loads(raw)
    except Exception:
        return ""

    choices = obj.get("choices") if isinstance(obj, dict) else None
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    content = obj.get("content") if isinstance(obj, dict) else None
    return content if isinstance(content, str) else ""


def _detect_repetition_loop(text: str) -> dict:
    """Detect a repeated long token sequence and return a safe truncation point."""
    meta = {
        "detected": False,
        "reason": "",
        "cutoff": len(text or ""),
        "original_chars": len(text or ""),
        "final_chars": len(text or ""),
    }

    if not _env_flag_enabled("KVEN2_REPETITION_GUARD") and os.getenv("KVEN2_REPETITION_GUARD") is not None:
        return meta

    minimum_chars = _env_int("KVEN2_REPETITION_MIN_CHARS", 1000, 400, 10000)
    if not isinstance(text, str) or len(text) < minimum_chars:
        return meta

    matches = list(re.finditer(r"\S+", text))
    if len(matches) < 110:
        return meta

    def normalize_token(token: str) -> str:
        cleaned = re.sub(r"^\W+|\W+$", "", token, flags=re.UNICODE).lower()
        return cleaned or token.lower()

    tokens = [normalize_token(m.group(0)) for m in matches]
    search_window_tokens = _env_int("KVEN2_REPETITION_SEARCH_TOKENS", 1400, 300, 5000)

    # A repeated exact sequence of 50+ tokens is extremely unlikely in a normal
    # answer but is characteristic of the observed Final check / Let's write loop.
    for sequence_tokens in (120, 80, 50):
        if len(tokens) < sequence_tokens * 2 + 10:
            continue
        current_start = len(tokens) - sequence_tokens
        tail = tuple(tokens[current_start:])
        earliest = max(0, current_start - search_window_tokens)
        latest = current_start - max(10, sequence_tokens // 2)
        for start in range(latest, earliest - 1, -1):
            if tuple(tokens[start:start + sequence_tokens]) == tail:
                cutoff = matches[current_start].start()
                meta.update({
                    "detected": True,
                    "reason": f"repeated_{sequence_tokens}_token_sequence",
                    "cutoff": cutoff,
                    "final_chars": cutoff,
                })
                return meta

    return meta


def _sanitize_repetitive_text(text: str) -> tuple[str, dict]:
    """Trim only the repeated tail; preserve the first complete occurrence."""
    text = text or ""
    meta = _detect_repetition_loop(text)
    if not meta.get("detected"):
        return text, meta

    cleaned = text[: int(meta.get("cutoff", len(text)))].rstrip()
    meta["final_chars"] = len(cleaned)
    logger.warning(
        "[FINAL_GUARD] repetition_detected reason=%s original_chars=%s final_chars=%s",
        meta.get("reason"),
        meta.get("original_chars"),
        meta.get("final_chars"),
    )
    return cleaned, meta


# -----------------------------------------------------------------------------
# Tool-calling probe + local gateway registry + bounded hidden decision loop
# -----------------------------------------------------------------------------
# Phase 2 for Kven II tools:
# - keep the confirmed gateway-owned model-driven get_time loop;
# - add the second tool: read_file through agent_sandbox.py /read_file;
# - compare incoming tool names with the local registry;
# - never forward tools directly to llama.cpp;
# - execute only one allowed tool per request;
# - keep shell/fetch/email out of the loop for this step.



def _json_preview(value, limit: int = 800) -> str:
    """Compact bounded JSON preview for logs; never raises."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text



def _apply_generation_passthrough(body: dict, payload: dict) -> list[str]:
    """Copy safe generation controls from incoming OpenAI-style request."""
    forwarded = []

    for key in PASSTHROUGH_PARAMS:
        if key in body and body[key] is not None:
            payload[key] = body[key]
            forwarded.append(key)

    if (
        "max_completion_tokens" in body
        and body["max_completion_tokens"] is not None
        and "max_tokens" not in payload
    ):
        payload["max_tokens"] = body["max_completion_tokens"]
        forwarded.append("max_completion_tokens->max_tokens")

    return forwarded



# -----------------------------------------------------------------------------
# OWUI RAG context detection + write_path sanitization
# -----------------------------------------------------------------------------

OWUI_RAG_TASK_MARKER = "### Task:\nRespond to the user query using the provided context"
OWUI_RAG_CONTEXT_OPEN = "<context>"
OWUI_RAG_CONTEXT_CLOSE = "</context>"
OWUI_RAG_SOURCE_OPEN = "<source "


def _is_owui_rag_context_text(text: str) -> bool:
    """Detect OpenWebUI Knowledge/RAG wrapper inserted into user.content."""
    if not isinstance(text, str):
        return False

    return (
        OWUI_RAG_TASK_MARKER in text
        and OWUI_RAG_CONTEXT_OPEN in text
        and OWUI_RAG_CONTEXT_CLOSE in text
        and OWUI_RAG_SOURCE_OPEN in text
    )


def _extract_xml_attr(attr_text: str, attr_name: str) -> str | None:
    """Tiny best-effort XML-ish attribute extractor for OWUI source tags."""
    try:
        m = re.search(rf'{re.escape(attr_name)}="([^"]*)"', attr_text)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


def _extract_owui_rag_source_summaries(text: str) -> list[dict]:
    """Return compact source metadata from real OWUI <context><source ...> tags only."""
    sources = []
    if not isinstance(text, str):
        return sources

    try:
        context_blocks = re.findall(
            r"<context\b[^>]*>(.*?)</context>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        context_text = "\n".join(context_blocks) if context_blocks else text

        for m in re.finditer(r"<source\b([^>]*)>", context_text, flags=re.IGNORECASE):
            attrs = m.group(1) or ""
            source = {
                "id": _extract_xml_attr(attrs, "id"),
                "name": _extract_xml_attr(attrs, "name"),
                "resource_type": _extract_xml_attr(attrs, "resource-type"),
                "resource_id": _extract_xml_attr(attrs, "resource-id"),
            }

            # Keep only actual source-like entries. This drops examples from OWUI instructions.
            if any(source.values()):
                sources.append(source)

            if len(sources) >= 10:
                break
    except Exception:
        return sources

    # OWUI may repeat several chunks with the same source id. Log unique
    # sources rather than one entry per chunk.
    unique_sources = []
    seen = set()
    for source in sources:
        key = (
            source.get("id"),
            source.get("name"),
            source.get("resource_type"),
            source.get("resource_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(source)

    return unique_sources

def _extract_owui_rag_user_query(text: str) -> str:
    """
    Extract the real user query from an OWUI Knowledge/RAG wrapper.

    OWUI may combine Knowledge, web-search, and other retrieval blocks in one
    synthetic user message. Always take the text after the *last* closing
    </context> tag, then defensively remove any residual context/source blocks.
    This prevents external source chunks from reaching Kven retrieval/write_path.
    """
    if not isinstance(text, str):
        return ""

    candidate = text
    if OWUI_RAG_CONTEXT_CLOSE in candidate:
        candidate = candidate.rsplit(OWUI_RAG_CONTEXT_CLOSE, 1)[1]

    # Defensive cleanup for malformed/nested/multi-context OWUI payloads.
    candidate = re.sub(
        r"<context\b[^>]*>.*?</context>",
        " ",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = re.sub(
        r"<source\b[^>]*>.*?</source>",
        " ",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = re.sub(r"</?(?:context|source)\b[^>]*>", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"[ \t]+", " ", candidate)
    candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()

    if candidate:
        # A genuine user query should be short. Bound the fallback without ever
        # retaining the beginning of a synthetic OWUI instruction/source block.
        return candidate[-2000:].strip()

    return ""


def _sanitize_owui_rag_messages_for_write_path(messages: list) -> tuple[list, dict]:
    """
    Remove OWUI RAG source chunks before write_path.

    This preserves the real user query while preventing external document chunks
    from being stored as Kven episodic/semantic memory.
    """
    if not isinstance(messages, list):
        return messages, {"detected": False}

    sanitized = []
    meta = {
        "detected": False,
        "rag_message_indices": [],
        "source_count": 0,
        "sources": [],
        "original_chars": 0,
        "sanitized_chars": 0,
    }

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue

        new_msg = dict(msg)
        content = new_msg.get("content")

        if new_msg.get("role") == "user" and isinstance(content, str) and _is_owui_rag_context_text(content):
            cleaned_query = _extract_owui_rag_user_query(content)
            sources = _extract_owui_rag_source_summaries(content)

            new_msg["content"] = cleaned_query

            meta["detected"] = True
            meta["rag_message_indices"].append(idx)
            meta["sources"].extend(sources)
            meta["original_chars"] += len(content)
            meta["sanitized_chars"] += len(cleaned_query)

        sanitized.append(new_msg)

    meta["source_count"] = len(meta["sources"])
    meta["sources"] = meta["sources"][:10]
    return sanitized, meta


def _sanitize_messages_for_write_path(messages: list) -> tuple[list, dict]:
    """
    Prepare conversation history for Kven memory extraction.

    In addition to removing OWUI RAG source chunks, drop native tool protocol
    messages so raw tool results and assistant.tool_calls cannot become episodic
    or semantic memories. The real user query and ordinary conversation messages
    are preserved.
    """
    sanitized, meta = _sanitize_owui_rag_messages_for_write_path(messages)
    if not isinstance(sanitized, list):
        return sanitized, meta

    filtered = []
    removed_indices = []

    for idx, msg in enumerate(sanitized):
        if not isinstance(msg, dict):
            filtered.append(msg)
            continue

        role = msg.get("role")
        if role == "tool":
            removed_indices.append(idx)
            continue

        if role == "assistant" and msg.get("tool_calls"):
            removed_indices.append(idx)
            continue

        clean_msg = dict(msg)
        clean_msg.pop("tool_calls", None)
        clean_msg.pop("tool_call_id", None)
        filtered.append(clean_msg)

    meta = dict(meta or {})
    meta["tool_protocol_message_indices_removed"] = removed_indices
    meta["tool_protocol_messages_removed"] = len(removed_indices)
    return filtered, meta

# -----------------------------------------------------------------------------
# OWUI incoming payload debug
# -----------------------------------------------------------------------------

def _env_flag_enabled(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _safe_content_shape(content) -> dict:
    """Return bounded, non-fatal summary of OpenAI/OWUI message content."""
    if isinstance(content, str):
        is_owui_rag = _is_owui_rag_context_text(content)
        return {
            "type": "str",
            "chars": len(content),
            "preview_head": content[:1000],
            "preview_tail": content[-1000:] if len(content) > 1000 else "",
            "contains_kven_rag_test_marker": "KVEN_RAG_TEST_2026_07_15" in content,
            "is_owui_rag_context": is_owui_rag,
            "contains_owui_context_tag": OWUI_RAG_CONTEXT_OPEN in content and OWUI_RAG_CONTEXT_CLOSE in content,
            "contains_owui_source_tag": OWUI_RAG_SOURCE_OPEN in content,
            "owui_rag_user_query_preview": _extract_owui_rag_user_query(content)[:500] if is_owui_rag else "",
            "owui_rag_sources": _extract_owui_rag_source_summaries(content) if is_owui_rag else [],
        }

    if isinstance(content, list):
        parts = []
        for item in content[:8]:
            if isinstance(item, dict):
                parts.append({
                    "type": item.get("type"),
                    "keys": sorted(item.keys()),
                    "text_preview": str(item.get("text", ""))[:200] if "text" in item else "",
                })
            else:
                parts.append({"type": type(item).__name__, "preview": repr(item)[:120]})
        return {
            "type": "list",
            "items": len(content),
            "parts_preview": parts,
        }

    if isinstance(content, dict):
        return {
            "type": "dict",
            "keys": sorted(content.keys()),
            "preview": _json_preview(content, limit=500),
        }

    return {
        "type": type(content).__name__,
        "preview": repr(content)[:200],
    }


def _summarize_incoming_payload(body: dict) -> dict:
    """Summarize OWUI/OpenAI-compatible payload without dumping full content."""
    known_openai_keys = {
        "model",
        "messages",
        "stream",
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "repeat_penalty",
        "tools",
        "tool_choice",
        "response_format",
        "metadata",
        "user",
    }

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    message_summaries = []
    tool_related_messages = []

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            message_summaries.append({
                "index": idx,
                "type": type(msg).__name__,
                "preview": repr(msg)[:200],
            })
            continue

        msg_keys = sorted(msg.keys())
        role = msg.get("role")
        has_tool_fields = any(k in msg for k in ("tool_calls", "tool_call_id", "name"))
        if role == "tool" or has_tool_fields:
            tool_related_messages.append(idx)

        message_summaries.append({
            "index": idx,
            "role": role,
            "keys": msg_keys,
            "content": _safe_content_shape(msg.get("content")),
            "has_tool_calls": "tool_calls" in msg,
            "tool_call_id": msg.get("tool_call_id"),
            "name": msg.get("name"),
        })

    owui_candidate_keys = [
        "files",
        "documents",
        "citations",
        "sources",
        "context",
        "contexts",
        "knowledge",
        "rag",
        "metadata",
        "params",
        "features",
        "variables",
        "tool_ids",
        "tool_servers",
        "chat_id",
        "session_id",
        "id",
    ]

    return {
        "top_level_keys": sorted(body.keys()),
        "unknown_top_level_keys": sorted(set(body.keys()) - known_openai_keys),
        "owui_candidate_keys_present": {
            k: {
                "type": type(body.get(k)).__name__,
                "preview": _json_preview(body.get(k), limit=800),
            }
            for k in owui_candidate_keys
            if k in body
        },
        "model": body.get("model"),
        "stream": body.get("stream"),
        "messages_count": len(messages),
        "roles": [m.get("role") for m in messages if isinstance(m, dict)],
        "incoming_tools": isinstance(body.get("tools"), list) and bool(body.get("tools")),
        "tools_count": len(body.get("tools", [])) if isinstance(body.get("tools"), list) else 0,
        "tool_choice": body.get("tool_choice"),
        "tool_related_message_indices": tool_related_messages,
        "messages_preview": message_summaries,
    }


def _env_int(name: str, default: int, min_value: int = 100, max_value: int = 20000) -> int:
    try:
        value = int(str(os.getenv(name, "")).strip())
    except Exception:
        value = default
    return max(min_value, min(value, max_value))


def _debug_log_incoming_payload(body: dict) -> None:
    """Log bounded incoming payload summary when explicitly enabled by env."""
    if not _env_flag_enabled("KVEN2_DEBUG_INCOMING_PAYLOAD"):
        return

    try:
        summary = _summarize_incoming_payload(body)
        limit = _env_int("KVEN2_DEBUG_INCOMING_PAYLOAD_MAX_CHARS", 6000, 1000, 20000)
        logger.info("[OWUI_PAYLOAD_DEBUG] %s", _json_preview(summary, limit=limit))
    except Exception as exc:
        logger.error("[OWUI_PAYLOAD_DEBUG] failed error=%s", exc, exc_info=True)


# -----------------------------------------------------------------------------
# OWUI tool policy: library is read-only; Kven II owns experiential memory
# -----------------------------------------------------------------------------

OWUI_BLOCKED_MEMORY_WRITE_TOOLS = frozenset({
    "add_memory",
    "update_memory",
    "replace_memory_content",
    "delete_memory",
})


def _native_tool_name(tool: dict) -> str:
    """Best-effort extraction of an OpenAI/OWUI function tool name."""
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function.get("name"))
    if tool.get("name"):
        return str(tool.get("name"))
    return ""


def _explicit_native_tool_choice_name(tool_choice) -> str:
    """Extract an explicitly forced function name from OpenAI tool_choice."""
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            return tool_choice
        return ""
    if not isinstance(tool_choice, dict):
        return ""
    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function.get("name"))
    if tool_choice.get("name"):
        return str(tool_choice.get("name"))
    return ""


OWUI_MEMORY_READ_TOOLS = frozenset({
    "search_memories",
    "list_memory_paths",
    "read_memory_path",
    "list_memories",
})

NATIVE_MUTATING_TOOLS = frozenset({
    "write_note",
    "replace_note_content",
    "create_tasks",
    "update_task",
    "create_automation",
    "update_automation",
    "toggle_automation",
    "delete_automation",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
})

PURE_MEMORY_WRITE_OPENERS = (
    "запомни",
    "запомнить",
    "зафиксируй",
    "фиксируем",
    "remember",
    "remember that",
    "store this",
    "record this",
)

MEMORY_WRITE_EXTERNAL_ACTION_MARKERS = (
    "найди",
    "узнай",
    "проверь",
    "получи",
    "загрузи",
    "прочитай",
    "search",
    "find",
    "look up",
    "fetch",
    "download",
    "read the",
    "http://",
    "https://",
)

NATIVE_TOOL_DECISION_POLICY = """
NATIVE TOOL DECISION POLICY:
- This is one bounded tool-decision pass. Do not narrate your plan.
- If a listed tool is necessary, emit a native OpenAI tool call immediately.
- Never print XML-like <tool_call>, <function>, or <parameter> tags.
- Never say that you will call a tool later. Either call it now or output exactly KVEN_NO_TOOL.
- If no tool is necessary, output exactly KVEN_NO_TOOL and nothing else.
- Never discuss which tools are or are not available.
- At most one mutating tool call is allowed in one user turn.
- A request to remember a user-supplied fact does not need an OWUI tool: Kven II persists it automatically after the answer.
""".strip()

NATIVE_TOOL_CORRECTION_POLICY = """
CORRECTION: Your previous attempt did not produce a valid native tool call.
Do not explain, plan, or emit XML. If a tool is needed, emit the native OpenAI tool call now.
Otherwise output exactly KVEN_NO_TOOL and nothing else.
""".strip()


def _is_pure_kven_memory_write_request(messages: list) -> bool:
    """True for a direct user-supplied memory statement that needs no external tool."""
    text = _last_user_text(messages).strip().lower()
    if not text:
        return False

    if not any(text.startswith(marker) for marker in PURE_MEMORY_WRITE_OPENERS):
        return False

    return not any(marker in text for marker in MEMORY_WRITE_EXTERNAL_ACTION_MARKERS)


def _disable_tools_for_pure_kven_memory_write(body: dict, messages: list) -> dict:
    """Route direct 'remember this fact' requests to Kven write_path only."""
    meta = {"matched": False, "tools_before": 0, "tools_after": 0, "tool_choice_cleared": False}
    if not isinstance(body, dict) or not _is_pure_kven_memory_write_request(messages):
        return meta

    tools = body.get("tools")
    if isinstance(tools, list):
        meta["tools_before"] = len(tools)
        body["tools"] = []
    if body.get("tool_choice") is not None:
        body.pop("tool_choice", None)
        meta["tool_choice_cleared"] = True

    meta["matched"] = True
    logger.info(
        "[OWUI_TOOL_POLICY] kven_memory_write_intent=True tools_disabled=%s "
        "tool_choice_cleared=%s",
        meta["tools_before"],
        meta["tool_choice_cleared"],
    )
    return meta


def _apply_owui_read_only_library_policy(body: dict) -> dict:
    """
    Remove OWUI Memory mutation tools before the model sees the catalogue.

    OWUI Knowledge/Collections and OWUI Memory remain readable reference
    sources. Durable user-authored experience is written only by Kven II's
    write_path, which prevents duplicate facts and ambiguous ownership.
    """
    meta = {
        "tools_before": 0,
        "tools_after": 0,
        "removed_names": [],
        "tool_choice_cleared": False,
    }

    if not isinstance(body, dict):
        return meta

    tools = body.get("tools")
    if not isinstance(tools, list):
        return meta

    meta["tools_before"] = len(tools)
    kept = []
    removed = []

    for tool in tools:
        name = _native_tool_name(tool)
        if name in OWUI_BLOCKED_MEMORY_WRITE_TOOLS:
            removed.append(name)
            continue
        kept.append(tool)

    body["tools"] = kept
    meta["tools_after"] = len(kept)
    meta["removed_names"] = sorted(set(removed))

    forced_name = _explicit_native_tool_choice_name(body.get("tool_choice"))
    if forced_name in OWUI_BLOCKED_MEMORY_WRITE_TOOLS:
        body.pop("tool_choice", None)
        meta["tool_choice_cleared"] = True

    if removed or meta["tool_choice_cleared"]:
        logger.info(
            "[OWUI_TOOL_POLICY] library_read_only=True tools_before=%s tools_after=%s "
            "removed_write_tools=%s names=%s tool_choice_cleared=%s",
            meta["tools_before"],
            meta["tools_after"],
            len(meta["removed_names"]),
            meta["removed_names"],
            meta["tool_choice_cleared"],
        )

    return meta


# -----------------------------------------------------------------------------
# Native OpenAI / OpenWebUI tool-calling transport
# -----------------------------------------------------------------------------
# OpenWebUI native tool calling sends OpenAI-compatible `tools`, assistant
# `tool_calls`, and follow-up `role=tool` messages. User-facing requests are
# enriched by Kven and then proxied without altering the native tool protocol.
# OWUI's own internal title/tag/summary prompts remain transparent pass-through.
# Raw tool results are always removed before Kven write_path.


def _message_has_native_tool_protocol_fields(message: dict) -> bool:
    """Detect OpenAI tool-call continuation messages."""
    if not isinstance(message, dict):
        return False

    if message.get("role") == "tool":
        return True

    if message.get("tool_calls"):
        return True

    if message.get("tool_call_id"):
        return True

    return False


def _is_native_openai_tool_protocol(body: dict, messages: list) -> bool:
    """Return True when request belongs to OpenAI native tool-calling protocol."""
    if isinstance(body.get("tools"), list) and bool(body.get("tools")):
        return True

    if body.get("tool_choice") is not None:
        return True

    if isinstance(messages, list) and any(
        _message_has_native_tool_protocol_fields(m) for m in messages if isinstance(m, dict)
    ):
        return True

    return False


def _summarize_native_tools(body: dict, messages: list) -> dict:
    """Compact diagnostic summary for native tool pass-through mode."""
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        tools = []

    tool_names = []
    for tool in tools[:80]:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = fn.get("name") or tool.get("name")
        if name:
            tool_names.append(name)

    tool_message_indices = []
    assistant_tool_call_indices = []
    if isinstance(messages, list):
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "tool" or msg.get("tool_call_id"):
                tool_message_indices.append(idx)
            if msg.get("tool_calls"):
                assistant_tool_call_indices.append(idx)

    return {
        "tools_count": len(tools),
        "tool_names": tool_names,
        "tool_choice": body.get("tool_choice"),
        "stream": body.get("stream", False),
        "tool_message_indices": tool_message_indices,
        "assistant_tool_call_indices": assistant_tool_call_indices,
    }


def _messages_include_native_tool_continuation(messages: list) -> bool:
    """
    True only for an active OpenAI tool continuation at the message tail.

    Historical tool calls from completed turns must not force later user
    messages into continuation mode. An active continuation has this shape:

        assistant(tool_calls) -> tool [-> tool ...] -> end of messages
    """
    if not isinstance(messages, list) or not messages:
        return False

    index = len(messages) - 1

    # Ignore malformed/non-dict tail entries without treating old history as
    # an active continuation.
    while index >= 0 and not isinstance(messages[index], dict):
        index -= 1

    saw_tool_result = False

    # One assistant tool call may produce several contiguous tool results.
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
        return False

    while index >= 0 and not isinstance(messages[index], dict):
        index -= 1

    if index < 0:
        return False

    assistant_message = messages[index]
    return (
        assistant_message.get("role") == "assistant"
        and isinstance(assistant_message.get("tool_calls"), list)
        and bool(assistant_message.get("tool_calls"))
    )


def _messages_include_tool_result(messages: list) -> bool:
    """True only when tool results belong to the active message-tail phase."""
    return _messages_include_native_tool_continuation(messages)


def _prepare_native_tool_payload(
    body: dict,
    messages: list,
    model_name: str,
    *,
    minimum_first_pass_tokens: int = 512,
) -> dict:
    """Build a native OpenAI tool payload with deterministic safeguards."""
    native_payload = dict(body)
    native_payload["model"] = model_name
    native_payload["messages"] = messages
    native_payload["cache_prompt"] = True

    # First pass: make tool selection deterministic and give Qwen enough budget
    # to emit tool_calls instead of spending the whole response in reasoning.
    if not _messages_include_tool_result(messages):
        if native_payload.get("temperature") is None:
            native_payload["temperature"] = 0.0
            logger.info("[OWUI_NATIVE_TOOLS] forced_temperature_zero reason=first_tool_call_pass")
        try:
            current_max_tokens = int(
                native_payload.get("max_tokens")
                or native_payload.get("max_completion_tokens")
                or 0
            )
        except Exception:
            current_max_tokens = 0
        safe_minimum = max(512, int(minimum_first_pass_tokens or 512))
        if current_max_tokens < safe_minimum:
            native_payload["max_tokens"] = safe_minimum
            native_payload.pop("max_completion_tokens", None)
            logger.info(
                "[OWUI_NATIVE_TOOLS] raised_max_tokens_for_tool_call old=%s new=%s",
                current_max_tokens,
                safe_minimum,
            )

    # Second pass: prevent the backend from requesting the same tool again.
    if _messages_include_tool_result(messages) and native_payload.get("tool_choice") != "none":
        native_payload["tool_choice"] = "none"
        logger.info("[OWUI_NATIVE_TOOLS] forced_tool_choice_none reason=tool_result_message_present")

    return native_payload


def _normalize_native_tool_response_for_owui(resp_json: dict) -> dict:
    """
    Make llama.cpp native tool-call responses closer to OpenAI's schema.

    llama.cpp returns useful `tool_calls`, but also emits non-standard
    `reasoning_content` and uses an empty string for assistant content. Some
    OWUI paths are stricter than curl tests, so normalize only the native
    tool-call shape and leave ordinary answers untouched.
    """
    if not isinstance(resp_json, dict):
        return resp_json

    choices = resp_json.get("choices")
    if not isinstance(choices, list):
        return resp_json

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        tool_calls = message.get("tool_calls")

        # OpenAI-compatible responses do not contain provider-specific reasoning
        # fields. Strip this for every native-tool response, not only successful
        # tool-call messages. Otherwise OWUI may see an empty content string plus
        # reasoning_content and render a blank/hanging answer.
        had_reasoning = "reasoning_content" in message
        message.pop("reasoning_content", None)

        # OpenAI commonly uses null content for assistant messages that only
        # carry tool_calls. This is safer for OWUI than an empty string.
        if tool_calls and message.get("content", "") == "":
            message["content"] = None

        # If the model failed to emit tool_calls and produced only hidden
        # reasoning, return a visible diagnostic instead of a silent blank. This
        # does not execute tools; it only makes failure mode observable.
        if (not tool_calls) and had_reasoning and message.get("content", "") == "":
            message["content"] = "[Kven native tools] Model returned reasoning only and did not emit tool_calls. Retry with temperature=0 and max_tokens>=512."

        if isinstance(tool_calls, list):
            for idx, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                call.setdefault("type", "function")
                if not call.get("id"):
                    call["id"] = f"call_{idx}"
                fn = call.get("function")
                if isinstance(fn, dict):
                    args = fn.get("arguments", "{}")
                    if not isinstance(args, str):
                        try:
                            fn["arguments"] = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            fn["arguments"] = str(args)

        # Be explicit for clients that key off finish_reason.
        if not choice.get("finish_reason"):
            choice["finish_reason"] = "tool_calls"

    return resp_json



def _with_native_decision_policy(messages: list, *, correction: bool = False) -> list:
    """Add bounded decision instructions while preserving a single leading system message."""
    policy = NATIVE_TOOL_DECISION_POLICY
    if correction:
        policy += "\n\n" + NATIVE_TOOL_CORRECTION_POLICY

    copied = []
    for msg in messages or []:
        copied.append(dict(msg) if isinstance(msg, dict) else msg)

    if copied and isinstance(copied[0], dict) and copied[0].get("role") == "system":
        existing = _content_to_text(copied[0].get("content")).strip()
        copied[0]["content"] = (existing + "\n\n--- NATIVE TOOL CONTROL ---\n\n" + policy).strip()
    else:
        copied.insert(0, {"role": "system", "content": policy})

    return copied


def _allowed_native_tool_names(payload: dict) -> set[str]:
    names = set()
    tools = payload.get("tools", []) if isinstance(payload, dict) else []
    if not isinstance(tools, list):
        return names
    for tool in tools:
        name = _native_tool_name(tool)
        if name:
            names.add(name)
    return names


def _coerce_tool_arguments(value) -> str | None:
    """Return canonical JSON arguments or None when arguments are unusable."""
    if value is None or value == "":
        return "{}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if not isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return None

    text = value.strip()
    if not text:
        return "{}"
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return json.dumps(parsed, ensure_ascii=False)


def _validate_and_limit_native_tool_calls(tool_calls, allowed_names: set[str]) -> tuple[list, dict]:
    """Validate names/arguments, deduplicate calls, and allow only one mutation."""
    meta = {
        "received": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "accepted": 0,
        "invalid": 0,
        "duplicates": 0,
        "mutations_dropped": 0,
        "total_dropped": 0,
    }
    if not isinstance(tool_calls, list):
        return [], meta

    accepted = []
    seen = set()
    mutation_seen = False
    max_calls = _env_int("KVEN2_NATIVE_TOOL_MAX_CALLS", 4, 1, 10)

    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            meta["invalid"] += 1
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            meta["invalid"] += 1
            continue
        name = str(fn.get("name") or "").strip()
        args = _coerce_tool_arguments(fn.get("arguments", "{}"))
        if not name or name not in allowed_names or args is None:
            meta["invalid"] += 1
            continue

        signature = (name, args)
        if signature in seen:
            meta["duplicates"] += 1
            continue
        seen.add(signature)

        if name in NATIVE_MUTATING_TOOLS:
            if mutation_seen:
                meta["mutations_dropped"] += 1
                continue
            mutation_seen = True

        if len(accepted) >= max_calls:
            meta["total_dropped"] += 1
            continue

        accepted.append({
            "id": str(call.get("id") or f"call_kven_{idx}"),
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    meta["accepted"] = len(accepted)
    return accepted, meta


def _extract_pseudo_native_tool_calls(text: str, allowed_names: set[str]) -> list:
    """Recover the XML-like pseudo tool syntax occasionally emitted by Qwen."""
    if not isinstance(text, str) or "<tool_call" not in text.lower():
        return []

    recovered = []
    blocks = re.findall(r"<tool_call\b[^>]*>(.*?)</tool_call>", text, flags=re.I | re.S)
    if not blocks:
        blocks = [text]

    for idx, block in enumerate(blocks[:4]):
        # JSON-shaped pseudo call: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        stripped = block.strip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                name = str(obj.get("name") or (obj.get("function") or {}).get("name") or "")
                arguments = obj.get("arguments")
                if arguments is None and isinstance(obj.get("function"), dict):
                    arguments = obj["function"].get("arguments", {})
                if name in allowed_names:
                    recovered.append({
                        "id": f"call_kven_pseudo_{idx}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments or {}},
                    })
                    continue
            except Exception:
                pass

        match = re.search(r"<function\s*=\s*([^>\s]+)\s*>", block, flags=re.I)
        if not match:
            continue
        name = match.group(1).strip().strip('"\'')
        if name not in allowed_names:
            continue

        args = {}
        for p_match in re.finditer(
            r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
            block,
            flags=re.I | re.S,
        ):
            key = p_match.group(1).strip().strip('"\'')
            value = p_match.group(2).strip()
            try:
                args[key] = json.loads(value)
            except Exception:
                args[key] = value

        recovered.append({
            "id": f"call_kven_pseudo_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    return recovered


_NATIVE_TOOL_FAILURE_MARKERS = (
    "<tool_call",
    "<function=",
    "i will use the tool",
    "i will call the tool",
    "i should use the tool",
    "let me use the tool",
    "i will execute the tool call",
    "tool call first",
    "вызову инструмент",
    "использую инструмент",
    "нужно вызвать инструмент",
)

_CONTINUATION_CONTENT_BLOCK_MARKERS = (
    *_NATIVE_TOOL_FAILURE_MARKERS,
    "<parameter=",
    "<think>",
    "<|think_start|>",
)


def _looks_like_failed_native_tool_attempt(
    content: str,
    reasoning: str,
) -> bool:
    combined = f"{reasoning}\n{content}".lower()
    return any(
        marker in combined
        for marker in _NATIVE_TOOL_FAILURE_MARKERS
    )


def _find_continuation_content_marker(
    text: str,
) -> tuple[int | None, str]:
    """Return the earliest forbidden marker in continuation content."""
    lowered = (text or "").lower()
    earliest = None
    matched = ""

    for marker in _CONTINUATION_CONTENT_BLOCK_MARKERS:
        position = lowered.find(marker)

        if position < 0:
            continue

        if earliest is None or position < earliest:
            earliest = position
            matched = marker

    return earliest, matched


def _continuation_safe_emit_end(text: str) -> int:
    """
    Hold only a suffix that may be the beginning of a forbidden marker.

    Normal text is released immediately. For example, a trailing "<to" is
    retained until it either becomes "<tool_call" or stops matching it.
    """
    lowered = (text or "").lower()
    holdback = 0

    for marker in _CONTINUATION_CONTENT_BLOCK_MARKERS:
        maximum = min(
            len(lowered),
            len(marker) - 1,
        )

        for size in range(maximum, 0, -1):
            if lowered.endswith(marker[:size]):
                holdback = max(holdback, size)
                break

    return len(text or "") - holdback


def _clean_direct_native_answer(content: str) -> str:
    cleaned = strip_reasoning(content or "")
    cleaned = re.sub(r"<tool_call\b[^>]*>.*?</tool_call>", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_clean_tool_continuation_messages(messages: list) -> list:
    """Flatten native tool protocol history into ordinary final-answer context."""
    sanitized, _ = _sanitize_owui_rag_messages_for_write_path(messages)
    clean_messages = []
    tool_results = []

    for idx, message in enumerate(sanitized or []):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = _content_to_text(message.get("content")).strip()

        if role == "tool":
            if content:
                label = (
                    message.get("name")
                    or message.get("tool_call_id")
                    or f"tool_result_{idx + 1}"
                )
                tool_results.append(f"[{label}]\n{content}")
            continue

        if role == "assistant" and message.get("tool_calls"):
            continue

        if (
            role == "assistant"
            and _looks_like_failed_native_tool_attempt(content, "")
        ):
            continue

        clean_message = dict(message)
        clean_message.pop("tool_calls", None)
        clean_message.pop("tool_call_id", None)
        clean_messages.append(clean_message)

    results_text = "\n\n".join(tool_results).strip()
    if not results_text:
        results_text = "[No readable tool results were returned.]"

    continuation_policy = f"""
KVEN TOOL CONTINUATION FINAL CONTROL:
Tool results are already available below.
Use them only when relevant to the user's latest request.
Do not call, request, describe, or simulate any tool.
Never output <tool_call>, <function>, or <parameter> tags.
Return only the completed user-facing answer.

TOOL RESULTS:
{results_text}
""".strip()

    if (
        clean_messages
        and isinstance(clean_messages[0], dict)
        and clean_messages[0].get("role") == "system"
    ):
        existing = _content_to_text(
            clean_messages[0].get("content")
        ).strip()
        clean_messages[0]["content"] = (
            existing
            + "\n\n--- TOOL CONTINUATION FINAL CONTROL ---\n\n"
            + continuation_policy
        ).strip()
    else:
        clean_messages.insert(
            0,
            {"role": "system", "content": continuation_policy},
        )

    return clean_messages


def _completion_with_tool_calls(base: dict, tool_calls: list) -> dict:
    """Build a normalized OpenAI completion containing only native tool calls."""
    result = {
        "id": base.get("id") or "chatcmpl-kven-tools",
        "object": base.get("object") or "chat.completion",
        "created": base.get("created") or 0,
        "model": base.get("model") or settings.MAIN_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }],
    }
    if base.get("usage") is not None:
        result["usage"] = base.get("usage")
    return result


def _completion_with_direct_answer(base: dict, content: str, finish_reason: str = "stop") -> dict:
    result = {
        "id": base.get("id") or "chatcmpl-kven-answer",
        "object": base.get("object") or "chat.completion",
        "created": base.get("created") or 0,
        "model": base.get("model") or settings.MAIN_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason or "stop",
        }],
    }
    if base.get("usage") is not None:
        result["usage"] = base.get("usage")
    return result


def _completion_json_to_sse(completion: dict) -> bytes:
    """Serialize one buffered OpenAI completion as a valid compact SSE stream."""
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    base = {
        "id": completion.get("id") or "chatcmpl-kven",
        "object": "chat.completion.chunk",
        "created": completion.get("created") or 0,
        "model": completion.get("model") or settings.MAIN_MODEL,
    }
    delta = {"role": "assistant"}
    if message.get("tool_calls"):
        calls = []
        for idx, call in enumerate(message.get("tool_calls") or []):
            copied = dict(call)
            copied.setdefault("index", idx)
            calls.append(copied)
        delta["content"] = None
        delta["tool_calls"] = calls
    else:
        delta["content"] = message.get("content") or ""

    first = dict(base)
    first["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
    final = dict(base)
    final["choices"] = [{
        "index": 0,
        "delta": {},
        "finish_reason": choice.get("finish_reason") or ("tool_calls" if message.get("tool_calls") else "stop"),
    }]
    if completion.get("usage") is not None:
        final["usage"] = completion.get("usage")

    text = (
        "data: " + json.dumps(first, ensure_ascii=False) + "\n\n"
        + "data: " + json.dumps(final, ensure_ascii=False) + "\n\n"
        + "data: [DONE]\n\n"
    )
    return text.encode("utf-8")


def _completion_response_for_client(completion: dict, stream_requested: bool) -> Response:
    if stream_requested:
        return Response(content=_completion_json_to_sse(completion), media_type="text/event-stream")
    return JSONResponse(content=completion)


async def _post_native_decision_json(
    payload: dict,
    chat_url: str,
    *,
    correction: bool,
    timeout_seconds: float,
    max_tokens: int,
) -> tuple[dict | None, str | None]:
    decision_payload = dict(payload)
    decision_payload["stream"] = False
    decision_payload["temperature"] = 0.0
    decision_payload["max_tokens"] = max_tokens
    decision_payload.pop("max_completion_tokens", None)

    # Tool selection must be a short classification pass. Thinking would spend
    # the entire bounded budget before Qwen emits either tool_calls or NO_TOOL.
    chat_template_kwargs = decision_payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    else:
        chat_template_kwargs = dict(chat_template_kwargs)

    chat_template_kwargs["enable_thinking"] = False
    decision_payload["chat_template_kwargs"] = chat_template_kwargs
    decision_payload["reasoning_format"] = "none"
    decision_payload.pop("reasoning_effort", None)

    decision_payload["messages"] = _with_native_decision_policy(
        payload.get("messages", []), correction=correction
    )

    logger.info(
        "[OWUI_NATIVE_GUARD] decision_start correction=%s max_tokens=%s "
        "timeout=%s thinking=False reasoning_format=none",
        correction,
        max_tokens,
        timeout_seconds,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(chat_url, json=decision_payload)
        if response.status_code >= 400:
            logger.warning(
                "[OWUI_NATIVE_GUARD] decision_http_error status=%s preview=%s",
                response.status_code,
                response.text[:1000],
            )
            return None, f"http_{response.status_code}"
        return response.json(), None
    except Exception as exc:
        logger.warning(
            "[OWUI_NATIVE_GUARD] decision_failed correction=%s error=%s",
            correction,
            exc,
            exc_info=True,
        )
        return None, str(exc)


async def _proxy_native_openai_tool_protocol(
    payload: dict,
    chat_url: str,
    *,
    timeout_seconds: float = 300.0,
) -> Response:
    """
    Forward native OpenAI tool-calling requests to llama.cpp and return response as-is.

    This is deliberately separate from _forward_to_backend_and_collect(), because that
    helper extracts only message.content and would drop message.tool_calls.
    """
    stream = bool(payload.get("stream", False))

    if stream:
        async def stream_backend():
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream("POST", chat_url, json=payload) as response:
                    logger.info(
                        "[OWUI_NATIVE_TOOLS] backend_stream_status=%s content_type=%s",
                        response.status_code,
                        response.headers.get("content-type", ""),
                    )
                    async for chunk in response.aiter_raw():
                        if chunk:
                            yield chunk

        return StreamingResponse(stream_backend(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(chat_url, json=payload)

    content_type = response.headers.get("content-type", "application/json")
    media_type = content_type.split(";", 1)[0] if content_type else "application/json"

    try:
        resp_json = response.json()
        choice0 = (resp_json.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        logger.info(
            "[OWUI_NATIVE_TOOLS] backend_json_status=%s finish_reason=%s has_tool_calls=%s content_len=%s",
            response.status_code,
            choice0.get("finish_reason"),
            bool(message.get("tool_calls")),
            len(message.get("content") or ""),
        )
        if message.get("tool_calls"):
            calls = []
            for call in message.get("tool_calls", [])[:10]:
                if isinstance(call, dict):
                    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                    calls.append({
                        "id": call.get("id"),
                        "type": call.get("type"),
                        "name": fn.get("name"),
                        "arguments_preview": str(fn.get("arguments", ""))[:500],
                    })
            logger.info("[OWUI_NATIVE_TOOLS] backend_tool_calls=%s", _json_preview(calls, limit=2000))

        normalized_json = _normalize_native_tool_response_for_owui(resp_json)
        return JSONResponse(content=normalized_json, status_code=response.status_code)
    except Exception as exc:
        logger.warning(
            "[OWUI_NATIVE_TOOLS] backend_non_json_or_parse_failed status=%s content_type=%s error=%s preview=%s",
            response.status_code,
            content_type,
            exc,
            response.text[:1000] if response.text else "",
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=media_type,
        )


def _schedule_hybrid_write_path(
    *,
    assistant_reply: str,
    write_path_messages: list,
    active_state: dict,
    owui_rag_meta: dict,
    skip_write_path: bool,
    generation_guard_meta: dict | None = None,
) -> None:
    """Apply the normal Kven memory safeguards after a hybrid native response."""
    logger.info(
        "[OWUI_NATIVE_HYBRID] final_assistant_reply_length=%s preview=%s",
        len(assistant_reply or ""),
        (assistant_reply or "")[:100],
    )

    if skip_write_path:
        logger.info("[OWUI_NATIVE_HYBRID] skipping_write_path reason=internal_request")
        return

    if generation_guard_meta and generation_guard_meta.get("detected"):
        logger.warning(
            "[OWUI_NATIVE_HYBRID] skipping_write_path reason=repetition_loop meta=%s",
            _json_preview(generation_guard_meta, limit=1000),
        )
        return

    if assistant_reply and len(assistant_reply.strip()) > 10:
        if owui_rag_meta.get("detected"):
            logger.info(
                "[OWUI_RAG_CONTEXT] write_path sanitized: external OWUI RAG source chunks "
                "and native tool protocol messages removed before memory pipeline."
            )
        else:
            logger.info(
                "[OWUI_NATIVE_HYBRID] write_path sanitized: native tool protocol "
                "messages removed before memory pipeline."
            )
        logger.info("[OWUI_NATIVE_HYBRID] triggering_background_write_path")
        asyncio.create_task(process_episodic(write_path_messages, assistant_reply, active_state))
        return

    logger.info(
        "[OWUI_NATIVE_HYBRID] skipping_write_path reason=assistant_reply_too_short_or_tool_call"
    )


async def _proxy_hybrid_final_response(
    payload: dict,
    chat_url: str,
    *,
    write_path_messages: list,
    active_state: dict,
    owui_rag_meta: dict,
    skip_write_path: bool,
    timeout_seconds: float = 1200.0,
) -> Response:
    """Proxy a bounded non-thinking final answer and safely schedule write_path."""
    final_payload = _apply_final_answer_safeguards(payload, route_label="hybrid_final")
    stream_requested = bool(final_payload.get("stream", False))

    if stream_requested:
        backend_lines, guard_meta = await _forward_to_backend_and_collect(
            final_payload,
            chat_url,
            timeout_seconds=timeout_seconds,
            route_label="hybrid_final",
        )
        assistant_reply = _extract_assistant_reply(backend_lines)
        _schedule_hybrid_write_path(
            assistant_reply=assistant_reply,
            write_path_messages=write_path_messages,
            active_state=active_state,
            owui_rag_meta=owui_rag_meta,
            skip_write_path=skip_write_path,
            generation_guard_meta=guard_meta,
        )

        async def generate():
            for line in backend_lines:
                yield line + "\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(chat_url, json=final_payload)

    content_type = response.headers.get("content-type", "application/json")
    media_type = content_type.split(";", 1)[0] if content_type else "application/json"

    try:
        resp_json = response.json()
        normalized_json = _normalize_native_tool_response_for_owui(resp_json)
        choice0 = (normalized_json.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        assistant_reply = strip_reasoning(message.get("content") or "")
        assistant_reply, guard_meta = _sanitize_repetitive_text(assistant_reply)
        message["content"] = assistant_reply
        if guard_meta.get("detected"):
            choice0["finish_reason"] = "stop"

        logger.info(
            "[OWUI_NATIVE_HYBRID] backend_json_status=%s finish_reason=%s "
            "has_tool_calls=%s content_len=%s repetition_detected=%s",
            response.status_code,
            choice0.get("finish_reason"),
            bool(message.get("tool_calls")),
            len(assistant_reply),
            bool(guard_meta.get("detected")),
        )

        _schedule_hybrid_write_path(
            assistant_reply=assistant_reply,
            write_path_messages=write_path_messages,
            active_state=active_state,
            owui_rag_meta=owui_rag_meta,
            skip_write_path=skip_write_path,
            generation_guard_meta=guard_meta,
        )
        return JSONResponse(content=normalized_json, status_code=response.status_code)
    except Exception as exc:
        logger.warning(
            "[OWUI_NATIVE_HYBRID] backend_non_json_or_parse_failed status=%s "
            "content_type=%s error=%s preview=%s",
            response.status_code,
            content_type,
            exc,
            response.text[:1000] if response.text else "",
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=media_type,
        )


async def _proxy_hybrid_continuation_final_response(
    payload: dict,
    chat_url: str,
    *,
    model_adapter,
    write_path_messages: list,
    active_state: dict,
    owui_rag_meta: dict,
    skip_write_path: bool,
    timeout_seconds: float = 1200.0,
) -> Response:
    """Validate a tool continuation answer before exposing it or writing memory."""
    client_stream_requested = bool(payload.get("stream", False))
    continuation_thinking = _env_bool(
        "KVEN2_TOOL_CONTINUATION_ENABLE_THINKING",
        True,
    )

    base_payload = _apply_final_answer_safeguards(
        payload,
        route_label=(
            "hybrid_continuation_live"
            if client_stream_requested
            else "hybrid_continuation_final"
        ),
        enable_thinking=(
            continuation_thinking
            if client_stream_requested
            else False
        ),
    )

    if client_stream_requested:
        base_payload["stream"] = True

        logger.info(
            "[OWUI_NATIVE_GUARD] continuation_live_stream=True "
            "thinking=%s tools_removed=True",
            continuation_thinking,
        )

        return _stream_main_chat_response(
            base_payload,
            chat_url,
            model_adapter=model_adapter,
            write_path_messages=write_path_messages,
            active_state=active_state,
            skip_write_path=skip_write_path,
            timeout_seconds=timeout_seconds,
            continuation_mode=True,
            owui_rag_meta=owui_rag_meta,
        )

    base_payload["stream"] = False

    last_base = {
        "model": payload.get("model") or settings.MAIN_MODEL,
    }

    for attempt in range(2):
        attempt_payload = dict(base_payload)
        attempt_messages = [
            dict(message) if isinstance(message, dict) else message
            for message in base_payload.get("messages", [])
        ]

        if attempt == 1:
            attempt_messages.append(
                {
                    "role": "user",
                    "content": (
                        "CORRECTION: The previous draft contained a tool call "
                        "or service markup and was rejected. Do not use tools "
                        "and do not mention this correction. Return only the "
                        "final answer based on the supplied tool results."
                    ),
                }
            )

        attempt_payload["messages"] = attempt_messages

        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds
            ) as client:
                response = await client.post(
                    chat_url,
                    json=attempt_payload,
                )

            if response.status_code >= 400:
                logger.warning(
                    "[OWUI_NATIVE_GUARD] continuation_http_error "
                    "attempt=%s status=%s preview=%s",
                    attempt + 1,
                    response.status_code,
                    response.text[:1000],
                )
                continue

            response_json = response.json()

            raw_choice = (
                response_json.get("choices") or [{}]
            )[0]
            raw_message = raw_choice.get("message") or {}

            raw_content = str(
                raw_message.get("content") or ""
            )
            reasoning = str(
                raw_message.get("reasoning_content") or ""
            )
            has_tool_calls = bool(
                raw_message.get("tool_calls")
            )

            normalized_json = (
                _normalize_native_tool_response_for_owui(
                    response_json
                )
            )
            last_base = normalized_json

            choice0 = (
                normalized_json.get("choices") or [{}]
            )[0]

            clean_answer = _clean_direct_native_answer(
                raw_content
            )
            clean_answer, guard_meta = (
                _sanitize_repetitive_text(clean_answer)
            )

            failed_attempt = (
                has_tool_calls
                or _looks_like_failed_native_tool_attempt(
                    raw_content,
                    reasoning,
                )
            )

            logger.info(
                "[OWUI_NATIVE_GUARD] continuation_attempt=%s "
                "status=%s raw_len=%s clean_len=%s "
                "has_tool_calls=%s failed_attempt=%s",
                attempt + 1,
                response.status_code,
                len(raw_content),
                len(clean_answer),
                has_tool_calls,
                failed_attempt,
            )

            if clean_answer and not failed_attempt:
                completion = _completion_with_direct_answer(
                    normalized_json,
                    clean_answer,
                    choice0.get("finish_reason") or "stop",
                )

                _schedule_hybrid_write_path(
                    assistant_reply=clean_answer,
                    write_path_messages=write_path_messages,
                    active_state=active_state,
                    owui_rag_meta=owui_rag_meta,
                    skip_write_path=skip_write_path,
                    generation_guard_meta=guard_meta,
                )

                return _completion_response_for_client(
                    completion,
                    client_stream_requested,
                )

            logger.warning(
                "[OWUI_NATIVE_GUARD] "
                "continuation_output_rejected "
                "attempt=%s preview=%s",
                attempt + 1,
                raw_content[:500],
            )

        except Exception as exc:
            logger.error(
                "[OWUI_NATIVE_GUARD] "
                "continuation_attempt_failed "
                "attempt=%s error=%s",
                attempt + 1,
                exc,
                exc_info=True,
            )

    fallback = (
        "Не удалось сформировать итоговый ответ по результатам "
        "инструмента без служебной разметки. Повторите запрос."
    )

    logger.error(
        "[OWUI_NATIVE_GUARD] continuation_safe_fallback "
        "tool_markup_not_exposed=True"
    )

    completion = _completion_with_direct_answer(
        last_base,
        fallback,
        "stop",
    )

    return _completion_response_for_client(
        completion,
        client_stream_requested,
    )



async def _proxy_hybrid_no_tool_final_response(
    payload: dict,
    chat_url: str,
    *,
    write_path_messages: list,
    active_state: dict,
    owui_rag_meta: dict,
    skip_write_path: bool,
    timeout_seconds: float = 1200.0,
) -> Response:
    """Produce the real answer after the bounded decision says no tool is needed."""
    final_payload = dict(payload)
    final_payload.pop("tools", None)
    final_payload.pop("tool_choice", None)
    final_payload.pop("parallel_tool_calls", None)

    stream_requested = bool(final_payload.get("stream", False))
    rag_detected = bool(owui_rag_meta.get("detected"))

    if stream_requested and not rag_detected:
        thinking_enabled = _env_bool(
            "KVEN2_MAIN_ENABLE_THINKING",
            True,
        )

        final_payload = _apply_final_answer_safeguards(
            final_payload,
            route_label="hybrid_no_tool_live",
            enable_thinking=thinking_enabled,
        )

        logger.info(
            "[OWUI_NATIVE_GUARD] no_tool_final "
            "live_stream=True thinking=%s tools_removed=True",
            thinking_enabled,
        )

        return _stream_main_chat_response(
            final_payload,
            chat_url,
            write_path_messages=write_path_messages,
            active_state=active_state,
            skip_write_path=skip_write_path,
            timeout_seconds=timeout_seconds,
        )

    logger.info(
        "[OWUI_NATIVE_GUARD] no_tool_final "
        "live_stream=False rag=%s stream_requested=%s tools_removed=True",
        rag_detected,
        stream_requested,
    )

    return await _proxy_hybrid_final_response(
        final_payload,
        chat_url,
        write_path_messages=write_path_messages,
        active_state=active_state,
        owui_rag_meta=owui_rag_meta,
        skip_write_path=skip_write_path,
        timeout_seconds=timeout_seconds,
    )


def _start_speculative_no_tool_stream(
    payload: dict,
    chat_url: str,
    *,
    write_path_messages: list,
    active_state: dict,
    owui_rag_meta: dict,
    skip_write_path: bool,
    timeout_seconds: float,
    max_chunks: int,
) -> SpeculativeStream:
    """
    Start the final main-model stream before the planner has completed.

    Output remains buffered, and completion side effects remain blocked,
    until the planner releases the completion gate after NO_TOOL.
    """
    final_payload = dict(payload)
    final_payload.pop("tools", None)
    final_payload.pop("tool_choice", None)
    final_payload.pop("parallel_tool_calls", None)

    thinking_enabled = _env_bool(
        "KVEN2_MAIN_ENABLE_THINKING",
        True,
    )

    final_payload = _apply_final_answer_safeguards(
        final_payload,
        route_label="hybrid_no_tool_speculative",
        enable_thinking=thinking_enabled,
    )
    final_payload["stream"] = True

    release_gate = asyncio.Event()

    source_response = _stream_main_chat_response(
        final_payload,
        chat_url,
        write_path_messages=write_path_messages,
        active_state=active_state,
        skip_write_path=skip_write_path,
        timeout_seconds=timeout_seconds,
        owui_rag_meta=owui_rag_meta,
        completion_gate=release_gate,
    )

    speculative = SpeculativeStream(
        source_response,
        release_gate=release_gate,
        max_chunks=max_chunks,
    )

    logger.info(
        "[SPECULATIVE_PLANNER] main_stream_started=True "
        "queue_max_chunks=%s thinking=%s",
        max_chunks,
        thinking_enabled,
    )

    return speculative


async def _proxy_hybrid_native_openai_tool_protocol(
    payload: dict,
    chat_url: str,
    *,
    model_adapter,
    write_path_messages: list,
    active_state: dict,
    owui_rag_meta: dict,
    skip_write_path: bool,
    timeout_seconds: float = 1200.0,
) -> Response:
    """
    Bounded native-tool gateway.

    First user pass is buffered and capped, so reasoning cannot run for thousands
    of tokens before a tool call. Valid native calls are returned to OWUI. XML-like
    pseudo calls are repaired once. Continuation passes after role=tool are forced
    to a final answer with tool_choice=none.
    """
    messages = payload.get("messages", [])
    stream_requested = bool(payload.get("stream", False))

    # Tool results already exist: remove the tool catalogue and native protocol
    # history, then validate the final answer before exposing it to the client.
    if _messages_include_native_tool_continuation(messages):
        final_payload = dict(payload)
        final_payload.pop("tools", None)
        final_payload.pop("tool_choice", None)
        final_payload["messages"] = _build_clean_tool_continuation_messages(
            messages
        )

        logger.info(
            "[OWUI_NATIVE_GUARD] continuation_final_answer "
            "tools_removed=True protocol_history_flattened=True"
        )

        return await _proxy_hybrid_continuation_final_response(
            final_payload,
            chat_url,
            model_adapter=model_adapter,
            write_path_messages=write_path_messages,
            active_state=active_state,
            owui_rag_meta=owui_rag_meta,
            skip_write_path=skip_write_path,
            timeout_seconds=timeout_seconds,
        )

    allowed_names = _allowed_native_tool_names(payload)
    if not allowed_names:
        logger.info(
            "[OWUI_NATIVE_GUARD] no_tools_after_policy "
            "route_to_final_answer=True"
        )
        return await _proxy_hybrid_no_tool_final_response(
            payload,
            chat_url,
            write_path_messages=write_path_messages,
            active_state=active_state,
            owui_rag_meta=owui_rag_meta,
            skip_write_path=skip_write_path,
            timeout_seconds=timeout_seconds,
        )

    planner_enabled = _env_bool(
        "KVEN2_PLANNER_TOOL_ROUTING_ENABLED",
        False,
    )
    tool_choice = payload.get("tool_choice")
    tool_choice_mode = (
        str(tool_choice).strip().lower()
        if isinstance(tool_choice, str)
        else ""
    )

    if planner_enabled and tool_choice_mode == "none":
        logger.info(
            "[PLANNER_ROUTER] bypass reason=tool_choice_none "
            "route_to_final_answer=True"
        )
        return await _proxy_hybrid_no_tool_final_response(
            payload,
            chat_url,
            write_path_messages=write_path_messages,
            active_state=active_state,
            owui_rag_meta=owui_rag_meta,
            skip_write_path=skip_write_path,
            timeout_seconds=timeout_seconds,
        )

    speculative_stream = None

    if planner_enabled:
        planner_timeout = _env_float(
            "KVEN2_PLANNER_TOOL_ROUTING_TIMEOUT",
            20.0,
            3.0,
            60.0,
        )
        explicit_tool_name = _explicit_native_tool_choice_name(
            tool_choice
        )

        speculative_enabled = _env_bool(
            "KVEN2_SPECULATIVE_PLANNER_ENABLED",
            False,
        )
        speculative_eligible = (
            speculative_enabled
            and stream_requested
            and not bool(owui_rag_meta.get("detected"))
            and not explicit_tool_name
            and tool_choice_mode in ("", "auto")
        )

        if speculative_enabled and not speculative_eligible:
            if not stream_requested:
                bypass_reason = "non_stream_request"
            elif owui_rag_meta.get("detected"):
                bypass_reason = "rag_detected"
            elif explicit_tool_name:
                bypass_reason = "explicit_tool_choice"
            else:
                bypass_reason = (
                    f"tool_choice_{tool_choice_mode or 'unknown'}"
                )

            logger.info(
                "[SPECULATIVE_PLANNER] bypass=True reason=%s",
                bypass_reason,
            )

        if speculative_eligible:
            queue_max_chunks = _env_int(
                "KVEN2_SPECULATIVE_QUEUE_MAX_CHUNKS",
                32,
                1,
                512,
            )

            speculative_stream = (
                _start_speculative_no_tool_stream(
                    payload,
                    chat_url,
                    write_path_messages=write_path_messages,
                    active_state=active_state,
                    owui_rag_meta=owui_rag_meta,
                    skip_write_path=skip_write_path,
                    timeout_seconds=timeout_seconds,
                    max_chunks=queue_max_chunks,
                )
            )
            await speculative_stream.wait_started()

        logger.info(
            "[PLANNER_ROUTER] routing_start tools_count=%s "
            "explicit_tool=%s required=%s timeout=%s "
            "speculative=%s",
            len(payload.get("tools", []))
            if isinstance(payload.get("tools"), list)
            else 0,
            explicit_tool_name or "",
            tool_choice_mode == "required",
            planner_timeout,
            speculative_stream is not None,
        )

        try:
            planner_result = await planner_route_tool_request(
                messages,
                payload.get("tools", []),
                allowed_names=allowed_names,
                explicit_tool_name=explicit_tool_name or None,
                timeout_seconds=planner_timeout,
            )
        except asyncio.CancelledError:
            if speculative_stream is not None:
                await speculative_stream.cancel(
                    reason="request_cancelled"
                )
            raise
        except Exception:
            if speculative_stream is not None:
                await speculative_stream.cancel(
                    reason="planner_exception"
                )
            raise
        planner_decision = str(
            planner_result.get("decision") or ""
        ).strip().upper()

        logger.info(
            "[PLANNER_ROUTER] routing_result decision=%s meta=%s "
            "error=%s",
            planner_decision,
            _json_preview(
                planner_result.get("meta") or {},
                limit=1600,
            ),
            str(planner_result.get("error") or "")[:500],
        )

        if planner_decision == "TOOL":
            planned_call = planner_result.get("tool_call")
            calls, call_meta = _validate_and_limit_native_tool_calls(
                [planned_call] if planned_call else [],
                allowed_names,
            )

            if calls:
                if speculative_stream is not None:
                    await speculative_stream.cancel(
                        reason="tool_selected"
                    )
                    speculative_stream = None

                logger.info(
                    "[PLANNER_ROUTER] native_tool_call_ready "
                    "name=%s validation=%s",
                    calls[0]["function"]["name"],
                    _json_preview(call_meta, limit=1000),
                )
                completion = _completion_with_tool_calls(
                    {
                        "model": (
                            payload.get("model")
                            or settings.MAIN_MODEL
                        )
                    },
                    calls,
                )
                return _completion_response_for_client(
                    completion,
                    stream_requested,
                )

            logger.warning(
                "[PLANNER_ROUTER] invalid_tool_call "
                "fallback=legacy_main validation=%s",
                _json_preview(call_meta, limit=1000),
            )

        elif (
            planner_decision == "NO_TOOL"
            and tool_choice_mode != "required"
        ):
            logger.info(
                "[PLANNER_ROUTER] no_tool "
                "route_to_final_answer=True speculative=%s",
                speculative_stream is not None,
            )

            if speculative_stream is not None:
                logger.info(
                    "[SPECULATIVE_PLANNER] "
                    "no_tool_release=True"
                )
                return speculative_stream.release_response()

            return await _proxy_hybrid_no_tool_final_response(
                payload,
                chat_url,
                write_path_messages=write_path_messages,
                active_state=active_state,
                owui_rag_meta=owui_rag_meta,
                skip_write_path=skip_write_path,
                timeout_seconds=timeout_seconds,
            )

        elif (
            planner_decision == "NO_TOOL"
            and tool_choice_mode == "required"
        ):
            logger.warning(
                "[PLANNER_ROUTER] no_tool_rejected "
                "reason=tool_choice_required fallback=legacy_main"
            )

        else:
            logger.warning(
                "[PLANNER_ROUTER] unavailable "
                "fallback=legacy_main decision=%s",
                planner_decision,
            )

    if speculative_stream is not None:
        await speculative_stream.cancel(
            reason="planner_fallback"
        )
        speculative_stream = None

    decision_timeout = float(
        _env_int(
            "KVEN2_NATIVE_TOOL_DECISION_TIMEOUT",
            120,
            30,
            300,
        )
    )

    # This pass only classifies TOOL_CALL versus NO_TOOL. A large budget lets
    # the model answer the whole question before the real streamed generation.
    decision_tokens_default = (
        192 if owui_rag_meta.get("detected") else 128
    )
    decision_tokens = _env_int(
        "KVEN2_NATIVE_TOOL_DECISION_MAX_TOKENS",
        decision_tokens_default,
        64,
        512,
    )

    last_base = {}
    for attempt in range(2):
        correction = attempt == 1
        response_json, error = await _post_native_decision_json(
            payload,
            chat_url,
            correction=correction,
            timeout_seconds=decision_timeout,
            max_tokens=decision_tokens,
        )
        if response_json is None:
            logger.info(
                "[OWUI_NATIVE_GUARD] decision_unavailable attempt=%s error=%s",
                attempt + 1,
                error,
            )
            break

        last_base = response_json
        choice0 = (response_json.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or "")
        finish_reason = str(choice0.get("finish_reason") or "")

        calls, call_meta = _validate_and_limit_native_tool_calls(
            message.get("tool_calls"), allowed_names
        )
        logger.info(
            "[OWUI_NATIVE_GUARD] decision_result attempt=%s finish_reason=%s "
            "content_len=%s reasoning_len=%s call_meta=%s",
            attempt + 1,
            finish_reason,
            len(content),
            len(reasoning),
            _json_preview(call_meta, limit=1000),
        )

        if calls:
            if call_meta.get("mutations_dropped") or call_meta.get("duplicates") or call_meta.get("invalid"):
                logger.warning("[OWUI_NATIVE_GUARD] tool_calls_filtered meta=%s", call_meta)
            completion = _completion_with_tool_calls(response_json, calls)
            logger.info(
                "[OWUI_NATIVE_GUARD] native_tool_calls_ready names=%s",
                [c["function"]["name"] for c in calls],
            )
            return _completion_response_for_client(completion, stream_requested)

        pseudo = _extract_pseudo_native_tool_calls(
            reasoning + "\n" + content, allowed_names
        )
        pseudo_calls, pseudo_meta = _validate_and_limit_native_tool_calls(pseudo, allowed_names)
        if pseudo_calls:
            logger.warning(
                "[OWUI_NATIVE_GUARD] pseudo_tool_call_repaired names=%s meta=%s",
                [c["function"]["name"] for c in pseudo_calls],
                pseudo_meta,
            )
            completion = _completion_with_tool_calls(response_json, pseudo_calls)
            return _completion_response_for_client(completion, stream_requested)

        clean_answer = _clean_direct_native_answer(content)
        clean_answer, decision_guard_meta = _sanitize_repetitive_text(clean_answer)
        failed_attempt = _looks_like_failed_native_tool_attempt(content, reasoning)

        if clean_answer and not failed_attempt and finish_reason != "length":
            no_tool_marker = clean_answer.strip() == "KVEN_NO_TOOL"
            logger.info(
                "[OWUI_NATIVE_GUARD] no_tool_decision "
                "marker=%s decision_content_len=%s "
                "route_to_full_final_answer=True",
                no_tool_marker,
                len(clean_answer),
            )
            return await _proxy_hybrid_no_tool_final_response(
                payload,
                chat_url,
                write_path_messages=write_path_messages,
                active_state=active_state,
                owui_rag_meta=owui_rag_meta,
                skip_write_path=skip_write_path,
                timeout_seconds=timeout_seconds,
            )

        if attempt == 0 and (failed_attempt or (not clean_answer and reasoning)):
            logger.warning(
                "[OWUI_NATIVE_GUARD] correction_retry reason=%s",
                "failed_tool_attempt" if failed_attempt else "reasoning_only",
            )
            continue

        break

    # Decision failure must never expose a blank response. Remove tools and run
    # the normal full-budget final-answer path without decision-control prompts.
    logger.warning(
        "[OWUI_NATIVE_GUARD] fallback_no_tool_answer "
        "reason=decision_exhausted route_to_full_final_answer=True"
    )
    return await _proxy_hybrid_no_tool_final_response(
        payload,
        chat_url,
        write_path_messages=write_path_messages,
        active_state=active_state,
        owui_rag_meta=owui_rag_meta,
        skip_write_path=skip_write_path,
        timeout_seconds=timeout_seconds,
    )


# -----------------------------------------------------------------------------
# OWUI internal request filtering
# -----------------------------------------------------------------------------
# OpenWebUI periodically sends internal prompts for chat titles, tags, summaries,
# metadata, etc. They often look like user messages starting with "### Task:".
# These requests must not receive Kven memory/system context and must not be
# written back to episodic/semantic memory.


def _content_to_text(content) -> str:
    """Best-effort conversion of OpenAI message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item.get("text")))
                elif "content" in item:
                    parts.append(str(item.get("content")))
                elif "text" in item:
                    parts.append(str(item.get("text")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _last_user_text(messages) -> str:
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ""
    return _content_to_text(user_messages[-1].get("content"))


def _merge_kven_system_context(messages: list, sys_block: str) -> tuple[list, dict]:
    """
    Return a llama.cpp/Qwen-safe message list with exactly one system message.

    OpenWebUI commonly sends its own leading system message (memory/tool runtime
    instructions). Prepending a second system message makes the Qwen Jinja chat
    template fail with "System message must be at the beginning". Merge every
    system message into one first message and preserve all non-system messages in
    their original order.
    """
    if not isinstance(messages, list):
        messages = []

    system_parts = []
    non_system_messages = []

    if isinstance(sys_block, str) and sys_block.strip():
        system_parts.append(sys_block.strip())

    for msg in messages:
        if not isinstance(msg, dict):
            non_system_messages.append(msg)
            continue

        if msg.get("role") == "system":
            text = _content_to_text(msg.get("content")).strip()
            if text:
                system_parts.append(text)
            continue

        non_system_messages.append(msg)

    merged_system = "\n\n--- OWUI SYSTEM CONTEXT ---\n\n".join(system_parts)
    enriched = []
    if merged_system:
        enriched.append({"role": "system", "content": merged_system})
    enriched.extend(non_system_messages)

    return enriched, {
        "incoming_system_messages": sum(
            1 for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "system"
        ),
        "merged_system_parts": len(system_parts),
        "merged_system_chars": len(merged_system),
        "output_messages": len(enriched),
    }


def detect_internal_owui_request(messages) -> tuple[bool, str]:
    """
    Deterministic filter for OpenWebUI service prompts.

    Returns:
        (True, reason)  -> bypass Kven memory/context/write_path
        (False, reason) -> normal user request
    """
    text = _last_user_text(messages)
    stripped = text.strip()
    lowered = stripped.lower()

    if not stripped:
        return False, "no_user_text"

    # Most OWUI metadata prompts use this exact heading.
    if stripped.startswith("### Task:") or lowered.startswith("### task:"):
        return True, "task_marker"

    # Additional conservative markers for title/tag/summary service prompts.
    internal_markers = [
        "generate a concise",
        "generate a title",
        "generate a short title",
        "create a concise title",
        "create a short title",
        "provide a concise title",
        "chat title",
        "conversation title",
        "title for this chat",
        "title this chat",
        "summarize the conversation",
        "summarise the conversation",
        "conversation summary",
        "chat summary",
        "generate tags",
        "tag the conversation",
        "categorize this conversation",
        "categorise this conversation",
    ]

    # Only apply fuzzy markers to service-prompt-looking texts. This avoids
    # accidentally filtering a normal user question about, for example, "chat title".
    service_prompt_shape = (
        lowered.startswith("task:")
        or "### task" in lowered[:200]
        or "you are given a chat" in lowered[:300]
        or "based on the conversation" in lowered[:300]
        or "the conversation above" in lowered[:300]
    )

    if service_prompt_shape and any(marker in lowered for marker in internal_markers):
        return True, "metadata_marker"

    return False, "no_internal_signature"



def _stream_main_chat_response(
    payload: dict,
    chat_url: str,
    *,
    model_adapter=None,
    write_path_messages: list,
    active_state: dict,
    skip_write_path: bool,
    timeout_seconds: float = 1200.0,
    continuation_mode: bool = False,
    owui_rag_meta: dict | None = None,
    completion_gate: asyncio.Event | None = None,
) -> StreamingResponse:
    """
    Stream reasoning and content live; retain final content for memory.

    In continuation mode, renewed tool calls and tool markup are blocked
    before they can reach the client.
    """

    async def generate():
        effective_model_adapter = (
            model_adapter
            or resolve_model_adapter(
                backend_model=settings.MAIN_MODEL,
                backend_url=BASE_BACKEND,
            )
        )
        stream_phase = (
            "continuation"
            if continuation_mode
            else "main"
        )
        thinking_enabled = bool(
            (
                payload.get("chat_template_kwargs")
                or {}
            ).get("enable_thinking")
        )
        content_normalizer = (
            effective_model_adapter
            .create_stream_content_normalizer(
                phase=stream_phase,
                thinking_enabled=thinking_enabled,
            )
        )
        content_normalizer_finished = False

        logger.info(
            "[MODEL_ADAPTER] stream_normalizer "
            "adapter=%s phase=%s thinking=%s",
            effective_model_adapter.adapter_id,
            stream_phase,
            thinking_enabled,
        )

        assistant_reply = ""
        reasoning_text = ""
        content_streamed = False
        continuation_emitted_chars = 0
        continuation_reasoning_marker_logged = False
        finish_reason = "stop"
        completion_base = {
            "model": payload.get("model") or settings.MAIN_MODEL,
        }
        guard_meta = {"detected": False}
        last_reasoning_check = 0
        last_content_check = 0
        check_interval = _env_int(
            "KVEN2_REPETITION_CHECK_INTERVAL",
            200,
            50,
            2000,
        )

        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds
            ) as client:
                async with client.stream(
                    "POST",
                    chat_url,
                    json=payload,
                ) as response:
                    content_type = response.headers.get(
                        "content-type",
                        "",
                    )
                    logger.info(
                        "[LIVE_STREAM] backend_status=%s "
                        "content_type=%s",
                        response.status_code,
                        content_type,
                    )
                    response.raise_for_status()

                    if content_type.startswith(
                        "text/event-stream"
                    ):
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue

                            raw = line[6:].strip()
                            if not raw:
                                continue
                            if raw == "[DONE]":
                                break

                            try:
                                obj = json.loads(raw)
                            except Exception:
                                logger.debug(
                                    "[LIVE_STREAM] "
                                    "ignored_non_json_data "
                                    "preview=%s",
                                    raw[:200],
                                )
                                continue

                            completion_base = obj
                            choice0 = (
                                obj.get("choices") or [{}]
                            )[0]
                            delta = choice0.get("delta") or {}

                            reasoning_piece = delta.get(
                                "reasoning_content"
                            )
                            content_piece = delta.get("content")

                            if (
                                continuation_mode
                                and (
                                    delta.get("tool_calls")
                                    or delta.get("function_call")
                                    or choice0.get("finish_reason")
                                    == "tool_calls"
                                )
                            ):
                                guard_meta = {
                                    "detected": True,
                                    "reason": (
                                        "structured_tool_call"
                                    ),
                                }
                                logger.warning(
                                    "[LIVE_STREAM] "
                                    "continuation_structured_tool_call_"
                                    "blocked=True"
                                )
                                break

                            if (
                                isinstance(reasoning_piece, str)
                                and reasoning_piece
                            ):
                                reasoning_text += reasoning_piece

                                if (
                                    continuation_mode
                                    and not continuation_reasoning_marker_logged
                                    and _looks_like_failed_native_tool_attempt(
                                        "",
                                        reasoning_text,
                                    )
                                ):
                                    continuation_reasoning_marker_logged = True
                                    logger.warning(
                                        "[LIVE_STREAM] "
                                        "continuation_reasoning_mentions_"
                                        "tool=True content_not_blocked=True"
                                    )

                                if (
                                    len(reasoning_text)
                                    - last_reasoning_check
                                    >= check_interval
                                ):
                                    last_reasoning_check = len(
                                        reasoning_text
                                    )
                                    _, reasoning_guard = (
                                        _sanitize_repetitive_text(
                                            reasoning_text
                                        )
                                    )
                                    if reasoning_guard.get(
                                        "detected"
                                    ):
                                        guard_meta = dict(
                                            reasoning_guard
                                        )
                                        guard_meta["reason"] = (
                                            "reasoning_"
                                            + str(
                                                reasoning_guard.get(
                                                    "reason"
                                                )
                                                or "loop"
                                            )
                                        )
                                        logger.warning(
                                            "[LIVE_STREAM] "
                                            "reasoning_aborted "
                                            "meta=%s",
                                            _json_preview(
                                                guard_meta,
                                                limit=1000,
                                            ),
                                        )
                                        break

                                safe_obj = dict(obj)
                                safe_choice = dict(choice0)
                                safe_delta = dict(delta)
                                safe_delta.pop("content", None)
                                safe_delta[
                                    "reasoning_content"
                                ] = reasoning_piece
                                safe_choice["delta"] = safe_delta
                                safe_choice[
                                    "finish_reason"
                                ] = None
                                safe_obj["choices"] = [
                                    safe_choice
                                ]

                                yield (
                                    "data: "
                                    + json.dumps(
                                        safe_obj,
                                        ensure_ascii=False,
                                    )
                                    + "\n\n"
                                )

                            elif delta.get("role"):
                                safe_obj = dict(obj)
                                safe_choice = dict(choice0)
                                safe_delta = dict(delta)
                                safe_delta.pop("content", None)
                                safe_choice["delta"] = safe_delta
                                safe_obj["choices"] = [
                                    safe_choice
                                ]

                                yield (
                                    "data: "
                                    + json.dumps(
                                        safe_obj,
                                        ensure_ascii=False,
                                    )
                                    + "\n\n"
                                )

                            if (
                                isinstance(content_piece, str)
                                and content_piece
                            ):
                                content_piece = (
                                    content_normalizer.feed(
                                        content_piece
                                    )
                                )

                            if (
                                isinstance(content_piece, str)
                                and content_piece
                            ):
                                candidate_reply = (
                                    assistant_reply
                                    + content_piece
                                )

                                if continuation_mode:
                                    (
                                        marker_position,
                                        marker,
                                    ) = _find_continuation_content_marker(
                                        candidate_reply
                                    )

                                    if marker_position is not None:
                                        assistant_reply = (
                                            candidate_reply[
                                                :marker_position
                                            ].rstrip()
                                        )
                                        guard_meta = {
                                            "detected": True,
                                            "reason": (
                                                "content_tool_markup"
                                            ),
                                            "marker": marker,
                                        }
                                        logger.warning(
                                            "[LIVE_STREAM] "
                                            "continuation_content_marker_"
                                            "blocked marker=%s",
                                            marker,
                                        )
                                        break

                                if (
                                    len(candidate_reply)
                                    - last_content_check
                                    >= check_interval
                                ):
                                    last_content_check = len(
                                        candidate_reply
                                    )
                                    (
                                        cleaned_candidate,
                                        live_content_guard,
                                    ) = _sanitize_repetitive_text(
                                        candidate_reply
                                    )

                                    if live_content_guard.get(
                                        "detected"
                                    ):
                                        assistant_reply = (
                                            cleaned_candidate
                                        )
                                        guard_meta = dict(
                                            live_content_guard
                                        )
                                        guard_meta["reason"] = (
                                            "content_"
                                            + str(
                                                live_content_guard.get(
                                                    "reason"
                                                )
                                                or "loop"
                                            )
                                        )
                                        logger.warning(
                                            "[LIVE_STREAM] "
                                            "content_aborted "
                                            "meta=%s",
                                            _json_preview(
                                                guard_meta,
                                                limit=1000,
                                            ),
                                        )
                                        break

                                assistant_reply = candidate_reply
                                emit_piece = content_piece

                                if continuation_mode:
                                    safe_emit_end = (
                                        _continuation_safe_emit_end(
                                            assistant_reply
                                        )
                                    )

                                    if (
                                        safe_emit_end
                                        > continuation_emitted_chars
                                    ):
                                        emit_piece = assistant_reply[
                                            continuation_emitted_chars:
                                            safe_emit_end
                                        ]
                                        continuation_emitted_chars = (
                                            safe_emit_end
                                        )
                                    else:
                                        emit_piece = ""

                                if emit_piece:
                                    content_streamed = True

                                    safe_obj = dict(obj)
                                    safe_choice = dict(choice0)
                                    safe_delta = dict(delta)
                                    safe_delta.pop(
                                        "reasoning_content",
                                        None,
                                    )
                                    safe_delta.pop(
                                        "tool_calls",
                                        None,
                                    )
                                    safe_delta.pop(
                                        "function_call",
                                        None,
                                    )
                                    safe_delta["content"] = (
                                        emit_piece
                                    )
                                    safe_choice["delta"] = (
                                        safe_delta
                                    )
                                    safe_choice[
                                        "finish_reason"
                                    ] = None
                                    safe_obj["choices"] = [
                                        safe_choice
                                    ]

                                    yield (
                                        "data: "
                                        + json.dumps(
                                            safe_obj,
                                            ensure_ascii=False,
                                        )
                                        + "\n\n"
                                    )

                            if choice0.get("finish_reason"):
                                finish_reason = str(
                                    choice0.get(
                                        "finish_reason"
                                    )
                                )

                    else:
                        response_json = json.loads(
                            await response.aread()
                        )
                        completion_base = response_json

                        choice0 = (
                            response_json.get("choices")
                            or [{}]
                        )[0]
                        message = (
                            choice0.get("message") or {}
                        )

                        reasoning = str(
                            message.get(
                                "reasoning_content"
                            )
                            or ""
                        )

                        if reasoning:
                            reasoning_event = {
                                "id": (
                                    response_json.get("id")
                                    or "chatcmpl-kven-live"
                                ),
                                "object": (
                                    "chat.completion.chunk"
                                ),
                                "created": (
                                    response_json.get(
                                        "created"
                                    )
                                    or 0
                                ),
                                "model": (
                                    response_json.get(
                                        "model"
                                    )
                                    or payload.get("model")
                                    or settings.MAIN_MODEL
                                ),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "reasoning_content": (
                                                reasoning
                                            ),
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }

                            yield (
                                "data: "
                                + json.dumps(
                                    reasoning_event,
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )

                        assistant_reply = (
                            content_normalizer.feed(
                                str(
                                    message.get("content")
                                    or ""
                                )
                            )
                            + content_normalizer.finish()
                        )
                        content_normalizer_finished = True

                        finish_reason = str(
                            choice0.get("finish_reason")
                            or "stop"
                        )

                        if (
                            continuation_mode
                            and (
                                message.get("tool_calls")
                                or message.get("function_call")
                                or finish_reason == "tool_calls"
                            )
                        ):
                            assistant_reply = ""
                            guard_meta = {
                                "detected": True,
                                "reason": (
                                    "structured_tool_call"
                                ),
                            }
                            logger.warning(
                                "[LIVE_STREAM] "
                                "continuation_json_tool_call_"
                                "blocked=True"
                            )

        except asyncio.CancelledError:
            logger.info(
                "[LIVE_STREAM] client_disconnected "
                "backend_stream_cancelled=True"
            )
            raise

        except Exception as exc:
            logger.error(
                "[LIVE_STREAM] stream_failed error=%s",
                exc,
                exc_info=True,
            )
            assistant_reply = (
                "Ошибка при потоковой генерации. "
                "Повторите запрос."
            )
            guard_meta = {
                "detected": True,
                "reason": "stream_exception",
            }

        if (
            not content_normalizer_finished
            and not guard_meta.get("detected")
        ):
            normalized_tail = content_normalizer.finish()
            content_normalizer_finished = True

            if normalized_tail:
                assistant_reply += normalized_tail

        client_reply = assistant_reply

        if (
            continuation_mode
            and not guard_meta.get("detected")
        ):
            (
                marker_position,
                marker,
            ) = _find_continuation_content_marker(
                client_reply
            )

            if marker_position is not None:
                client_reply = client_reply[
                    :marker_position
                ].rstrip()
                guard_meta = {
                    "detected": True,
                    "reason": "content_tool_markup",
                    "marker": marker,
                }

        assistant_reply = _clean_direct_native_answer(
            client_reply
        )
        assistant_reply, content_guard = (
            _sanitize_repetitive_text(
                assistant_reply
            )
        )

        if content_guard.get("detected"):
            guard_meta = content_guard
            client_reply = assistant_reply

        guard_reason = str(
            guard_meta.get("reason") or ""
        )
        continuation_repetition_guard = (
            "repeated_" in guard_reason
        )
        continuation_failure_sent = False

        if continuation_mode:
            if (
                guard_meta.get("detected")
                and not continuation_repetition_guard
            ):
                fallback = (
                    "Не удалось безопасно завершить ответ "
                    "по результатам инструмента. "
                    "Повторите запрос."
                )

                if content_streamed:
                    fallback = "\n\n" + fallback

                fallback_event = {
                    "id": (
                        completion_base.get("id")
                        or "chatcmpl-kven-live"
                    ),
                    "object": "chat.completion.chunk",
                    "created": (
                        completion_base.get("created")
                        or 0
                    ),
                    "model": (
                        completion_base.get("model")
                        or payload.get("model")
                        or settings.MAIN_MODEL
                    ),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": fallback,
                            },
                            "finish_reason": None,
                        }
                    ],
                }

                yield (
                    "data: "
                    + json.dumps(
                        fallback_event,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

                content_streamed = True
                continuation_failure_sent = True
                assistant_reply = ""

            elif (
                not guard_meta.get("detected")
                and continuation_emitted_chars
                < len(client_reply)
            ):
                remaining = client_reply[
                    continuation_emitted_chars:
                ]

                if remaining:
                    remaining_event = {
                        "id": (
                            completion_base.get("id")
                            or "chatcmpl-kven-live"
                        ),
                        "object": (
                            "chat.completion.chunk"
                        ),
                        "created": (
                            completion_base.get("created")
                            or 0
                        ),
                        "model": (
                            completion_base.get("model")
                            or payload.get("model")
                            or settings.MAIN_MODEL
                        ),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": remaining,
                                },
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield (
                        "data: "
                        + json.dumps(
                            remaining_event,
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                    content_streamed = True
                    continuation_emitted_chars = len(
                        client_reply
                    )

        if (
            completion_gate is not None
            and not completion_gate.is_set()
        ):
            logger.info(
                "[LIVE_STREAM] completion_gate_waiting=True"
            )

            try:
                await completion_gate.wait()
            except asyncio.CancelledError:
                logger.info(
                    "[LIVE_STREAM] completion_gate_cancelled=True"
                )
                raise

            logger.info(
                "[LIVE_STREAM] completion_gate_released=True"
            )

        if continuation_failure_sent:
            pass
        elif (
            guard_meta.get("detected")
            and not assistant_reply
        ):
            assistant_reply = (
                "Генерация остановлена из-за "
                "повторяющегося цикла. "
                "Повторите запрос."
            )
        elif not assistant_reply:
            assistant_reply = (
                "Backend завершил поток без "
                "итогового ответа."
            )
            guard_meta = {
                "detected": True,
                "reason": "empty_final_content",
            }

        logger.info(
            "[LIVE_STREAM] completed "
            "reasoning_chars=%s content_chars=%s "
            "finish_reason=%s guard_detected=%s "
            "content_streamed_live=%s",
            len(reasoning_text),
            len(assistant_reply),
            finish_reason,
            bool(guard_meta.get("detected")),
            content_streamed,
        )

        if continuation_mode:
            _schedule_hybrid_write_path(
                assistant_reply=assistant_reply,
                write_path_messages=write_path_messages,
                active_state=active_state,
                owui_rag_meta=owui_rag_meta or {},
                skip_write_path=skip_write_path,
                generation_guard_meta=guard_meta,
            )
        elif guard_meta.get("detected"):
            logger.warning(
                "[ROUTE] Skipping [WRITE_PATH] "
                "reason=live_stream_guard meta=%s",
                _json_preview(
                    guard_meta,
                    limit=1000,
                ),
            )
        elif (
            not skip_write_path
            and len(assistant_reply.strip()) > 10
        ):
            logger.info(
                "[ROUTE] ✅ Triggering background "
                "task [WRITE_PATH]"
            )
            asyncio.create_task(
                process_episodic(
                    write_path_messages,
                    assistant_reply,
                    active_state,
                )
            )

        final_finish_reason = (
            "stop"
            if guard_meta.get("detected")
            else (finish_reason or "stop")
        )

        event_id = (
            completion_base.get("id")
            or "chatcmpl-kven-live"
        )
        event_created = (
            completion_base.get("created")
            or 0
        )
        event_model = (
            completion_base.get("model")
            or payload.get("model")
            or settings.MAIN_MODEL
        )

        # A non-SSE backend has not emitted content yet. Convert its complete
        # answer into one normal content delta before the terminal event.
        if not content_streamed:
            content_event = {
                "id": event_id,
                "object": "chat.completion.chunk",
                "created": event_created,
                "model": event_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": assistant_reply,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(
                    content_event,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        terminal_event = {
            "id": event_id,
            "object": "chat.completion.chunk",
            "created": event_created,
            "model": event_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": final_finish_reason,
                }
            ],
        }

        if completion_base.get("usage") is not None:
            terminal_event["usage"] = (
                completion_base.get("usage")
            )

        if (
            completion_base.get("system_fingerprint")
            is not None
        ):
            terminal_event["system_fingerprint"] = (
                completion_base.get(
                    "system_fingerprint"
                )
            )

        yield (
            "data: "
            + json.dumps(
                terminal_event,
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _forward_to_backend_and_collect(
    payload: dict,
    chat_url: str,
    *,
    timeout_seconds: float = 1200.0,
    route_label: str = "main",
) -> tuple[list[str], dict]:
    """
    Preserve buffered SSE behavior while terminating repeated final generations.

    The backend stream is closed as soon as a long repeated token sequence is
    observed. The repeated tail is removed and replaced with a compact valid SSE
    completion. This prevents both UI flooding and wasted generation time.
    """
    backend_chunks = []
    accumulated_text = ""
    guard_meta = {
        "detected": False,
        "reason": "",
        "cutoff": 0,
        "original_chars": 0,
        "final_chars": 0,
    }
    last_guard_check_chars = 0
    guard_check_interval = _env_int("KVEN2_REPETITION_CHECK_INTERVAL", 200, 50, 2000)

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream("POST", chat_url, json=payload) as response:
            logger.info(
                f"[ROUTE] Backend response status: {response.status_code} route_label={route_label}"
            )
            if response.headers.get("content-type", "").startswith("text/event-stream"):
                logger.info(f"[ROUTE] ✅ SSE stream detected. Parsing chunks... route_label={route_label}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    backend_chunks.append(line)
                    piece = _extract_sse_content_piece(line)
                    if piece:
                        accumulated_text += piece

                    if len(accumulated_text) - last_guard_check_chars >= guard_check_interval:
                        last_guard_check_chars = len(accumulated_text)
                        cleaned, current_meta = _sanitize_repetitive_text(accumulated_text)
                        if current_meta.get("detected"):
                            accumulated_text = cleaned
                            guard_meta = current_meta
                            logger.warning(
                                "[FINAL_GUARD] backend_stream_aborted route_label=%s meta=%s",
                                route_label,
                                _json_preview(guard_meta, limit=1000),
                            )
                            break

                logger.info(
                    f"[ROUTE] ✅ Stream completed. Total chunks: {len(backend_chunks)} "
                    f"route_label={route_label} repetition_detected={guard_meta.get('detected')}"
                )
            else:
                logger.info(
                    f"[ROUTE] ⚠️ Non-SSE response detected. Falling back to JSON read... route_label={route_label}"
                )
                data = await response.aread()
                resp_json = json.loads(data)
                content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    accumulated_text = strip_reasoning(content)
                    accumulated_text, guard_meta = _sanitize_repetitive_text(accumulated_text)
                    backend_chunks.append(f"data: {json.dumps({'content': accumulated_text}, ensure_ascii=False)}")
                    logger.info(
                        f"[ROUTE] ✅ JSON fallback content extracted. Length: {len(accumulated_text)} "
                        f"route_label={route_label} repetition_detected={guard_meta.get('detected')}"
                    )

    if guard_meta.get("detected"):
        completion = _completion_with_direct_answer(
            {"model": payload.get("model") or settings.MAIN_MODEL},
            accumulated_text,
            "stop",
        )
        backend_chunks = [
            line
            for line in _completion_json_to_sse(completion).decode("utf-8").splitlines()
            if line.startswith("data: ")
        ]

    return backend_chunks, guard_meta


def _extract_assistant_reply(backend_chunks: list[str]) -> str:
    """Robust parsing for both flat and OpenAI-style delta chunks."""
    assistant_reply = ""
    for line in backend_chunks:
        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw == "[DONE]":
                continue
            try:
                chunk_obj = json.loads(raw)

                # 1. Try standard OpenAI structure: choices -> delta -> content
                choices = chunk_obj.get("choices", [])
                content = None
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")

                # 2. Fallback: flat structure {"content": ...}
                if not content:
                    content = chunk_obj.get("content")
                if content:
                    assistant_reply += content

            except json.JSONDecodeError:
                logger.debug(f"[ROUTE] Non-JSON chunk ignored: {raw[:50]}...")

    return strip_reasoning(assistant_reply)


@router.get("/models")
async def list_models():
    try:
        logger.info("[ROUTE] >>> Fetching models list from backend")
        async with httpx.AsyncClient(timeout=1000.0) as client:
            r = await client.get(f"{BASE_BACKEND}/models")
            r.raise_for_status()
            logger.info("[ROUTE] ✅ Models list retrieved successfully")
            return Response(content=r.content, media_type="application/json")
    except Exception as e:
        logger.error(f"[MODELS] Error: {e}", exc_info=True)
        return JSONResponse(content={"object": "list", "data": []}, status_code=200)


@router.get("/slots")
async def list_slots():
    return [{"id": "default", "name": "Kven Gateway", "object": "slot"}]


@router.post("/chat/completions")
async def handle_chat(request: Request):
    request_id = uuid.uuid4().hex[:12]
    _REQUEST_ID.set(request_id)

    try:
        logger.info("[ROUTE] >>> Incoming request to /chat/completions")
        logger.info(f"[ROUTE_VERSION] {ROUTES_TOOLS_PROBE_VERSION}")
        body = await request.json()
        _debug_log_incoming_payload(body)

        # OWUI is the read-only library layer. Remove only mutation tools from
        # the model-visible catalogue; Knowledge and memory read tools remain.
        _apply_owui_read_only_library_policy(body)

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        # A direct "remember this user-supplied fact" request belongs exclusively
        # to Kven write_path. Hide every OWUI tool for that turn so the model does
        # not waste reasoning on read-only memory functions.
        _disable_tools_for_pure_kven_memory_write(body, messages)

        model_name = body.get("model") or settings.MAIN_MODEL
        backend_model_name = settings.MAIN_MODEL
        model_adapter = resolve_model_adapter(
            backend_model=backend_model_name,
            backend_url=BASE_BACKEND,
        )
        msg_count = len(messages)
        logger.info(f"[ROUTE] Model: {model_name}, Messages count: {msg_count}")
        logger.info(
            "[MODEL_ADAPTER] selected adapter=%s requested_model=%s "
            "backend_model=%s backend_url=%s adapter_enabled=True",
            model_adapter.adapter_id,
            model_name,
            backend_model_name,
            BASE_BACKEND,
        )
        tool_loop_log_incoming_tool_request(body)

        # Detect and sanitize OWUI RAG before native tool routing. OWUI 0.10.2
        # advertises its full tools catalog on ordinary Knowledge/RAG requests;
        # `tools` alone therefore cannot mean "bypass all Kven context".
        write_path_messages, owui_rag_meta = _sanitize_messages_for_write_path(messages)
        if owui_rag_meta.get("detected"):
            logger.info(
                "[OWUI_RAG_CONTEXT] detected=True rag_message_indices=%s source_count=%s "
                "sources=%s original_chars=%s sanitized_chars=%s "
                "tool_protocol_messages_removed=%s",
                owui_rag_meta.get("rag_message_indices"),
                owui_rag_meta.get("source_count"),
                _json_preview(owui_rag_meta.get("sources"), limit=1200),
                owui_rag_meta.get("original_chars"),
                owui_rag_meta.get("sanitized_chars"),
                owui_rag_meta.get("tool_protocol_messages_removed", 0),
            )

        # Classify OWUI service prompts before native-tool routing. Ordinary
        # user-facing native-tool requests must receive the same Kven profile,
        # semantic/project context and vector retrieval as the normal route.
        # Only OWUI's own title/tag/summary requests remain transparent.
        internal_request, internal_reason = detect_internal_owui_request(messages)
        if internal_request and owui_rag_meta.get("detected"):
            logger.info(
                "[OWUI_FILTER] override_internal_false reason=owui_rag_context original_reason=%s",
                internal_reason,
            )
            internal_request = False
            internal_reason = "owui_rag_context_user_facing"
        logger.info(f"[OWUI_FILTER] internal={internal_request} reason={internal_reason}")

        native_protocol = _is_native_openai_tool_protocol(body, messages)
        native_continuation = _messages_include_native_tool_continuation(messages)
        hybrid_native_user = bool(native_protocol and not internal_request)

        if hybrid_native_user:
            logger.info(
                "[OWUI_NATIVE_HYBRID] selected rag=%s continuation=%s tools_count=%s",
                bool(owui_rag_meta.get("detected")),
                native_continuation,
                len(body.get("tools", [])) if isinstance(body.get("tools"), list) else 0,
            )

        # Internal OWUI metadata prompts must not receive Kven memory/retrieval,
        # even if a future OWUI version happens to attach a tools catalogue.
        if native_protocol and internal_request:
            chat_url = f"{BASE_BACKEND}/chat/completions"
            native_payload = _prepare_native_tool_payload(body, messages, model_name)
            logger.info(
                "[OWUI_NATIVE_TOOLS] internal_pass_through_start backend=%s summary=%s",
                chat_url,
                _json_preview(_summarize_native_tools(body, messages), limit=4000),
            )
            return await _proxy_native_openai_tool_protocol(native_payload, chat_url)

        active_state = {}
        skip_write_path = internal_request

        if internal_request:
            # Internal OWUI metadata prompts should be proxied as-is.
            # Do not add Kven identity, memory, project context, active_state,
            # vector retrieval context, history snapshots, or write_path.
            enriched_messages = messages
            logger.info(
                "[OWUI_FILTER] Bypassing Kven context/retrieval/write_path for internal OWUI request."
            )
        else:
            # 1. Версионирование
            logger.info("[ROUTE] Loading active state from DB...")
            active_state = await load_active_state()
            logger.info(f"[ROUTE] Active state keys: {list(active_state.keys())}")

            await save_history_snapshot(active_state)
            logger.info("[ROUTE] ✅ History snapshot saved")

            # 2. Сборка промпта
            profile = load_agent_profile()
            logger.info(
                "[ROUTE] Profile loaded. Current time is resolved by tool only when needed."
            )

            sys_block = ""
            if profile:
                sys_block += f"name: {profile.get('agent_name', 'Kven II')}\n"
                sys_block += "role: You are my friend.\n\n"
                sys_block += f"You are {profile.get('agent_name', 'Kven II')}.\n"
                sys_block += f"Agent Role: {profile.get('agent_role', '')}\n"
                sys_block += f"Project History: {profile.get('project_history', '')}\n"
                sys_block += f"Owner: {profile.get('owner', '')}\n"
                sys_block += f"Mission: {profile.get('mission', '')}\n\n"

            sys_block += (
                "CURRENT DATE AND TIME POLICY:\n"
                "- Current server time is not injected into every prompt.\n"
                "- When the current date or time matters, use the get_time tool if available.\n"
                "- Do not guess the current date or time.\n"
                "- Do not claim that realtime access is unavailable when get_time is available.\n\n"
            )
            sys_block += (
                "MEMORY OWNERSHIP POLICY:\n"
                "- OWUI Knowledge/Collections and OWUI Memory are read-only reference sources.\n"
                "- Never attempt to add, update, replace, or delete OWUI memory.\n"
                "- Kven II owns durable experiential memory and persists grounded user facts automatically after the response.\n"
                "- This persistence is automatic and invisible; do not search for or request a memory-writing tool.\n"
                "- When the user says to remember a fact, acknowledge it directly without inventing extra preferences.\n\n"
            )

            tool_availability_context = tool_loop_format_gateway_tool_availability_context(body)
            if tool_availability_context:
                sys_block += tool_availability_context
                logger.info(
                    "[TOOLS_LOOP] Injected gateway tool availability context for tools=%s",
                    tool_loop_exposed_enabled_tool_names(body),
                )

            # PHASE 2: Семантическая память
            semantic_memories = await get_semantic_context(limit=5)
            if semantic_memories:
                sys_block += f"\n--- SEMANTIC MEMORY (Learned Knowledge) ---\n{semantic_memories}\n"
                logger.debug(f"[ROUTE] Semantic context added. Length: {len(semantic_memories)} chars")

            # PHASE 3: Проект
            current_project_id = active_state.get('current_project_id', 1)
            project_goal = await get_project_context(current_project_id)
            if project_goal:
                sys_block += f"\n{project_goal}\n"
                logger.debug(f"[ROUTE] Project context added: {project_goal}")

            sys_block += "ACTIVE STATE:\n"
            sys_block += f"- Active Problem: {active_state.get('active_problem', 'None')}\n"
            sys_block += f"- Salience: {active_state.get('salience', 0.0)}\n"
            sys_block += f"- Confidence: {active_state.get('confidence', 0.0)}\n"

            # PHASE 4: Read-Only Retrieval Injection (ИСПРАВЛЕНА ОШИБКА ТИПА)
            # For OWUI RAG, retrieve against the real query after </context>, not
            # against the synthetic prompt plus external source chunks.
            # Always retrieve from the sanitized conversation. Besides removing
            # OWUI RAG source chunks, this drops assistant.tool_calls and role=tool
            # payloads while preserving the real user request on continuation passes.
            retrieval_messages = write_path_messages
            user_query = _last_user_text(retrieval_messages)

            if user_query:
                vector_context = await retrieve_context(user_query)
                if vector_context:
                    # Безопасное преобразование: извлекаем content из словарей или приводим к str
                    if isinstance(vector_context, list):
                        if len(vector_context) > 0 and isinstance(vector_context[0], dict):
                            vector_text = "\n".join([item.get("content", str(item)) for item in vector_context])
                        else:
                            vector_text = "\n".join(str(item) for item in vector_context)
                    else:
                        vector_text = str(vector_context)
                    sys_block += f"\nVECTOR RETRIEVAL CONTEXT:\n{vector_text}\n"

            if len(sys_block) > 6400:
                sys_block = sys_block[:5800] + "\n[TRUNCATED: SYSTEM BLOCK LIMIT]"
                logger.warning("[ROUTE] ⚠️ System block truncated. Original length > 6400 chars")

            enriched_messages, system_merge_meta = _merge_kven_system_context(messages, sys_block)
            logger.info(
                "[ROUTE] System context merged: kven_chars=%s incoming_system_messages=%s "
                "merged_parts=%s merged_chars=%s total_payload_messages=%s",
                len(sys_block),
                system_merge_meta.get("incoming_system_messages"),
                system_merge_meta.get("merged_system_parts"),
                system_merge_meta.get("merged_system_chars"),
                len(enriched_messages),
            )

        payload = {
            "model": model_name,
            "messages": enriched_messages,
            "stream": body.get("stream", False),
            "temperature": body.get("temperature", 0.7),
            "cache_prompt": True,
        }

        forwarded_params = _apply_generation_passthrough(body, payload)
        if forwarded_params:
            logger.info(f"[ROUTE] Forwarding generation params: {forwarded_params}")
        else:
            logger.info("[ROUTE] No additional generation params to forward.")

        chat_url = f"{BASE_BACKEND}/chat/completions"
        logger.info(f"[ROUTE] Forwarding to backend: {chat_url}")

        if hybrid_native_user:
            # v10j uses a bounded non-streaming decision pass inside the proxy.
            # This floor is retained only for compatibility with continuation/final
            # payload preparation; the guard applies its own strict decision cap.
            first_pass_floor = 4096 if owui_rag_meta.get("detected") else 512
            native_payload = _prepare_native_tool_payload(
                body,
                enriched_messages,
                model_name,
                minimum_first_pass_tokens=first_pass_floor,
            )
            logger.info(
                "[OWUI_NATIVE_HYBRID] pass_through_start backend=%s rag=%s summary=%s "
                "enriched_messages=%s first_pass_floor=%s",
                chat_url,
                bool(owui_rag_meta.get("detected")),
                _json_preview(_summarize_native_tools(body, messages), limit=4000),
                len(enriched_messages),
                first_pass_floor,
            )
            return await _proxy_hybrid_native_openai_tool_protocol(
                native_payload,
                chat_url,
                model_adapter=model_adapter,
                write_path_messages=write_path_messages,
                active_state=active_state,
                owui_rag_meta=owui_rag_meta,
                skip_write_path=skip_write_path,
            )

        if not internal_request:
            main_chat_thinking = (
                _env_bool("KVEN2_MAIN_ENABLE_THINKING", True)
                and not bool(owui_rag_meta.get("detected"))
                and not tool_loop_enabled_for_registered_tools(body)
            )

            payload = _apply_final_answer_safeguards(
                payload,
                route_label="main_final",
                enable_thinking=main_chat_thinking,
            )

            logger.info(
                "[CHAT_POLICY] main_chat_thinking=%s rag=%s "
                "gateway_tool_loop=%s stream=%s",
                main_chat_thinking,
                bool(owui_rag_meta.get("detected")),
                tool_loop_enabled_for_registered_tools(body),
                bool(payload.get("stream", False)),
            )

        backend_chunks = []
        assistant_reply = ""
        generation_guard_meta = {"detected": False}

        # OWUI Knowledge/RAG already carries external context and may advertise
        # legacy tools in a non-native shape. Never start Kven's older hidden
        # gateway tool-decision loop for such a request: it can conflict with the
        # OWUI wrapper and produce an invalid llama.cpp payload. Native/default
        # RAG tool calls are handled by the hybrid branch above.
        rag_blocks_gateway_tool_loop = bool(owui_rag_meta.get("detected"))
        gateway_tool_loop_enabled = tool_loop_enabled_for_registered_tools(body)

        if rag_blocks_gateway_tool_loop and gateway_tool_loop_enabled:
            logger.info(
                "[OWUI_RAG_CONTEXT] skipping_gateway_tool_loop "
                "reason=owui_rag_context_managed_by_owui_or_hybrid"
            )

        live_main_stream = (
            bool(payload.get("stream", False))
            and not internal_request
            and not rag_blocks_gateway_tool_loop
            and not gateway_tool_loop_enabled
        )

        if live_main_stream:
            logger.info(
                "[LIVE_STREAM] selected route_label=main_final "
                "thinking=%s",
                bool(
                    (payload.get("chat_template_kwargs") or {}).get(
                        "enable_thinking"
                    )
                ),
            )
            return _stream_main_chat_response(
                payload,
                chat_url,
                model_adapter=model_adapter,
                write_path_messages=write_path_messages,
                active_state=active_state,
                skip_write_path=skip_write_path,
            )

        # Bounded gateway-owned model-driven loop: allow exactly one registry tool
        # call per request. For explicit get_time, skip the hidden decision request
        # because no arguments are needed. For read_file, even an explicit tool_choice
        # still needs the hidden decision pass so the model can supply path/max_chars.
        if (
            not internal_request
            and not rag_blocks_gateway_tool_loop
            and gateway_tool_loop_enabled
        ):
            try:
                explicit_name = tool_loop_explicit_tool_choice_name(body)
                if explicit_name == "get_time":
                    decision_reply = '{"kven_tool_call":{"name":"get_time","arguments":{}}}'
                    logger.info(
                        "[TOOLS_LOOP] decision_pass_skipped reason=explicit_tool_choice tool=get_time"
                    )
                else:
                    decision_prompt = tool_loop_format_hidden_tool_decision_prompt(body)
                    decision_payload = {
                        "model": model_name,
                        "messages": enriched_messages + [{"role": "user", "content": decision_prompt}],
                        "stream": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 512,
                    }
                    logger.info(
                        "[TOOLS_LOOP] decision_pass_start tools=%s explicit_tool=%s max_tokens=%s messages=%s extractor=direct_json",
                        tool_loop_exposed_enabled_tool_names(body),
                        explicit_name,
                        decision_payload["max_tokens"],
                        len(decision_payload["messages"]),
                    )
                    decision_reply, _decision_diagnostics = await tool_loop_forward_tool_decision_and_extract_text(
                        decision_payload,
                        chat_url,
                        timeout_seconds=240.0,
                    )
                logger.info(
                    "[TOOLS_LOOP] decision_pass_complete length=%s preview=%s",
                    len(decision_reply),
                    decision_reply[:500],
                )
                requested_call = tool_loop_extract_gateway_tool_call(decision_reply)
            except Exception as exc:
                requested_call = None
                logger.error(
                    "[TOOLS_LOOP] decision_pass_failed fallback_to_normal_answer error=%s",
                    exc,
                    exc_info=True,
                )

            if requested_call:
                logger.info(
                    "[TOOLS_LOOP] model_requested_tool name=%s arguments=%s",
                    requested_call.get("name"),
                    _json_preview(requested_call.get("arguments"), limit=1000),
                )
                tool_result = await sandbox_execute_gateway_tool(requested_call)
                tool_result_message = tool_loop_format_tool_result_message(tool_result)
                loop_messages = enriched_messages + [
                    {
                        "role": "assistant",
                        "content": json.dumps({"kven_tool_call": requested_call}, ensure_ascii=False),
                    },
                    {"role": "user", "content": tool_result_message},
                ]
                loop_payload = dict(payload)
                loop_payload["messages"] = loop_messages
                logger.info(
                    "[TOOLS_LOOP] final_pass_with_tool_start tool=%s status=%s messages=%s",
                    tool_result.get("tool_name"),
                    tool_result.get("status"),
                    len(loop_messages),
                )
                backend_chunks, generation_guard_meta = await _forward_to_backend_and_collect(
                    loop_payload,
                    chat_url,
                    route_label="final_with_tool",
                )
                assistant_reply = _extract_assistant_reply(backend_chunks)
                if not assistant_reply.strip():
                    fallback_reply = tool_loop_format_empty_final_tool_fallback(tool_result)
                    if fallback_reply:
                        assistant_reply = fallback_reply
                        backend_chunks = [
                            "data: " + json.dumps({"content": assistant_reply}, ensure_ascii=False)
                        ]
                        generation_guard_meta = {"detected": False, "reason": "tool_fallback"}
                        logger.warning(
                            "[TOOLS_LOOP] final_pass_empty_using_tool_fallback tool=%s fallback_length=%s",
                            tool_result.get("tool_name"),
                            len(assistant_reply),
                        )
                logger.info(
                    "[TOOLS_LOOP] final_pass_with_tool_complete tool=%s final_length=%s",
                    tool_result.get("tool_name"),
                    len(assistant_reply),
                )
            else:
                logger.info("[TOOLS_LOOP] no_model_tool_call_detected fallback_to_normal_answer")
                backend_chunks, generation_guard_meta = await _forward_to_backend_and_collect(
                    payload,
                    chat_url,
                    route_label="normal_after_no_tool",
                )
                assistant_reply = _extract_assistant_reply(backend_chunks)
        else:
            backend_chunks, generation_guard_meta = await _forward_to_backend_and_collect(
                payload,
                chat_url,
                route_label="normal",
            )
            assistant_reply = _extract_assistant_reply(backend_chunks)

        async def generate():
            for chunk in backend_chunks:
                yield chunk + "\n"

        logger.info(
            f"[ROUTE] Final assistant_reply length: {len(assistant_reply)} chars. "
            f"First 100: {assistant_reply[:100]}"
        )

        if skip_write_path:
            logger.info("[OWUI_FILTER] Internal request detected. Skipping [WRITE_PATH].")
        elif generation_guard_meta.get("detected"):
            logger.warning(
                "[ROUTE] Skipping [WRITE_PATH] reason=repetition_loop meta=%s",
                _json_preview(generation_guard_meta, limit=1000),
            )
        elif assistant_reply and len(assistant_reply.strip()) > 10:
            if owui_rag_meta.get("detected"):
                logger.info(
                    "[OWUI_RAG_CONTEXT] write_path sanitized: external OWUI RAG source chunks removed before memory pipeline."
                )
            logger.info("[ROUTE] ✅ Triggering background task [WRITE_PATH]")
            asyncio.create_task(process_episodic(write_path_messages, assistant_reply, active_state))
        else:
            logger.warning(
                "[ROUTE] ⚠️ Assistant reply too short/empty. "
                "Skipping [WRITE_PATH] to prevent garbage accumulation."
            )

        stream_requested = bool(payload.get("stream", False))

        if stream_requested:
            return StreamingResponse(generate(), media_type="text/event-stream")

        completion = _completion_with_direct_answer(
            {"model": payload.get("model") or settings.MAIN_MODEL},
            assistant_reply,
            "stop",
        )
        logger.info(
            "[ROUTE] Returning OpenAI JSON response "
            "stream_requested=False content_len=%s",
            len(assistant_reply),
        )
        return JSONResponse(content=completion)

    except Exception as e:
        logger.error(f"[ROUTE] Gateway Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
