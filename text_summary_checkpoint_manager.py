"""Policy layer for generating persistent text-summary checkpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from context_window import (
    build_text_summary_checkpoint,
    build_text_summary_compaction_preview,
    find_matching_text_summary_checkpoint,
)
from text_summary_checkpoint_store import (
    load_text_summary_checkpoints,
    save_text_summary_checkpoint,
)
from text_summary_generator import generate_text_summary

logger = logging.getLogger(__name__)

_GENERATION_LOCK = asyncio.Lock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)

    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(os.getenv(name, "")).strip())
    except Exception:
        value = default
    return max(minimum, min(value, maximum))


def _env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(str(os.getenv(name, "")).strip())
    except Exception:
        value = default
    return max(minimum, min(value, maximum))


def _message_slice_json_chars(
    messages: list,
    start: int,
    end: int,
) -> int:
    try:
        return len(
            json.dumps(
                messages[start:end],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except Exception:
        return 0


async def maybe_generate_text_summary_checkpoint(
    messages: list,
    *,
    route_label: str,
) -> dict | None:
    """Generate and persist one safe summary checkpoint when warranted.

    The feature is disabled by default. It never mutates the request and
    fails open on planner, validation, or database errors.
    """

    if not _env_bool(
        "KVEN2_TEXT_SUMMARY_GENERATION_ENABLED",
        False,
    ):
        return None

    try:
        tail_messages = _env_int(
            "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES",
            12,
            1,
            200,
        )
        min_messages = _env_int(
            "KVEN2_TEXT_SUMMARY_GENERATION_MIN_MESSAGES",
            8,
            1,
            2000,
        )
        min_chars = _env_int(
            "KVEN2_TEXT_SUMMARY_GENERATION_MIN_CHARS",
            6000,
            1000,
            1000000,
        )
        refresh_messages = _env_int(
            "KVEN2_TEXT_SUMMARY_REFRESH_MIN_MESSAGES",
            8,
            1,
            2000,
        )
        refresh_chars = _env_int(
            "KVEN2_TEXT_SUMMARY_REFRESH_MIN_CHARS",
            4000,
            1000,
            1000000,
        )
        load_limit = _env_int(
            "KVEN2_TEXT_SUMMARY_CHECKPOINT_LOAD_LIMIT",
            64,
            1,
            4096,
        )
        store_limit = _env_int(
            "KVEN2_TEXT_SUMMARY_CHECKPOINT_STORE_LIMIT",
            256,
            1,
            4096,
        )
        timeout_seconds = _env_float(
            "KVEN2_TEXT_SUMMARY_GENERATION_TIMEOUT",
            90.0,
            1.0,
            900.0,
        )
        max_tokens = _env_int(
            "KVEN2_TEXT_SUMMARY_GENERATION_MAX_TOKENS",
            1024,
            128,
            4096,
        )
        max_input_chars = _env_int(
            "KVEN2_TEXT_SUMMARY_GENERATION_MAX_INPUT_CHARS",
            80000,
            4000,
            500000,
        )

        _, boundary_report = (
            build_text_summary_compaction_preview(
                messages,
                summary_text="",
                tail_messages=tail_messages,
            )
        )
        system_prefix_messages = int(
            boundary_report.get(
                "system_prefix_messages",
                0,
            )
            or 0
        )
        safe_summary_boundary_end = int(
            boundary_report.get(
                "safe_summary_boundary_end",
                system_prefix_messages,
            )
            or system_prefix_messages
        )
        candidate_message_count = max(
            0,
            safe_summary_boundary_end
            - system_prefix_messages,
        )
        candidate_chars = _message_slice_json_chars(
            messages,
            system_prefix_messages,
            safe_summary_boundary_end,
        )

        if boundary_report.get(
            "active_tool_continuation"
        ):
            logger.info(
                "[TEXT_SUMMARY_GENERATION] "
                "route_label=%s status=skipped_active_tool",
                route_label,
            )
            return None

        if (
            candidate_message_count < min_messages
            or candidate_chars < min_chars
        ):
            logger.info(
                "[TEXT_SUMMARY_GENERATION] "
                "route_label=%s status=below_initial_threshold "
                "candidate_messages=%s candidate_chars=%s",
                route_label,
                candidate_message_count,
                candidate_chars,
            )
            return None

        async with _GENERATION_LOCK:
            checkpoints = (
                await load_text_summary_checkpoints(
                    max_summarized_message_count=(
                        candidate_message_count
                    ),
                    limit=load_limit,
                )
            )
            previous_checkpoint, match_report = (
                find_matching_text_summary_checkpoint(
                    messages,
                    checkpoints,
                )
            )

            previous_prefix_end = (
                system_prefix_messages
            )
            previous_summary = None

            if previous_checkpoint is not None:
                selected_prefix_end = match_report.get(
                    "selected_summarized_prefix_end"
                )
                if isinstance(
                    selected_prefix_end,
                    int,
                ) and not isinstance(
                    selected_prefix_end,
                    bool,
                ):
                    previous_prefix_end = (
                        selected_prefix_end
                    )
                previous_summary = previous_checkpoint.get(
                    "summary_text"
                )

            new_message_count = max(
                0,
                safe_summary_boundary_end
                - previous_prefix_end,
            )
            new_chars = _message_slice_json_chars(
                messages,
                previous_prefix_end,
                safe_summary_boundary_end,
            )

            if previous_checkpoint is not None and (
                new_message_count < refresh_messages
                and new_chars < refresh_chars
            ):
                logger.info(
                    "[TEXT_SUMMARY_GENERATION] "
                    "route_label=%s status=checkpoint_fresh "
                    "new_messages=%s new_chars=%s",
                    route_label,
                    new_message_count,
                    new_chars,
                )
                return previous_checkpoint

            generation_messages = messages[
                previous_prefix_end:
                safe_summary_boundary_end
            ]
            summary_text, generation_meta = (
                await generate_text_summary(
                    generation_messages,
                    prior_summary=previous_summary,
                    timeout_seconds=timeout_seconds,
                    max_tokens=max_tokens,
                    max_input_chars=max_input_chars,
                )
            )
            checkpoint = build_text_summary_checkpoint(
                messages,
                summary_text=summary_text,
                summarized_prefix_end=(
                    safe_summary_boundary_end
                ),
            )
            saved = await save_text_summary_checkpoint(
                checkpoint,
                max_checkpoints=store_limit,
            )

            logger.info(
                "[TEXT_SUMMARY_GENERATION] %s",
                json.dumps(
                    {
                        "route_label": str(
                            route_label or "unknown"
                        ),
                        "status": (
                            "saved" if saved else "save_failed"
                        ),
                        "candidate_messages": (
                            candidate_message_count
                        ),
                        "candidate_chars": candidate_chars,
                        "previous_checkpoint": bool(
                            previous_checkpoint
                        ),
                        "new_messages": new_message_count,
                        "new_chars": new_chars,
                        "checkpoint_id": checkpoint.get(
                            "checkpoint_id"
                        ),
                        "summary_chars": checkpoint.get(
                            "summary_chars"
                        ),
                        "generation": generation_meta,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )

            return checkpoint if saved else None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "[TEXT_SUMMARY_GENERATION] failed "
            "route_label=%s error_type=%s error=%s",
            route_label,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return None
