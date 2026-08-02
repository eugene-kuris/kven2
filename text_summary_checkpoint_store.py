import asyncio
import hashlib
import json
import logging
from typing import Any

import sqlite as kven_sqlite

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHECKPOINTS = 256
DEFAULT_LOAD_LIMIT = 64
MAX_STORE_LIMIT = 4096
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX_DIGITS
    )


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_limit(value: Any, *, default: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_STORE_LIMIT
    ):
        return default
    return value


def _normalize_checkpoint(checkpoint: Any) -> tuple[dict, str]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")

    checkpoint_id = checkpoint.get("checkpoint_id")
    checkpoint_version = checkpoint.get("checkpoint_version")
    hash_scope = checkpoint.get("hash_scope")
    summarized_message_count = checkpoint.get(
        "summarized_message_count"
    )
    prefix_sha256 = checkpoint.get("prefix_sha256")
    summary_sha256 = checkpoint.get("summary_sha256")
    summary_text = checkpoint.get("summary_text")
    summary_chars = checkpoint.get("summary_chars")

    if not _is_sha256_hex(checkpoint_id):
        raise ValueError("checkpoint_id must be a SHA-256 digest")
    if not isinstance(checkpoint_version, str) or not checkpoint_version:
        raise ValueError("checkpoint_version must not be empty")
    if not isinstance(hash_scope, str) or not hash_scope:
        raise ValueError("hash_scope must not be empty")
    if (
        isinstance(summarized_message_count, bool)
        or not isinstance(summarized_message_count, int)
        or summarized_message_count < 1
    ):
        raise ValueError(
            "summarized_message_count must be a positive integer"
        )
    if not _is_sha256_hex(prefix_sha256):
        raise ValueError("prefix_sha256 must be a SHA-256 digest")
    if not _is_sha256_hex(summary_sha256):
        raise ValueError("summary_sha256 must be a SHA-256 digest")
    if (
        not isinstance(summary_text, str)
        or not summary_text
        or summary_text != summary_text.strip()
    ):
        raise ValueError("summary_text must be normalized and non-empty")
    if (
        isinstance(summary_chars, bool)
        or not isinstance(summary_chars, int)
        or summary_chars != len(summary_text)
    ):
        raise ValueError("summary_chars does not match summary_text")

    expected_summary_sha256 = hashlib.sha256(
        summary_text.encode("utf-8")
    ).hexdigest()
    if summary_sha256 != expected_summary_sha256:
        raise ValueError("summary_sha256 does not match summary_text")

    checkpoint_identity = {
        "checkpoint_version": checkpoint_version,
        "hash_scope": hash_scope,
        "summarized_message_count": summarized_message_count,
        "prefix_sha256": prefix_sha256,
        "summary_sha256": summary_sha256,
    }
    if checkpoint_id != _canonical_json_sha256(checkpoint_identity):
        raise ValueError("checkpoint_id does not match checkpoint identity")

    normalized = {
        "checkpoint_version": checkpoint_version,
        "hash_scope": hash_scope,
        "summarized_message_count": summarized_message_count,
        "prefix_sha256": prefix_sha256,
        "summary_sha256": summary_sha256,
        "checkpoint_id": checkpoint_id,
        "summary_text": summary_text,
        "summary_chars": summary_chars,
    }
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return normalized, payload


def _sync_save_text_summary_checkpoint(
    checkpoint: dict,
    *,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> bool:
    try:
        normalized, payload = _normalize_checkpoint(checkpoint)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "[TEXT_SUMMARY_CHECKPOINT_STORE] save_rejected error=%s",
            exc,
        )
        return False

    store_limit = _normalize_limit(
        max_checkpoints,
        default=DEFAULT_MAX_CHECKPOINTS,
    )
    try:
        conn = kven_sqlite.get_connection()
    except Exception as exc:
        logger.warning(
            "[TEXT_SUMMARY_CHECKPOINT_STORE] connection_failed "
            "operation=save error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return False

    with kven_sqlite.db_lock:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO text_summary_checkpoints (
                    checkpoint_id,
                    checkpoint_version,
                    hash_scope,
                    summarized_message_count,
                    prefix_sha256,
                    summary_sha256,
                    summary_chars,
                    checkpoint_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    checkpoint_version,
                    hash_scope,
                    summarized_message_count,
                    prefix_sha256
                ) DO UPDATE SET
                    checkpoint_id = excluded.checkpoint_id,
                    summary_sha256 = excluded.summary_sha256,
                    summary_chars = excluded.summary_chars,
                    checkpoint_json = excluded.checkpoint_json,
                    updated_at = CURRENT_TIMESTAMP,
                    last_used = CURRENT_TIMESTAMP
                """,
                (
                    normalized["checkpoint_id"],
                    normalized["checkpoint_version"],
                    normalized["hash_scope"],
                    normalized["summarized_message_count"],
                    normalized["prefix_sha256"],
                    normalized["summary_sha256"],
                    normalized["summary_chars"],
                    payload,
                ),
            )
            conn.execute(
                """DELETE FROM text_summary_checkpoints
                WHERE rowid IN (
                    SELECT rowid
                    FROM text_summary_checkpoints
                    ORDER BY
                        last_used DESC,
                        updated_at DESC,
                        rowid DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (store_limit,),
            )
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.warning(
                "[TEXT_SUMMARY_CHECKPOINT_STORE] save_failed "
                "error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return False
        finally:
            conn.close()


def _sync_load_text_summary_checkpoints(
    *,
    max_summarized_message_count: int | None = None,
    limit: int = DEFAULT_LOAD_LIMIT,
) -> list[dict]:
    load_limit = _normalize_limit(limit, default=DEFAULT_LOAD_LIMIT)
    max_count = max_summarized_message_count
    if (
        max_count is not None
        and (
            isinstance(max_count, bool)
            or not isinstance(max_count, int)
            or max_count < 1
        )
    ):
        max_count = None

    try:
        conn = kven_sqlite.get_connection()
    except Exception as exc:
        logger.warning(
            "[TEXT_SUMMARY_CHECKPOINT_STORE] connection_failed "
            "operation=load error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return []

    try:
        with kven_sqlite.db_lock:
            if max_count is None:
                rows = conn.execute(
                    """SELECT checkpoint_json
                    FROM text_summary_checkpoints
                    ORDER BY
                        summarized_message_count DESC,
                        last_used DESC,
                        rowid DESC
                    LIMIT ?
                    """,
                    (load_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT checkpoint_json
                    FROM text_summary_checkpoints
                    WHERE summarized_message_count <= ?
                    ORDER BY
                        summarized_message_count DESC,
                        last_used DESC,
                        rowid DESC
                    LIMIT ?
                    """,
                    (max_count, load_limit),
                ).fetchall()
    except Exception as exc:
        logger.warning(
            "[TEXT_SUMMARY_CHECKPOINT_STORE] load_failed "
            "error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return []
    finally:
        conn.close()

    checkpoints = []
    for row in rows:
        try:
            checkpoint = json.loads(row[0])
            normalized, _ = _normalize_checkpoint(checkpoint)
        except Exception as exc:
            logger.warning(
                "[TEXT_SUMMARY_CHECKPOINT_STORE] corrupt_row_skipped "
                "error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            continue
        checkpoints.append(normalized)
    return checkpoints


def _sync_mark_text_summary_checkpoint_used(
    checkpoint_id: str,
) -> bool:
    if not _is_sha256_hex(checkpoint_id):
        return False

    try:
        conn = kven_sqlite.get_connection()
    except Exception as exc:
        logger.warning(
            "[TEXT_SUMMARY_CHECKPOINT_STORE] connection_failed "
            "operation=mark_used error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return False

    with kven_sqlite.db_lock:
        try:
            cursor = conn.execute(
                """UPDATE text_summary_checkpoints
                SET
                    last_used = CURRENT_TIMESTAMP,
                    usage_count = usage_count + 1
                WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            )
            conn.commit()
            return cursor.rowcount == 1
        except Exception as exc:
            conn.rollback()
            logger.warning(
                "[TEXT_SUMMARY_CHECKPOINT_STORE] mark_used_failed "
                "error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return False
        finally:
            conn.close()


def _sync_clear_text_summary_checkpoints() -> bool:
    try:
        conn = kven_sqlite.get_connection()
    except Exception as exc:
        logger.warning(
            "[TEXT_SUMMARY_CHECKPOINT_STORE] connection_failed "
            "operation=clear error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return False

    with kven_sqlite.db_lock:
        try:
            conn.execute("DELETE FROM text_summary_checkpoints")
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.warning(
                "[TEXT_SUMMARY_CHECKPOINT_STORE] clear_failed "
                "error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return False
        finally:
            conn.close()


async def save_text_summary_checkpoint(
    checkpoint: dict,
    *,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> bool:
    return await asyncio.to_thread(
        _sync_save_text_summary_checkpoint,
        checkpoint,
        max_checkpoints=max_checkpoints,
    )


async def load_text_summary_checkpoints(
    *,
    max_summarized_message_count: int | None = None,
    limit: int = DEFAULT_LOAD_LIMIT,
) -> list[dict]:
    return await asyncio.to_thread(
        _sync_load_text_summary_checkpoints,
        max_summarized_message_count=max_summarized_message_count,
        limit=limit,
    )


async def mark_text_summary_checkpoint_used(
    checkpoint_id: str,
) -> bool:
    return await asyncio.to_thread(
        _sync_mark_text_summary_checkpoint_used,
        checkpoint_id,
    )


async def clear_text_summary_checkpoints() -> bool:
    return await asyncio.to_thread(
        _sync_clear_text_summary_checkpoints
    )
