from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from telegram_bot_api import split_telegram_text
from context_window import estimate_tokens_from_chars


@dataclass(frozen=True)
class TelegramJob:
    id: int
    update_id: int
    chat_id: int
    user_id: int
    message_id: int
    text: str
    status: str
    attempts: int
    delivery_attempts: int
    response_text: str | None
    stream_id: int | None = None
    batch_update_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TelegramDelivery:
    chunk_id: int
    job_id: int
    update_id: int
    chat_id: int
    user_id: int
    source_message_id: int
    chunk_index: int
    chunk_count: int
    text: str
    status: str
    attempts: int
    reply_to_message_id: int | None

    @property
    def id(self) -> int:
        """Compatibility alias for the parent job ID."""
        return self.job_id

    @property
    def message_id(self) -> int:
        """Compatibility alias for the source Telegram message."""
        return self.source_message_id

    @property
    def response_text(self) -> str:
        """Compatibility alias for single-part delivery callers."""
        return self.text

    @property
    def delivery_attempts(self) -> int:
        """Compatibility alias for this chunk's attempt count."""
        return self.attempts


class TelegramStore:
    def __init__(
        self,
        db_path: str,
        *,
        batch_debounce_seconds: float = 0.0,
        exact_tail_token_budget: int = 4096,
    ):
        self.db_path = db_path
        if batch_debounce_seconds < 0:
            raise ValueError(
                "Telegram batch debounce must be "
                "non-negative"
            )
        if exact_tail_token_budget <= 0:
            raise ValueError(
                "Telegram exact-tail budget must be positive"
            )
        self.batch_debounce_seconds = float(batch_debounce_seconds)
        self.exact_tail_token_budget = int(exact_tail_token_budget)

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def get_next_update_offset(self) -> int:
        return await asyncio.to_thread(
            self._get_next_update_offset_sync
        )

    async def advance_update_offset(
        self,
        next_offset: int,
    ) -> None:
        if (
            not isinstance(next_offset, int)
            or isinstance(next_offset, bool)
            or next_offset < 0
        ):
            raise ValueError(
                "Telegram update offset must be "
                "a non-negative integer"
            )

        await asyncio.to_thread(
            self._advance_update_offset_sync,
            next_offset,
        )

    async def enqueue_text_update(
        self,
        *,
        update_id: int,
        chat_id: int,
        user_id: int,
        message_id: int,
        text: str,
        raw_update: dict[str, Any],
        message_date: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._enqueue_text_update_sync,
            update_id,
            chat_id,
            user_id,
            message_id,
            text,
            raw_update,
            message_date,
            reply_to_message_id,
        )

    async def claim_next_job(self) -> TelegramJob | None:
        return await asyncio.to_thread(
            self._claim_next_job_sync
        )

    async def claim_next_delivery(
        self,
    ) -> TelegramDelivery | None:
        return await asyncio.to_thread(
            self._claim_next_delivery_sync
        )

    async def save_response(
        self,
        job_id: int,
        response_text: str,
    ) -> None:
        await asyncio.to_thread(
            self._save_response_sync,
            job_id,
            response_text,
        )

    async def mark_delivery_chunk_delivered(
        self,
        chunk_id: int,
        telegram_message_id: int,
    ) -> bool:
        return await asyncio.to_thread(
            self._mark_delivery_chunk_delivered_sync,
            chunk_id,
            telegram_message_id,
        )

    async def mark_delivered(
        self,
        job_id: int,
        telegram_message_id: int,
    ) -> None:
        """Compatibility wrapper for legacy single-part callers."""
        await asyncio.to_thread(
            self._mark_delivered_sync,
            job_id,
            telegram_message_id,
        )

    async def load_conversation(
        self,
        chat_id: int,
        *,
        through_update_id: int | None = None,
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(
            self._load_conversation_sync,
            chat_id,
            through_update_id,
        )

    async def build_generation_context(
        self,
        job: TelegramJob,
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(
            self._build_generation_context_sync,
            job,
        )

    async def recover_incomplete_jobs(self) -> int:
        return await asyncio.to_thread(
            self._recover_incomplete_jobs_sync
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_sync(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as connection:
            connection.execute(
                "PRAGMA journal_mode = WAL"
            )
            connection.execute(
                "PRAGMA synchronous = NORMAL"
            )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS telegram_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS telegram_updates (
                    update_id INTEGER PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    received_at TEXT NOT NULL DEFAULT (
                        strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    )
                );

                CREATE TABLE IF NOT EXISTS telegram_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    update_id INTEGER NOT NULL UNIQUE,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL
                        DEFAULT 'queued'
                        CHECK (
                            status IN (
                                'queued',
                                'processing',
                                'responded',
                                'sending',
                                'delivered',
                                'failed'
                            )
                        ),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    response_text TEXT,
                    telegram_response_message_id INTEGER,
                    last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    ),
                    updated_at TEXT NOT NULL DEFAULT (
                        strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    ),
                    started_at TEXT,
                    delivery_started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (update_id)
                        REFERENCES telegram_updates(
                            update_id
                        )
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_telegram_jobs_status_id
                    ON telegram_jobs(status, id);

                CREATE TABLE IF NOT EXISTS
                    telegram_delivery_chunks (
                        id INTEGER
                            PRIMARY KEY AUTOINCREMENT,
                        job_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        status TEXT NOT NULL
                            DEFAULT 'pending'
                            CHECK (
                                status IN (
                                    'pending',
                                    'sending',
                                    'delivered'
                                )
                            ),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        telegram_message_id INTEGER,
                        created_at TEXT NOT NULL DEFAULT (
                            strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            )
                        ),
                        updated_at TEXT NOT NULL DEFAULT (
                            strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            )
                        ),
                        started_at TEXT,
                        delivered_at TEXT,
                        UNIQUE(job_id, chunk_index),
                        FOREIGN KEY (job_id)
                            REFERENCES telegram_jobs(id)
                            ON DELETE CASCADE
                    );

                CREATE INDEX IF NOT EXISTS
                    idx_telegram_delivery_chunks_status
                    ON telegram_delivery_chunks(
                        status,
                        job_id,
                        chunk_index
                    );

                CREATE TABLE IF NOT EXISTS
                    telegram_messages (
                        id INTEGER
                            PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK (
                            role IN (
                                'user',
                                'assistant'
                            )
                        ),
                        content TEXT NOT NULL,
                        source_update_id INTEGER NOT NULL,
                        telegram_message_id INTEGER,
                        created_at TEXT NOT NULL DEFAULT (
                            strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            )
                        ),
                        UNIQUE(
                            source_update_id,
                            role
                        ),
                        FOREIGN KEY (
                            source_update_id
                        )
                            REFERENCES telegram_updates(
                                update_id
                            )
                            ON DELETE CASCADE
                    );

                CREATE INDEX IF NOT EXISTS
                    idx_telegram_messages_chat_id_id
                    ON telegram_messages(
                        chat_id,
                        id
                    );

                CREATE TABLE IF NOT EXISTS telegram_streams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    UNIQUE(chat_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS telegram_job_messages (
                    job_id INTEGER NOT NULL,
                    update_id INTEGER NOT NULL UNIQUE,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(job_id, ordinal),
                    FOREIGN KEY(job_id) REFERENCES telegram_jobs(id),
                    FOREIGN KEY(update_id) REFERENCES telegram_updates(update_id)
                );

                """
            )
            for table, column, declaration in (
                ("telegram_updates", "message_date", "INTEGER"),
                ("telegram_updates", "reply_to_message_id", "INTEGER"),
                ("telegram_jobs", "stream_id", "INTEGER"),
                ("telegram_jobs", "ready_at", "REAL"),
                ("telegram_messages", "stream_id", "INTEGER"),
                ("telegram_messages", "telegram_date", "INTEGER"),
                ("telegram_messages", "reply_to_message_id", "INTEGER"),
            ):
                self._ensure_column(
                    connection,
                    table,
                    column,
                    declaration,
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_telegram_jobs_stream_status "
                "ON telegram_jobs(stream_id, status, id)"
            )
            self._migrate_relationship_rows(connection)

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(
                f"PRAGMA table_info({table})"
            )
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _migrate_relationship_rows(connection: sqlite3.Connection) -> None:
        TelegramStore._backfill_update_metadata(connection)
        connection.execute(
            """INSERT OR IGNORE INTO telegram_streams(chat_id, user_id)
               SELECT DISTINCT chat_id, user_id FROM telegram_updates"""
        )
        connection.execute(
            """UPDATE telegram_jobs SET stream_id = (
                   SELECT id FROM telegram_streams s
                   WHERE s.chat_id = telegram_jobs.chat_id
                     AND s.user_id = telegram_jobs.user_id)
               WHERE stream_id IS NULL"""
        )
        connection.execute(
            """UPDATE telegram_jobs SET ready_at = 0 WHERE ready_at IS NULL"""
        )
        connection.execute(
            """INSERT OR IGNORE INTO telegram_job_messages(job_id, update_id, ordinal)
               SELECT id, update_id, 0 FROM telegram_jobs"""
        )
        connection.execute(
            """UPDATE telegram_messages SET stream_id = (
                   SELECT j.stream_id FROM telegram_jobs j
                   WHERE j.update_id = telegram_messages.source_update_id)
               WHERE stream_id IS NULL"""
        )
        connection.execute(
            """UPDATE telegram_messages
               SET telegram_date = (
                       SELECT COALESCE(
                           telegram_messages.telegram_date,
                           u.message_date
                       )
                       FROM telegram_updates AS u
                       WHERE u.update_id = telegram_messages.source_update_id
                   ),
                   reply_to_message_id = (
                       SELECT COALESCE(
                           telegram_messages.reply_to_message_id,
                           u.reply_to_message_id
                       )
                       FROM telegram_updates AS u
                       WHERE u.update_id = telegram_messages.source_update_id
                   )
               WHERE role = 'user'
                 AND (telegram_date IS NULL OR reply_to_message_id IS NULL)"""
        )

    @staticmethod
    def _backfill_update_metadata(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """SELECT update_id, message_id, raw_json
               FROM telegram_updates
               WHERE message_date IS NULL OR reply_to_message_id IS NULL"""
        ).fetchall()
        for row in rows:
            try:
                raw = json.loads(str(row["raw_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            message = raw.get("message")
            if not isinstance(message, dict):
                message = raw.get("edited_message")
            if not isinstance(message, dict):
                continue
            raw_message_id = message.get("message_id")
            if (
                not isinstance(raw_message_id, int)
                or isinstance(raw_message_id, bool)
                or raw_message_id != int(row["message_id"])
            ):
                continue
            message_date = message.get("date")
            if (
                not isinstance(message_date, int)
                or isinstance(message_date, bool)
                or message_date < 0
            ):
                message_date = None
            reply = message.get("reply_to_message")
            reply_id = reply.get("message_id") if isinstance(reply, dict) else None
            if (
                not isinstance(reply_id, int)
                or isinstance(reply_id, bool)
                or reply_id <= 0
            ):
                reply_id = None
            connection.execute(
                """UPDATE telegram_updates
                   SET message_date = COALESCE(message_date, ?),
                       reply_to_message_id = COALESCE(reply_to_message_id, ?)
                   WHERE update_id = ?""",
                (message_date, reply_id, row["update_id"]),
            )

    @staticmethod
    def _advance_offset_sync(
        connection: sqlite3.Connection,
        next_offset: int,
    ) -> None:
        row = connection.execute(
            """
            SELECT value
            FROM telegram_state
            WHERE key = 'next_update_offset'
            """
        ).fetchone()

        current_offset = (
            int(row["value"])
            if row is not None
            else 0
        )
        new_offset = max(
            current_offset,
            next_offset,
        )

        connection.execute(
            """
            INSERT INTO telegram_state(
                key,
                value
            )
            VALUES(
                'next_update_offset',
                ?
            )
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value
            """,
            (str(new_offset),),
        )

    def _get_next_update_offset_sync(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT value
                FROM telegram_state
                WHERE key = 'next_update_offset'
                """
            ).fetchone()

        return (
            int(row["value"])
            if row is not None
            else 0
        )

    def _advance_update_offset_sync(
        self,
        next_offset: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                self._advance_offset_sync(
                    connection,
                    next_offset,
                )
                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _enqueue_text_update_sync(
        self,
        update_id: int,
        chat_id: int,
        user_id: int,
        message_id: int,
        text: str,
        raw_update: dict[str, Any],
        message_date: int | None,
        reply_to_message_id: int | None,
    ) -> bool:
        raw_json = json.dumps(
            raw_update,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                existing = connection.execute(
                    """
                    SELECT 1
                    FROM telegram_updates
                    WHERE update_id = ?
                    """,
                    (update_id,),
                ).fetchone()

                if existing is not None:
                    self._advance_offset_sync(
                        connection,
                        update_id + 1,
                    )
                    connection.execute("COMMIT")
                    return False

                connection.execute(
                    """
                    INSERT INTO telegram_updates(
                        update_id,
                        chat_id,
                        user_id,
                        message_id,
                        text,
                        raw_json, message_date, reply_to_message_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        update_id,
                        chat_id,
                        user_id,
                        message_id,
                        text,
                        raw_json,
                        message_date,
                        reply_to_message_id,
                    ),
                )

                connection.execute(
                    "INSERT OR IGNORE INTO telegram_streams(chat_id, user_id) VALUES (?, ?)",
                    (chat_id, user_id),
                )
                stream_id = int(connection.execute(
                    "SELECT id FROM telegram_streams WHERE chat_id = ? AND user_id = ?",
                    (chat_id, user_id),
                ).fetchone()["id"])

                queued = connection.execute(
                    "SELECT id FROM telegram_jobs WHERE stream_id = ? AND status = 'queued' ORDER BY id DESC LIMIT 1",
                    (stream_id,),
                ).fetchone()

                if queued is None:
                    cursor = connection.execute(
                    """
                    INSERT INTO telegram_jobs(
                        update_id, stream_id, ready_at,
                        chat_id,
                        user_id,
                        message_id,
                        text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        update_id,
                        stream_id,
                        time.time() + self.batch_debounce_seconds,
                        chat_id,
                        user_id,
                        message_id,
                        text,
                    ),
                )
                    job_id = int(cursor.lastrowid)
                    ordinal = 0
                else:
                    job_id = int(queued["id"])
                    ordinal = int(connection.execute(
                        "SELECT COUNT(*) AS count FROM telegram_job_messages WHERE job_id = ?",
                        (job_id,),
                    ).fetchone()["count"])
                    connection.execute(
                        "UPDATE telegram_jobs SET ready_at = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
                        (time.time() + self.batch_debounce_seconds, job_id),
                    )
                connection.execute(
                    "INSERT INTO telegram_job_messages(job_id, update_id, ordinal) VALUES (?, ?, ?)",
                    (job_id, update_id, ordinal),
                )

                connection.execute(
                    """
                    INSERT INTO telegram_messages(
                        chat_id, stream_id, telegram_date, reply_to_message_id,
                        role,
                        content,
                        source_update_id,
                        telegram_message_id
                    )
                    VALUES (
                        ?, ?, ?, ?,
                        'user',
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        chat_id, stream_id, message_date, reply_to_message_id,
                        text,
                        update_id,
                        message_id,
                    ),
                )

                self._advance_offset_sync(
                    connection,
                    update_id + 1,
                )

                connection.execute("COMMIT")
                return True

            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _claim_next_job_sync(
        self,
    ) -> TelegramJob | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                row = connection.execute(
                    """
                    SELECT *
                    FROM telegram_jobs
                    WHERE status = 'queued'
                      AND ready_at <= ?
                      AND NOT EXISTS (
                          SELECT 1 FROM telegram_jobs active
                          WHERE active.stream_id = telegram_jobs.stream_id
                            AND active.status = 'processing'
                      )
                    ORDER BY id
                    LIMIT 1
                    """,
                    (time.time(),),
                ).fetchone()

                if row is None:
                    connection.execute("COMMIT")
                    return None

                connection.execute(
                    """
                    UPDATE telegram_jobs
                    SET status = 'processing',
                        attempts = attempts + 1,
                        started_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        ),
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        ),
                        last_error = NULL
                    WHERE id = ?
                      AND status = 'queued'
                    """,
                    (row["id"],),
                )

                claimed = connection.execute(
                    """
                    SELECT *
                    FROM telegram_jobs
                    WHERE id = ?
                    """,
                    (row["id"],),
                ).fetchone()

                batch_update_ids = tuple(
                    int(item["update_id"])
                    for item in connection.execute(
                        "SELECT update_id FROM telegram_job_messages WHERE job_id = ? ORDER BY ordinal",
                        (row["id"],),
                    ).fetchall()
                )

                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

        return self._job_from_row(claimed, batch_update_ids)

    def _claim_next_delivery_sync(
        self,
    ) -> TelegramDelivery | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                job = connection.execute(
                    """
                    SELECT *
                    FROM telegram_jobs
                    WHERE status = 'responded'
                      AND response_text IS NOT NULL
                    ORDER BY id
                    LIMIT 1
                    """
                ).fetchone()

                if job is None:
                    connection.execute("COMMIT")
                    return None

                chunk_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM telegram_delivery_chunks
                        WHERE job_id = ?
                        """,
                        (job["id"],),
                    ).fetchone()["count"]
                )

                if chunk_count == 0:
                    chunks = split_telegram_text(
                        str(job["response_text"])
                    )
                    connection.executemany(
                        """
                        INSERT INTO telegram_delivery_chunks(
                            job_id,
                            chunk_index,
                            text
                        )
                        VALUES (?, ?, ?)
                        """,
                        [
                            (
                                job["id"],
                                chunk_index,
                                chunk_text,
                            )
                            for chunk_index, chunk_text
                            in enumerate(chunks)
                        ],
                    )

                row = connection.execute(
                    """
                    SELECT
                        chunk.id AS chunk_id,
                        chunk.job_id,
                        job.update_id,
                        job.chat_id,
                        job.user_id,
                        job.message_id
                            AS source_message_id,
                        chunk.chunk_index,
                        (
                            SELECT COUNT(*)
                            FROM telegram_delivery_chunks
                            WHERE job_id = chunk.job_id
                        ) AS chunk_count,
                        chunk.text,
                        chunk.status,
                        chunk.attempts,
                        CASE
                            WHEN chunk.chunk_index = 0
                                THEN job.message_id
                            ELSE NULL
                        END AS reply_to_message_id
                    FROM telegram_delivery_chunks AS chunk
                    JOIN telegram_jobs AS job
                      ON job.id = chunk.job_id
                    WHERE chunk.job_id = ?
                      AND chunk.status = 'pending'
                      AND NOT EXISTS (
                          SELECT 1
                          FROM telegram_delivery_chunks
                              AS previous
                          WHERE previous.job_id =
                                chunk.job_id
                            AND previous.chunk_index <
                                chunk.chunk_index
                            AND previous.status !=
                                'delivered'
                      )
                    ORDER BY chunk.chunk_index
                    LIMIT 1
                    """,
                    (job["id"],),
                ).fetchone()

                if row is None:
                    remaining = int(
                        connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM telegram_delivery_chunks
                            WHERE job_id = ?
                              AND status != 'delivered'
                            """,
                            (job["id"],),
                        ).fetchone()["count"]
                    )

                    if remaining == 0:
                        connection.execute(
                            """
                            UPDATE telegram_jobs
                            SET status = 'delivered',
                                completed_at = COALESCE(
                                    completed_at,
                                    strftime(
                                        '%Y-%m-%dT%H:%M:%fZ',
                                        'now'
                                    )
                                ),
                                updated_at = strftime(
                                    '%Y-%m-%dT%H:%M:%fZ',
                                    'now'
                                )
                            WHERE id = ?
                            """,
                            (job["id"],),
                        )
                        connection.execute("COMMIT")
                        return None

                    raise RuntimeError(
                        "Telegram delivery state is inconsistent "
                        f"for job {job['id']}"
                    )

                cursor = connection.execute(
                    """
                    UPDATE telegram_delivery_chunks
                    SET status = 'sending',
                        attempts = attempts + 1,
                        started_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        ),
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE id = ?
                      AND status = 'pending'
                    """,
                    (row["chunk_id"],),
                )

                if cursor.rowcount != 1:
                    connection.execute("ROLLBACK")
                    return None

                connection.execute(
                    """
                    UPDATE telegram_jobs
                    SET status = 'sending',
                        delivery_attempts =
                            delivery_attempts + 1,
                        delivery_started_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        ),
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        ),
                        last_error = NULL
                    WHERE id = ?
                      AND status = 'responded'
                    """,
                    (job["id"],),
                )

                claimed = connection.execute(
                    """
                    SELECT
                        chunk.id AS chunk_id,
                        chunk.job_id,
                        job.update_id,
                        job.chat_id,
                        job.user_id,
                        job.message_id
                            AS source_message_id,
                        chunk.chunk_index,
                        (
                            SELECT COUNT(*)
                            FROM telegram_delivery_chunks
                            WHERE job_id = chunk.job_id
                        ) AS chunk_count,
                        chunk.text,
                        chunk.status,
                        chunk.attempts,
                        CASE
                            WHEN chunk.chunk_index = 0
                                THEN job.message_id
                            ELSE NULL
                        END AS reply_to_message_id
                    FROM telegram_delivery_chunks AS chunk
                    JOIN telegram_jobs AS job
                      ON job.id = chunk.job_id
                    WHERE chunk.id = ?
                    """,
                    (row["chunk_id"],),
                ).fetchone()

                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

        return self._delivery_from_row(claimed)

    def _save_response_sync(
        self,
        job_id: int,
        response_text: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                job = connection.execute(
                    """
                    SELECT *
                    FROM telegram_jobs
                    WHERE id = ?
                    """,
                    (job_id,),
                ).fetchone()

                if job is None:
                    raise KeyError(
                        f"Telegram job not found: {job_id}"
                    )

                existing_response = job[
                    "response_text"
                ]

                if (
                    existing_response is not None
                    and job["status"]
                    in {
                        "responded",
                        "sending",
                        "delivered",
                    }
                ):
                    if existing_response != response_text:
                        raise ValueError(
                            "Telegram job "
                            f"{job_id} already has "
                            "a different response"
                        )

                    connection.execute("COMMIT")
                    return

                connection.execute(
                    """
                    INSERT OR IGNORE INTO
                        telegram_messages(
                            chat_id, stream_id,
                            role,
                            content,
                            source_update_id,
                            telegram_message_id
                        )
                    VALUES (
                        ?, ?,
                        'assistant',
                        ?,
                        ?,
                        NULL
                    )
                    """,
                    (
                        job["chat_id"],
                        job["stream_id"],
                        response_text,
                        job["update_id"],
                    ),
                )

                connection.execute(
                    """
                    UPDATE telegram_jobs
                    SET response_text = ?,
                        status = CASE
                            WHEN status = 'delivered'
                                THEN 'delivered'
                            ELSE 'responded'
                        END,
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE id = ?
                    """,
                    (
                        response_text,
                        job_id,
                    ),
                )

                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _mark_delivery_chunk_delivered_sync(
        self,
        chunk_id: int,
        telegram_message_id: int,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                row = connection.execute(
                    """
                    SELECT
                        chunk.*,
                        job.update_id
                    FROM telegram_delivery_chunks AS chunk
                    JOIN telegram_jobs AS job
                      ON job.id = chunk.job_id
                    WHERE chunk.id = ?
                    """,
                    (chunk_id,),
                ).fetchone()

                if row is None:
                    raise KeyError(
                        "Telegram delivery chunk not found: "
                        f"{chunk_id}"
                    )

                if row["status"] == "delivered":
                    stored_message_id = int(
                        row["telegram_message_id"]
                    )

                    if stored_message_id != telegram_message_id:
                        raise ValueError(
                            "Telegram delivery chunk "
                            f"{chunk_id} already has a "
                            "different Telegram message ID"
                        )

                    remaining = int(
                        connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM telegram_delivery_chunks
                            WHERE job_id = ?
                              AND status != 'delivered'
                            """,
                            (row["job_id"],),
                        ).fetchone()["count"]
                    )
                    connection.execute("COMMIT")
                    return remaining == 0

                if row["status"] != "sending":
                    raise ValueError(
                        "Telegram delivery chunk "
                        f"{chunk_id} is not being sent"
                    )

                connection.execute(
                    """
                    UPDATE telegram_delivery_chunks
                    SET status = 'delivered',
                        telegram_message_id = ?,
                        delivered_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        ),
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE id = ?
                      AND status = 'sending'
                    """,
                    (
                        telegram_message_id,
                        chunk_id,
                    ),
                )

                if int(row["chunk_index"]) == 0:
                    connection.execute(
                        """
                        UPDATE telegram_jobs
                        SET telegram_response_message_id =
                                COALESCE(
                                    telegram_response_message_id,
                                    ?
                                )
                        WHERE id = ?
                        """,
                        (
                            telegram_message_id,
                            row["job_id"],
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE telegram_messages
                        SET telegram_message_id = ?
                        WHERE source_update_id = ?
                          AND role = 'assistant'
                        """,
                        (
                            telegram_message_id,
                            row["update_id"],
                        ),
                    )

                remaining = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM telegram_delivery_chunks
                        WHERE job_id = ?
                          AND status != 'delivered'
                        """,
                        (row["job_id"],),
                    ).fetchone()["count"]
                )
                completed = remaining == 0

                if completed:
                    connection.execute(
                        """
                        UPDATE telegram_jobs
                        SET status = 'delivered',
                            delivery_started_at = NULL,
                            updated_at = strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            ),
                            completed_at = strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            )
                        WHERE id = ?
                        """,
                        (row["job_id"],),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE telegram_jobs
                        SET status = 'responded',
                            delivery_started_at = NULL,
                            updated_at = strftime(
                                '%Y-%m-%dT%H:%M:%fZ',
                                'now'
                            )
                        WHERE id = ?
                        """,
                        (row["job_id"],),
                    )

                connection.execute("COMMIT")
                return completed

            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _mark_delivered_sync(
        self,
        job_id: int,
        telegram_message_id: int,
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id
                FROM telegram_delivery_chunks
                WHERE job_id = ?
                  AND status IN (
                      'sending',
                      'delivered'
                  )
                ORDER BY
                    CASE status
                        WHEN 'sending' THEN 0
                        ELSE 1
                    END,
                    chunk_index
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()

        if row is None:
            raise ValueError(
                "Telegram job "
                f"{job_id} has no claimed delivery chunk"
            )

        self._mark_delivery_chunk_delivered_sync(
            int(row["id"]),
            telegram_message_id,
        )

    def _load_conversation_sync(
        self,
        chat_id: int,
        through_update_id: int | None,
    ) -> list[dict[str, str]]:
        query = """
            SELECT role, content
            FROM telegram_messages
            WHERE chat_id = ?
        """
        parameters: list[int] = [chat_id]

        if through_update_id is not None:
            query += """
                AND source_update_id <= ?
            """
            parameters.append(through_update_id)

        query += """
            ORDER BY
                source_update_id,
                CASE role
                    WHEN 'user' THEN 0
                    WHEN 'assistant' THEN 1
                    ELSE 2
                END,
                id
        """

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            {
                "role": str(row["role"]),
                "content": str(row["content"]),
            }
            for row in rows
        ]

    @staticmethod
    def _display_time(epoch: int | None, stored: str) -> str:
        if epoch is not None:
            value = datetime.fromtimestamp(epoch, timezone.utc)
        else:
            value = datetime.fromisoformat(stored.replace("Z", "+00:00"))
        return value.astimezone(ZoneInfo("Europe/Kyiv")).isoformat(timespec="seconds")

    def _build_generation_context_sync(self, job: TelegramJob) -> list[dict[str, str]]:
        batch_ids = job.batch_update_ids or (job.update_id,)
        cutoff = max(batch_ids)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT m.id, m.role, m.content, m.source_update_id,
                          m.telegram_message_id, m.created_at, m.telegram_date,
                          m.reply_to_message_id
                   FROM telegram_messages m
                   WHERE m.stream_id = ? AND m.source_update_id <= ?
                   ORDER BY m.id""",
                (job.stream_id, cutoff),
            ).fetchall()
            references = {}
            for row in rows:
                reply_id = row["reply_to_message_id"]
                if row["source_update_id"] in batch_ids and reply_id is not None:
                    referenced = connection.execute(
                        """SELECT m.id, m.role, m.content, m.source_update_id,
                                  m.telegram_message_id, m.created_at,
                                  m.telegram_date, m.reply_to_message_id
                           FROM telegram_messages AS m
                           WHERE m.stream_id = ?
                             AND (
                                 m.telegram_message_id = ?
                                 OR (
                                     m.role = 'assistant'
                                     AND EXISTS (
                                         SELECT 1
                                         FROM telegram_jobs AS j
                                         JOIN telegram_delivery_chunks AS c
                                           ON c.job_id = j.id
                                         WHERE j.update_id = m.source_update_id
                                           AND c.telegram_message_id = ?
                                     )
                                 )
                             )
                           ORDER BY m.id LIMIT 1""",
                        (job.stream_id, reply_id, reply_id),
                    ).fetchone()
                    if referenced is not None:
                        references[int(referenced["id"])] = referenced

        def project(row: sqlite3.Row, *, referenced: bool = False) -> dict[str, str]:
            timestamp = self._display_time(row["telegram_date"], row["created_at"])
            prefix = f"[Telegram time: {timestamp}]"
            if referenced:
                prefix = f"[Explicit reply reference; {prefix[1:]}"
            if row["reply_to_message_id"] is not None:
                prefix += f" [Replies to Telegram message {row['reply_to_message_id']}]"
            return {"role": str(row["role"]), "content": f"{prefix}\n{row['content']}"}

        current = [row for row in rows if row["source_update_id"] in batch_ids and row["role"] == "user"]
        ordinary = [row for row in rows if row not in current and int(row["id"]) not in references]
        selected = []
        used = sum(estimate_tokens_from_chars(len(project(row)["content"])) for row in current)
        used += sum(estimate_tokens_from_chars(len(project(row, referenced=True)["content"])) for row in references.values())
        for row in reversed(ordinary):
            cost = estimate_tokens_from_chars(len(project(row)["content"]))
            if used + cost <= self.exact_tail_token_budget:
                selected.append(row)
                used += cost
            else:
                break
        selected.reverse()
        included_ids = {int(row["id"]) for row in selected + current}
        output = [project(row) for row in selected]
        for row_id, row in sorted(references.items()):
            if row_id not in included_ids:
                output.append(project(row, referenced=True))
        for row in current:
            metadata = project(row)["content"].split("\n", 1)[0]
            output.append(
                {
                    "role": "system",
                    "content": f"Telegram transport metadata for the next user message: {metadata}",
                }
            )
            output.append({"role": "user", "content": str(row["content"])})
        return output

    def _recover_incomplete_jobs_sync(
        self,
    ) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                delivery_jobs = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM telegram_jobs
                        WHERE status = 'sending'
                        """
                    ).fetchone()["count"]
                )

                generation_cursor = connection.execute(
                    """
                    UPDATE telegram_jobs
                    SET status = 'queued',
                        started_at = NULL,
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE status = 'processing'
                    """
                )

                connection.execute(
                    """
                    UPDATE telegram_delivery_chunks
                    SET status = 'pending',
                        started_at = NULL,
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE status = 'sending'
                    """
                )

                connection.execute(
                    """
                    UPDATE telegram_jobs
                    SET status = 'responded',
                        delivery_started_at = NULL,
                        updated_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now'
                        )
                    WHERE status = 'sending'
                    """
                )

                recovered = (
                    int(generation_cursor.rowcount)
                    + delivery_jobs
                )

                connection.execute("COMMIT")
                return recovered

            except Exception:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _delivery_from_row(
        row: sqlite3.Row,
    ) -> TelegramDelivery:
        return TelegramDelivery(
            chunk_id=int(row["chunk_id"]),
            job_id=int(row["job_id"]),
            update_id=int(row["update_id"]),
            chat_id=int(row["chat_id"]),
            user_id=int(row["user_id"]),
            source_message_id=int(
                row["source_message_id"]
            ),
            chunk_index=int(row["chunk_index"]),
            chunk_count=int(row["chunk_count"]),
            text=str(row["text"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            reply_to_message_id=(
                int(row["reply_to_message_id"])
                if row["reply_to_message_id"]
                is not None
                else None
            ),
        )

    @staticmethod
    def _job_from_row(
        row: sqlite3.Row,
        batch_update_ids: tuple[int, ...] = (),
    ) -> TelegramJob:
        return TelegramJob(
            id=int(row["id"]),
            update_id=int(row["update_id"]),
            chat_id=int(row["chat_id"]),
            user_id=int(row["user_id"]),
            message_id=int(row["message_id"]),
            text=str(row["text"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            delivery_attempts=int(
                row["delivery_attempts"]
            ),
            response_text=(
                str(row["response_text"])
                if row["response_text"]
                is not None
                else None
            ),
            stream_id=(int(row["stream_id"]) if row["stream_id"] is not None else None),
            batch_update_ids=batch_update_ids,
        )
