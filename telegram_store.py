from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


class TelegramStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

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
    ) -> bool:
        return await asyncio.to_thread(
            self._enqueue_text_update_sync,
            update_id,
            chat_id,
            user_id,
            message_id,
            text,
            raw_update,
        )

    async def claim_next_job(self) -> TelegramJob | None:
        return await asyncio.to_thread(
            self._claim_next_job_sync
        )

    async def claim_next_delivery(
        self,
    ) -> TelegramJob | None:
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

    async def mark_delivered(
        self,
        job_id: int,
        telegram_message_id: int,
    ) -> None:
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
                """
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
                        raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        update_id,
                        chat_id,
                        user_id,
                        message_id,
                        text,
                        raw_json,
                    ),
                )

                connection.execute(
                    """
                    INSERT INTO telegram_jobs(
                        update_id,
                        chat_id,
                        user_id,
                        message_id,
                        text
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        update_id,
                        chat_id,
                        user_id,
                        message_id,
                        text,
                    ),
                )

                connection.execute(
                    """
                    INSERT INTO telegram_messages(
                        chat_id,
                        role,
                        content,
                        source_update_id,
                        telegram_message_id
                    )
                    VALUES (
                        ?,
                        'user',
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        chat_id,
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
                    ORDER BY id
                    LIMIT 1
                    """
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

                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

        return self._job_from_row(claimed)

    def _claim_next_delivery_sync(
        self,
    ) -> TelegramJob | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                row = connection.execute(
                    """
                    SELECT *
                    FROM telegram_jobs
                    WHERE status = 'responded'
                    ORDER BY id
                    LIMIT 1
                    """
                ).fetchone()

                if row is None:
                    connection.execute("COMMIT")
                    return None

                cursor = connection.execute(
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
                    (row["id"],),
                )

                if cursor.rowcount != 1:
                    connection.execute("ROLLBACK")
                    return None

                claimed = connection.execute(
                    """
                    SELECT *
                    FROM telegram_jobs
                    WHERE id = ?
                    """,
                    (row["id"],),
                ).fetchone()

                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

        return self._job_from_row(claimed)

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
                    and existing_response
                    != response_text
                    and job["status"]
                    in {"responded", "delivered"}
                ):
                    raise ValueError(
                        "Telegram job "
                        f"{job_id} already has "
                        "a different response"
                    )

                connection.execute(
                    """
                    INSERT OR IGNORE INTO
                        telegram_messages(
                            chat_id,
                            role,
                            content,
                            source_update_id,
                            telegram_message_id
                        )
                    VALUES (
                        ?,
                        'assistant',
                        ?,
                        ?,
                        NULL
                    )
                    """,
                    (
                        job["chat_id"],
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

    def _mark_delivered_sync(
        self,
        job_id: int,
        telegram_message_id: int,
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

                if job["response_text"] is None:
                    raise ValueError(
                        "Telegram job "
                        f"{job_id} has no saved response"
                    )

                connection.execute(
                    """
                    UPDATE telegram_jobs
                    SET status = 'delivered',
                        telegram_response_message_id = ?,
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
                    (
                        telegram_message_id,
                        job_id,
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
                        job["update_id"],
                    ),
                )

                connection.execute("COMMIT")

            except Exception:
                connection.execute("ROLLBACK")
                raise

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

    def _recover_incomplete_jobs_sync(
        self,
    ) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
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

                delivery_cursor = connection.execute(
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
                    + int(delivery_cursor.rowcount)
                )

                connection.execute("COMMIT")
                return recovered

            except Exception:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _job_from_row(
        row: sqlite3.Row,
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
        )
