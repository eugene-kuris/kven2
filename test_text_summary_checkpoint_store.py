import asyncio
import copy
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import context_window
import sqlite as kven_sqlite
import text_summary_checkpoint_store as checkpoint_store


class TextSummaryCheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = kven_sqlite.DB_PATH
        kven_sqlite.DB_PATH = os.path.join(
            self.temp_dir.name,
            "memory.db",
        )
        kven_sqlite._sync_init_db()

    def tearDown(self):
        kven_sqlite.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    @staticmethod
    def _messages(label: str, exchanges: int = 1) -> list:
        messages = [
            {
                "role": "system",
                "content": f"runtime system {label}",
            }
        ]
        for index in range(exchanges):
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"{label} question {index}",
                    },
                    {
                        "role": "assistant",
                        "content": f"{label} answer {index}",
                    },
                ]
            )
        return messages

    def _checkpoint(
        self,
        label: str,
        *,
        exchanges: int = 1,
        summary_text: str | None = None,
    ) -> dict:
        messages = self._messages(label, exchanges)
        return context_window.build_text_summary_checkpoint(
            messages,
            summary_text=(
                summary_text
                or f"Summary for {label}."
            ),
            summarized_prefix_end=len(messages),
        )

    def _row_count(self) -> int:
        conn = kven_sqlite.get_connection()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM text_summary_checkpoints"
            ).fetchone()[0]
        finally:
            conn.close()

    def test_round_trip_matches_extended_history(self):
        messages = self._messages("alpha")
        checkpoint = context_window.build_text_summary_checkpoint(
            messages,
            summary_text="Alpha summary.",
            summarized_prefix_end=len(messages),
        )

        self.assertTrue(
            checkpoint_store._sync_save_text_summary_checkpoint(
                checkpoint
            )
        )

        loaded = (
            checkpoint_store
            ._sync_load_text_summary_checkpoints(
                max_summarized_message_count=4,
            )
        )
        extended_messages = [
            {
                "role": "system",
                "content": "regenerated runtime system",
            },
            *messages[1:],
            {
                "role": "user",
                "content": "follow-up question",
            },
        ]
        matched, report = (
            context_window
            .find_matching_text_summary_checkpoint(
                extended_messages,
                loaded,
            )
        )

        self.assertEqual(len(loaded), 1)
        self.assertIsNotNone(matched)
        self.assertEqual(
            matched["checkpoint_id"],
            checkpoint["checkpoint_id"],
        )
        self.assertTrue(report["selected"])

    def test_same_prefix_replaces_previous_summary(self):
        messages = self._messages("replace")
        first = context_window.build_text_summary_checkpoint(
            messages,
            summary_text="First summary.",
            summarized_prefix_end=len(messages),
        )
        second = context_window.build_text_summary_checkpoint(
            messages,
            summary_text="Second summary.",
            summarized_prefix_end=len(messages),
        )

        self.assertTrue(
            checkpoint_store._sync_save_text_summary_checkpoint(
                first
            )
        )
        self.assertTrue(
            checkpoint_store._sync_save_text_summary_checkpoint(
                second
            )
        )

        loaded = (
            checkpoint_store
            ._sync_load_text_summary_checkpoints()
        )
        self.assertEqual(self._row_count(), 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0]["checkpoint_id"],
            second["checkpoint_id"],
        )
        self.assertEqual(
            loaded[0]["summary_text"],
            "Second summary.",
        )

    def test_store_limit_prunes_oldest_rows(self):
        checkpoints = [
            self._checkpoint(f"item-{index}")
            for index in range(4)
        ]

        for checkpoint in checkpoints:
            self.assertTrue(
                checkpoint_store
                ._sync_save_text_summary_checkpoint(
                    checkpoint,
                    max_checkpoints=2,
                )
            )

        loaded = (
            checkpoint_store
            ._sync_load_text_summary_checkpoints(
                limit=10,
            )
        )
        loaded_ids = {
            checkpoint["checkpoint_id"]
            for checkpoint in loaded
        }

        self.assertEqual(self._row_count(), 2)
        self.assertEqual(
            loaded_ids,
            {
                checkpoints[2]["checkpoint_id"],
                checkpoints[3]["checkpoint_id"],
            },
        )

    def test_load_filters_count_and_skips_corrupt_rows(self):
        short_checkpoint = self._checkpoint(
            "short",
            exchanges=1,
        )
        long_checkpoint = self._checkpoint(
            "long",
            exchanges=2,
        )
        self.assertTrue(
            checkpoint_store._sync_save_text_summary_checkpoint(
                short_checkpoint
            )
        )
        self.assertTrue(
            checkpoint_store._sync_save_text_summary_checkpoint(
                long_checkpoint
            )
        )

        conn = kven_sqlite.get_connection()
        try:
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
                """,
                (
                    "f" * 64,
                    "corrupt-version",
                    "corrupt-scope",
                    1,
                    "e" * 64,
                    "d" * 64,
                    1,
                    "{",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        loaded = (
            checkpoint_store
            ._sync_load_text_summary_checkpoints(
                max_summarized_message_count=2,
                limit=10,
            )
        )

        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0]["checkpoint_id"],
            short_checkpoint["checkpoint_id"],
        )

    def test_concurrent_saves_are_serialized(self):
        checkpoints = [
            self._checkpoint(f"parallel-{index}")
            for index in range(12)
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    checkpoint_store
                    ._sync_save_text_summary_checkpoint,
                    checkpoints,
                )
            )

        self.assertTrue(all(results))
        self.assertEqual(self._row_count(), 12)

    def test_async_api_marks_and_clears_checkpoint(self):
        checkpoint = self._checkpoint("async")

        async def scenario():
            saved = await (
                checkpoint_store
                .save_text_summary_checkpoint(checkpoint)
            )
            loaded = await (
                checkpoint_store
                .load_text_summary_checkpoints()
            )
            marked = await (
                checkpoint_store
                .mark_text_summary_checkpoint_used(
                    checkpoint["checkpoint_id"]
                )
            )
            cleared = await (
                checkpoint_store
                .clear_text_summary_checkpoints()
            )
            return saved, loaded, marked, cleared

        saved, loaded, marked, cleared = asyncio.run(
            scenario()
        )

        self.assertTrue(saved)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(marked)
        self.assertTrue(cleared)
        self.assertEqual(self._row_count(), 0)

    def test_connection_failures_are_fail_open(self):
        checkpoint = self._checkpoint("connection-failure")

        with mock.patch.object(
            kven_sqlite,
            "get_connection",
            side_effect=OSError("database unavailable"),
        ):
            self.assertFalse(
                checkpoint_store
                ._sync_save_text_summary_checkpoint(
                    checkpoint
                )
            )
            self.assertEqual(
                checkpoint_store
                ._sync_load_text_summary_checkpoints(),
                [],
            )
            self.assertFalse(
                checkpoint_store
                ._sync_mark_text_summary_checkpoint_used(
                    checkpoint["checkpoint_id"]
                )
            )
            self.assertFalse(
                checkpoint_store
                ._sync_clear_text_summary_checkpoints()
            )

    def test_invalid_checkpoint_is_rejected_without_db_change(self):
        checkpoint = self._checkpoint("invalid")
        corrupted = copy.deepcopy(checkpoint)
        corrupted["summary_chars"] += 1

        self.assertFalse(
            checkpoint_store._sync_save_text_summary_checkpoint(
                corrupted
            )
        )
        self.assertEqual(self._row_count(), 0)


if __name__ == "__main__":
    unittest.main()
