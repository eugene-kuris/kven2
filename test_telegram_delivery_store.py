import sqlite3
import tempfile
import unittest
from pathlib import Path

from telegram_bot_api import (
    split_telegram_text,
    telegram_text_units,
)
from telegram_store import (
    TelegramDelivery,
    TelegramStore,
)


class TelegramDeliveryStoreTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = (
            Path(self.temp_dir.name)
            / "telegram.db"
        )
        self.store = TelegramStore(
            str(self.db_path)
        )
        await self.store.init()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def enqueue_response(
        self,
        response_text: str,
        *,
        update_id: int = 10,
        chat_id: int = 20,
        message_id: int = 40,
    ) -> int:
        inserted = (
            await self.store.enqueue_text_update(
                update_id=update_id,
                chat_id=chat_id,
                user_id=30,
                message_id=message_id,
                text="question",
                raw_update={
                    "update_id": update_id,
                },
            )
        )
        self.assertTrue(inserted)

        job = await self.store.claim_next_job()
        self.assertIsNotNone(job)

        await self.store.save_response(
            job.id,
            response_text,
        )

        return job.id

    def read_job_state(
        self,
        job_id: int,
    ) -> sqlite3.Row:
        connection = sqlite3.connect(
            self.db_path
        )
        connection.row_factory = sqlite3.Row

        try:
            row = connection.execute(
                """
                SELECT
                    status,
                    telegram_response_message_id
                FROM telegram_jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise AssertionError(
                f"Missing job {job_id}"
            )

        return row

    def read_chunk_rows(
        self,
        job_id: int,
    ) -> list[sqlite3.Row]:
        connection = sqlite3.connect(
            self.db_path
        )
        connection.row_factory = sqlite3.Row

        try:
            rows = connection.execute(
                """
                SELECT
                    chunk_index,
                    text,
                    status,
                    attempts,
                    telegram_message_id
                FROM telegram_delivery_chunks
                WHERE job_id = ?
                ORDER BY chunk_index
                """,
                (job_id,),
            ).fetchall()
        finally:
            connection.close()

        return rows

    async def test_long_response_is_delivered_in_order(
        self,
    ):
        response_text = (
            ("A" * 3500)
            + "\n\n"
            + ("B" * 3500)
            + "\n\n"
            + ("C" * 3500)
        )
        expected_chunks = split_telegram_text(
            response_text
        )
        self.assertGreater(
            len(expected_chunks),
            1,
        )

        job_id = await self.enqueue_response(
            response_text
        )

        claimed_texts = []
        first_telegram_message_id = 500

        for index, expected_text in enumerate(
            expected_chunks
        ):
            delivery = (
                await self.store.claim_next_delivery()
            )

            self.assertIsInstance(
                delivery,
                TelegramDelivery,
            )
            self.assertEqual(
                delivery.job_id,
                job_id,
            )
            self.assertEqual(
                delivery.chunk_index,
                index,
            )
            self.assertEqual(
                delivery.chunk_count,
                len(expected_chunks),
            )
            self.assertEqual(
                delivery.text,
                expected_text,
            )
            self.assertEqual(
                delivery.attempts,
                1,
            )
            self.assertLessEqual(
                telegram_text_units(delivery.text),
                4000,
            )

            if index == 0:
                self.assertEqual(
                    delivery.reply_to_message_id,
                    40,
                )
            else:
                self.assertIsNone(
                    delivery.reply_to_message_id
                )

            claimed_texts.append(delivery.text)

            completed = (
                await self.store
                .mark_delivery_chunk_delivered(
                    delivery.chunk_id,
                    first_telegram_message_id
                    + index,
                )
            )

            self.assertEqual(
                completed,
                index
                == len(expected_chunks) - 1,
            )

        self.assertEqual(
            "".join(claimed_texts),
            response_text,
        )
        self.assertIsNone(
            await self.store.claim_next_delivery()
        )

        job_state = self.read_job_state(job_id)

        self.assertEqual(
            job_state["status"],
            "delivered",
        )
        self.assertEqual(
            job_state[
                "telegram_response_message_id"
            ],
            first_telegram_message_id,
        )

        chunk_rows = self.read_chunk_rows(job_id)

        self.assertEqual(
            [row["text"] for row in chunk_rows],
            expected_chunks,
        )
        self.assertTrue(
            all(
                row["status"] == "delivered"
                for row in chunk_rows
            )
        )

    async def test_recovery_skips_confirmed_chunks(
        self,
    ):
        response_text = "\n\n".join(
            [
                "A" * 3000,
                "B" * 3000,
                "C" * 3000,
            ]
        )
        expected_chunks = split_telegram_text(
            response_text
        )
        self.assertGreaterEqual(
            len(expected_chunks),
            3,
        )

        await self.enqueue_response(response_text)

        first = (
            await self.store.claim_next_delivery()
        )
        self.assertEqual(first.chunk_index, 0)

        completed = (
            await self.store
            .mark_delivery_chunk_delivered(
                first.chunk_id,
                500,
            )
        )
        self.assertFalse(completed)

        interrupted = (
            await self.store.claim_next_delivery()
        )
        self.assertEqual(
            interrupted.chunk_index,
            1,
        )
        self.assertEqual(
            interrupted.attempts,
            1,
        )

        recovered = (
            await self.store.recover_incomplete_jobs()
        )
        self.assertEqual(recovered, 1)

        resumed = (
            await self.store.claim_next_delivery()
        )

        self.assertEqual(
            resumed.chunk_id,
            interrupted.chunk_id,
        )
        self.assertEqual(
            resumed.chunk_index,
            1,
        )
        self.assertEqual(
            resumed.attempts,
            2,
        )
        self.assertIsNone(
            resumed.reply_to_message_id
        )

        await self.store.mark_delivery_chunk_delivered(
            resumed.chunk_id,
            501,
        )

        following = (
            await self.store.claim_next_delivery()
        )
        self.assertEqual(
            following.chunk_index,
            2,
        )

    async def test_delivery_mark_is_idempotent(
        self,
    ):
        job_id = await self.enqueue_response(
            "short answer"
        )

        delivery = (
            await self.store.claim_next_delivery()
        )

        completed = (
            await self.store
            .mark_delivery_chunk_delivered(
                delivery.chunk_id,
                500,
            )
        )
        repeated = (
            await self.store
            .mark_delivery_chunk_delivered(
                delivery.chunk_id,
                500,
            )
        )

        self.assertTrue(completed)
        self.assertTrue(repeated)

        with self.assertRaises(ValueError):
            await self.store \
                .mark_delivery_chunk_delivered(
                    delivery.chunk_id,
                    501,
                )

        self.assertEqual(
            self.read_job_state(job_id)["status"],
            "delivered",
        )

    async def test_response_cannot_change_after_delivery_started(
        self,
    ):
        job_id = await self.enqueue_response(
            ("A" * 3500)
            + "\n\n"
            + ("B" * 3500)
        )

        delivery = (
            await self.store.claim_next_delivery()
        )
        self.assertIsNotNone(delivery)

        with self.assertRaises(ValueError):
            await self.store.save_response(
                job_id,
                "different answer",
            )

        chunk_rows = self.read_chunk_rows(job_id)

        self.assertGreater(
            len(chunk_rows),
            1,
        )
        self.assertEqual(
            chunk_rows[0]["status"],
            "sending",
        )


if __name__ == "__main__":
    unittest.main()
