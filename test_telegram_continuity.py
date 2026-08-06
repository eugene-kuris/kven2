import sqlite3
import tempfile
import unittest
from pathlib import Path

from telegram_store import TelegramStore


class TelegramContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "telegram.db"
        self.store = TelegramStore(str(self.path), exact_tail_token_budget=80)
        await self.store.init()

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def enqueue(self, update_id, text, *, message_id=None, date=None, reply=None):
        return await self.store.enqueue_text_update(
            update_id=update_id,
            chat_id=200,
            user_id=300,
            message_id=message_id or update_id,
            text=text,
            raw_update={"update_id": update_id},
            message_date=date,
            reply_to_message_id=reply,
        )

    def rows(self, query, parameters=()):
        with sqlite3.connect(self.path) as connection:
            return connection.execute(query, parameters).fetchall()

    async def test_stream_is_reused_and_duplicate_is_append_idempotent(self):
        await self.enqueue(1, "one")
        await self.enqueue(2, "two")
        self.assertFalse(await self.enqueue(2, "two"))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_streams")[0][0], 1)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_messages")[0][0], 2)

    async def test_rapid_messages_form_one_ordered_durable_batch(self):
        await self.enqueue(1, "one")
        await self.enqueue(2, "two")
        job = await self.store.claim_next_job()
        self.assertEqual(job.batch_update_ids, (1, 2))
        self.assertEqual([m[0] for m in self.rows(
            "SELECT update_id FROM telegram_job_messages WHERE job_id=? ORDER BY ordinal", (job.id,)
        )], [1, 2])

    async def test_message_arriving_during_generation_is_next_batch(self):
        await self.enqueue(1, "first")
        active = await self.store.claim_next_job()
        await self.enqueue(2, "later")
        self.assertIsNone(await self.store.claim_next_job())
        await self.store.save_response(active.id, "answer")
        following = await self.store.claim_next_job()
        self.assertEqual(following.batch_update_ids, (2,))

    async def test_context_has_times_batch_boundaries_and_reply_reference(self):
        await self.enqueue(1, "old exact text", message_id=101, date=1_700_000_000)
        old_job = await self.store.claim_next_job()
        await self.store.save_response(old_job.id, "old answer")
        for number in range(2, 8):
            await self.enqueue(number, "x" * 90)
            job = await self.store.claim_next_job()
            await self.store.save_response(job.id, "y" * 90)
        await self.enqueue(8, "reply now", date=1_700_100_000, reply=101)
        job = await self.store.claim_next_job()
        context = await self.store.build_generation_context(job)
        joined = "\n".join(item["content"] for item in context)
        self.assertIn("old exact text", joined)
        self.assertIn("Explicit reply reference", joined)
        self.assertIn("2023-", joined)
        self.assertEqual(joined.count("old exact text"), 1)
        self.assertTrue(context[-1]["content"].endswith("reply now"))

    async def test_current_batch_is_never_trimmed_and_answers_are_whole(self):
        await self.enqueue(1, "historical")
        first = await self.store.claim_next_job()
        answer = "A" * 500
        await self.store.save_response(first.id, answer)
        await self.enqueue(2, "B" * 500)
        current = await self.store.claim_next_job()
        context = await self.store.build_generation_context(current)
        self.assertTrue(context[-1]["content"].endswith("B" * 500))
        contents = [item["content"] for item in context]
        self.assertFalse(any(answer in content and not content.endswith(answer) for content in contents))

    async def test_restart_migrates_legacy_schema_idempotently(self):
        await self.enqueue(1, "pending")
        restarted = TelegramStore(str(self.path))
        await restarted.init()
        await restarted.init()
        job = await restarted.claim_next_job()
        self.assertEqual(job.batch_update_ids, (1,))

    async def test_long_delivery_is_one_assistant_transcript_entry(self):
        await self.enqueue(1, "question")
        job = await self.store.claim_next_job()
        await self.store.save_response(job.id, "z" * 9000)
        while True:
            delivery = await self.store.claim_next_delivery()
            if delivery is None:
                break
            await self.store.mark_delivery_chunk_delivered(delivery.chunk_id, 500 + delivery.chunk_index)
        self.assertGreater(self.rows("SELECT COUNT(*) FROM telegram_delivery_chunks")[0][0], 1)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_messages WHERE role='assistant'")[0][0], 1)


if __name__ == "__main__":
    unittest.main()
