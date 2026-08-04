import tempfile
import unittest
from pathlib import Path

from telegram_store import TelegramStore


class TelegramStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "telegram.db"
        self.store = TelegramStore(str(self.db_path))
        await self.store.init()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def test_enqueue_is_idempotent_and_advances_offset(self):
        raw_update = {
            "update_id": 100,
            "message": {
                "message_id": 7,
                "text": "hello",
            },
        }

        inserted = await self.store.enqueue_text_update(
            update_id=100,
            chat_id=200,
            user_id=300,
            message_id=7,
            text="hello",
            raw_update=raw_update,
        )
        duplicate = await self.store.enqueue_text_update(
            update_id=100,
            chat_id=200,
            user_id=300,
            message_id=7,
            text="hello",
            raw_update=raw_update,
        )

        self.assertTrue(inserted)
        self.assertFalse(duplicate)
        self.assertEqual(
            await self.store.get_next_update_offset(),
            101,
        )
        self.assertEqual(
            await self.store.load_conversation(200),
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        )

    async def test_conversation_is_ordered_by_update_and_can_be_cut_off(
        self,
    ):
        await self.store.enqueue_text_update(
            update_id=10,
            chat_id=20,
            user_id=30,
            message_id=40,
            text="first question",
            raw_update={"update_id": 10},
        )
        await self.store.enqueue_text_update(
            update_id=11,
            chat_id=20,
            user_id=30,
            message_id=41,
            text="second question",
            raw_update={"update_id": 11},
        )

        first_job = await self.store.claim_next_job()
        self.assertIsNotNone(first_job)
        self.assertEqual(first_job.update_id, 10)

        await self.store.save_response(
            first_job.id,
            "first answer",
        )

        self.assertEqual(
            await self.store.load_conversation(
                20,
                through_update_id=10,
            ),
            [
                {
                    "role": "user",
                    "content": "first question",
                },
                {
                    "role": "assistant",
                    "content": "first answer",
                },
            ],
        )

        self.assertEqual(
            await self.store.load_conversation(20),
            [
                {
                    "role": "user",
                    "content": "first question",
                },
                {
                    "role": "assistant",
                    "content": "first answer",
                },
                {
                    "role": "user",
                    "content": "second question",
                },
            ],
        )

    async def test_job_lifecycle_persists_assistant_reply(self):
        await self.store.enqueue_text_update(
            update_id=10,
            chat_id=20,
            user_id=30,
            message_id=40,
            text="question",
            raw_update={"update_id": 10},
        )

        generation_job = await self.store.claim_next_job()

        self.assertIsNotNone(generation_job)
        self.assertEqual(generation_job.update_id, 10)
        self.assertEqual(generation_job.attempts, 1)
        self.assertIsNone(
            await self.store.claim_next_job()
        )

        await self.store.save_response(
            generation_job.id,
            "answer",
        )

        delivery_job = (
            await self.store.claim_next_delivery()
        )

        self.assertIsNotNone(delivery_job)
        self.assertEqual(
            delivery_job.id,
            generation_job.id,
        )
        self.assertEqual(
            delivery_job.response_text,
            "answer",
        )
        self.assertEqual(
            delivery_job.delivery_attempts,
            1,
        )
        self.assertIsNone(
            await self.store.claim_next_delivery()
        )

        await self.store.mark_delivered(
            delivery_job.id,
            50,
        )

        self.assertEqual(
            await self.store.load_conversation(20),
            [
                {
                    "role": "user",
                    "content": "question",
                },
                {
                    "role": "assistant",
                    "content": "answer",
                },
            ],
        )

    async def test_processing_job_is_recovered_after_restart(self):
        await self.store.enqueue_text_update(
            update_id=1,
            chat_id=2,
            user_id=3,
            message_id=4,
            text="retry generation",
            raw_update={"update_id": 1},
        )

        first_claim = await self.store.claim_next_job()

        self.assertIsNotNone(first_claim)
        self.assertEqual(first_claim.attempts, 1)

        recovered = (
            await self.store.recover_incomplete_jobs()
        )
        second_claim = await self.store.claim_next_job()

        self.assertEqual(recovered, 1)
        self.assertIsNotNone(second_claim)
        self.assertEqual(
            second_claim.id,
            first_claim.id,
        )
        self.assertEqual(
            second_claim.attempts,
            2,
        )

    async def test_sending_job_is_recovered_without_regeneration(
        self,
    ):
        await self.store.enqueue_text_update(
            update_id=1,
            chat_id=2,
            user_id=3,
            message_id=4,
            text="do not regenerate",
            raw_update={"update_id": 1},
        )

        generation_job = await self.store.claim_next_job()
        self.assertIsNotNone(generation_job)

        await self.store.save_response(
            generation_job.id,
            "saved answer",
        )

        first_delivery = (
            await self.store.claim_next_delivery()
        )

        self.assertIsNotNone(first_delivery)
        self.assertEqual(
            first_delivery.response_text,
            "saved answer",
        )
        self.assertEqual(
            first_delivery.delivery_attempts,
            1,
        )

        recovered = (
            await self.store.recover_incomplete_jobs()
        )
        second_delivery = (
            await self.store.claim_next_delivery()
        )

        self.assertEqual(recovered, 1)
        self.assertIsNotNone(second_delivery)
        self.assertEqual(
            second_delivery.id,
            first_delivery.id,
        )
        self.assertEqual(
            second_delivery.response_text,
            "saved answer",
        )
        self.assertEqual(
            second_delivery.delivery_attempts,
            2,
        )

        self.assertIsNone(
            await self.store.claim_next_job()
        )


if __name__ == "__main__":
    unittest.main()
