import unittest
from typing import Any

from telegram_store import TelegramJob
from telegram_workers import run_generation_once


def make_job(
    *,
    job_id: int = 1,
    update_id: int = 10,
    chat_id: int = 20,
) -> TelegramJob:
    return TelegramJob(
        id=job_id,
        update_id=update_id,
        chat_id=chat_id,
        user_id=30,
        message_id=40,
        text="question",
        status="processing",
        attempts=1,
        delivery_attempts=0,
        response_text=None,
    )


class FakeStore:
    def __init__(
        self,
        *,
        jobs: list[TelegramJob] | None = None,
        conversation: list[dict[str, str]] | None = None,
        load_error: Exception | None = None,
        save_error: Exception | None = None,
    ):
        self.jobs = list(jobs or [])
        self.conversation = list(
            conversation
            or [
                {
                    "role": "user",
                    "content": "question",
                }
            ]
        )
        self.load_error = load_error
        self.save_error = save_error
        self.claim_calls = 0
        self.load_calls: list[dict[str, Any]] = []
        self.save_calls: list[tuple[int, str]] = []

    async def claim_next_job(
        self,
    ) -> TelegramJob | None:
        self.claim_calls += 1

        if not self.jobs:
            return None

        return self.jobs.pop(0)

    async def load_conversation(
        self,
        chat_id: int,
        *,
        through_update_id: int | None = None,
    ) -> list[dict[str, str]]:
        self.load_calls.append(
            {
                "chat_id": chat_id,
                "through_update_id": through_update_id,
            }
        )

        if self.load_error is not None:
            raise self.load_error

        return list(self.conversation)

    async def save_response(
        self,
        job_id: int,
        response_text: str,
    ) -> None:
        if self.save_error is not None:
            raise self.save_error

        self.save_calls.append(
            (
                job_id,
                response_text,
            )
        )


class FakeKvenClient:
    def __init__(
        self,
        *,
        answer: str = "answer",
        error: Exception | None = None,
    ):
        self.answer = answer
        self.error = error
        self.calls: list[
            list[dict[str, str]]
        ] = []

    async def generate_reply(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        self.calls.append(list(messages))

        if self.error is not None:
            raise self.error

        return self.answer


class TelegramGenerationWorkerTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_no_queued_job_does_nothing(self):
        store = FakeStore()
        client = FakeKvenClient()

        processed = await run_generation_once(
            store,
            client,
        )

        self.assertFalse(processed)
        self.assertEqual(store.claim_calls, 1)
        self.assertEqual(store.load_calls, [])
        self.assertEqual(store.save_calls, [])
        self.assertEqual(client.calls, [])

    async def test_generation_uses_history_through_job_update(
        self,
    ):
        job = make_job(
            job_id=5,
            update_id=101,
            chat_id=202,
        )
        conversation = [
            {
                "role": "user",
                "content": "first",
            },
            {
                "role": "assistant",
                "content": "reply",
            },
            {
                "role": "user",
                "content": "second",
            },
        ]
        store = FakeStore(
            jobs=[job],
            conversation=conversation,
        )
        client = FakeKvenClient(
            answer="generated answer",
        )

        processed = await run_generation_once(
            store,
            client,
        )

        self.assertTrue(processed)
        self.assertEqual(
            store.load_calls,
            [
                {
                    "chat_id": 202,
                    "through_update_id": 101,
                }
            ],
        )
        self.assertEqual(
            client.calls,
            [conversation],
        )
        self.assertEqual(
            store.save_calls,
            [
                (
                    5,
                    "generated answer",
                )
            ],
        )

    async def test_only_one_job_is_processed(self):
        first = make_job(
            job_id=1,
            update_id=10,
        )
        second = make_job(
            job_id=2,
            update_id=11,
        )
        store = FakeStore(
            jobs=[
                first,
                second,
            ]
        )
        client = FakeKvenClient()

        processed = await run_generation_once(
            store,
            client,
        )

        self.assertTrue(processed)
        self.assertEqual(
            store.save_calls,
            [
                (
                    1,
                    "answer",
                )
            ],
        )
        self.assertEqual(
            [job.id for job in store.jobs],
            [2],
        )

    async def test_generation_failure_is_not_saved(
        self,
    ):
        store = FakeStore(
            jobs=[make_job()]
        )
        client = FakeKvenClient(
            error=RuntimeError(
                "model unavailable"
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "model unavailable",
        ):
            await run_generation_once(
                store,
                client,
            )

        self.assertEqual(
            len(client.calls),
            1,
        )
        self.assertEqual(
            store.save_calls,
            [],
        )

    async def test_history_failure_skips_generation(
        self,
    ):
        store = FakeStore(
            jobs=[make_job()],
            load_error=RuntimeError(
                "database unavailable"
            ),
        )
        client = FakeKvenClient()

        with self.assertRaisesRegex(
            RuntimeError,
            "database unavailable",
        ):
            await run_generation_once(
                store,
                client,
            )

        self.assertEqual(
            client.calls,
            [],
        )
        self.assertEqual(
            store.save_calls,
            [],
        )

    async def test_save_failure_is_propagated(
        self,
    ):
        store = FakeStore(
            jobs=[make_job()],
            save_error=RuntimeError(
                "write failed"
            ),
        )
        client = FakeKvenClient(
            answer="generated",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "write failed",
        ):
            await run_generation_once(
                store,
                client,
            )

        self.assertEqual(
            client.calls,
            [
                [
                    {
                        "role": "user",
                        "content": "question",
                    }
                ]
            ],
        )
        self.assertEqual(
            store.save_calls,
            [],
        )


if __name__ == "__main__":
    unittest.main()
