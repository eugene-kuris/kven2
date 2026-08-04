import unittest
from typing import Any

from telegram_store import TelegramDelivery
from telegram_workers import run_delivery_once


def make_delivery(
    *,
    chunk_id: int = 11,
    job_id: int = 12,
    chat_id: int = 13,
    source_message_id: int = 14,
    chunk_index: int = 0,
    chunk_count: int = 1,
    text: str = "answer",
    attempts: int = 1,
    reply_to_message_id: int | None = 14,
) -> TelegramDelivery:
    return TelegramDelivery(
        chunk_id=chunk_id,
        job_id=job_id,
        update_id=15,
        chat_id=chat_id,
        user_id=16,
        source_message_id=source_message_id,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        text=text,
        status="sending",
        attempts=attempts,
        reply_to_message_id=reply_to_message_id,
    )


class FakeDeliveryStore:
    def __init__(
        self,
        *,
        deliveries: list[TelegramDelivery] | None = None,
        mark_error: Exception | None = None,
        completed: bool = False,
        events: list[str] | None = None,
    ):
        self.deliveries = list(deliveries or [])
        self.mark_error = mark_error
        self.completed = completed
        self.events = events
        self.claim_calls = 0
        self.mark_calls: list[tuple[int, int]] = []

    async def claim_next_delivery(
        self,
    ) -> TelegramDelivery | None:
        self.claim_calls += 1

        if not self.deliveries:
            return None

        return self.deliveries.pop(0)

    async def mark_delivery_chunk_delivered(
        self,
        chunk_id: int,
        telegram_message_id: int,
    ) -> bool:
        if self.events is not None:
            self.events.append("mark")

        if self.mark_error is not None:
            raise self.mark_error

        self.mark_calls.append(
            (
                chunk_id,
                telegram_message_id,
            )
        )

        return self.completed


class FakeTelegramBot:
    def __init__(
        self,
        *,
        message_id: int = 500,
        send_error: Exception | None = None,
        events: list[str] | None = None,
    ):
        self.message_id = message_id
        self.send_error = send_error
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> int:
        if self.events is not None:
            self.events.append("send")

        self.calls.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": (
                    reply_to_message_id
                ),
            }
        )

        if self.send_error is not None:
            raise self.send_error

        return self.message_id


class TelegramDeliveryWorkerTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_no_pending_delivery_does_nothing(
        self,
    ):
        store = FakeDeliveryStore()
        bot = FakeTelegramBot()

        processed = await run_delivery_once(
            store,
            bot,
        )

        self.assertFalse(processed)
        self.assertEqual(store.claim_calls, 1)
        self.assertEqual(store.mark_calls, [])
        self.assertEqual(bot.calls, [])

    async def test_chunk_is_sent_and_confirmed(
        self,
    ):
        events: list[str] = []
        delivery = make_delivery(
            chunk_id=21,
            chat_id=22,
            text="generated answer",
            reply_to_message_id=23,
        )
        store = FakeDeliveryStore(
            deliveries=[delivery],
            completed=True,
            events=events,
        )
        bot = FakeTelegramBot(
            message_id=501,
            events=events,
        )

        processed = await run_delivery_once(
            store,
            bot,
        )

        self.assertTrue(processed)
        self.assertEqual(
            bot.calls,
            [
                {
                    "chat_id": 22,
                    "text": "generated answer",
                    "reply_to_message_id": 23,
                }
            ],
        )
        self.assertEqual(
            store.mark_calls,
            [
                (
                    21,
                    501,
                )
            ],
        )
        self.assertEqual(
            events,
            [
                "send",
                "mark",
            ],
        )

    async def test_later_chunk_is_not_a_reply(
        self,
    ):
        delivery = make_delivery(
            chunk_id=31,
            chunk_index=1,
            chunk_count=3,
            text="second part",
            reply_to_message_id=None,
        )
        store = FakeDeliveryStore(
            deliveries=[delivery],
        )
        bot = FakeTelegramBot(
            message_id=601,
        )

        processed = await run_delivery_once(
            store,
            bot,
        )

        self.assertTrue(processed)
        self.assertEqual(
            bot.calls[0][
                "reply_to_message_id"
            ],
            None,
        )
        self.assertEqual(
            store.mark_calls,
            [
                (
                    31,
                    601,
                )
            ],
        )

    async def test_only_one_chunk_is_processed(
        self,
    ):
        first = make_delivery(
            chunk_id=41,
            chunk_index=0,
            chunk_count=2,
            text="first",
        )
        second = make_delivery(
            chunk_id=42,
            chunk_index=1,
            chunk_count=2,
            text="second",
            reply_to_message_id=None,
        )
        store = FakeDeliveryStore(
            deliveries=[
                first,
                second,
            ]
        )
        bot = FakeTelegramBot(
            message_id=701,
        )

        processed = await run_delivery_once(
            store,
            bot,
        )

        self.assertTrue(processed)
        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(
            store.mark_calls,
            [
                (
                    41,
                    701,
                )
            ],
        )
        self.assertEqual(
            [
                delivery.chunk_id
                for delivery in store.deliveries
            ],
            [42],
        )

    async def test_send_failure_is_not_confirmed(
        self,
    ):
        store = FakeDeliveryStore(
            deliveries=[make_delivery()]
        )
        bot = FakeTelegramBot(
            send_error=RuntimeError(
                "Telegram unavailable"
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Telegram unavailable",
        ):
            await run_delivery_once(
                store,
                bot,
            )

        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(store.mark_calls, [])

    async def test_confirmation_failure_is_propagated(
        self,
    ):
        events: list[str] = []
        store = FakeDeliveryStore(
            deliveries=[make_delivery()],
            mark_error=RuntimeError(
                "database unavailable"
            ),
            events=events,
        )
        bot = FakeTelegramBot(
            message_id=801,
            events=events,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "database unavailable",
        ):
            await run_delivery_once(
                store,
                bot,
            )

        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(store.mark_calls, [])
        self.assertEqual(
            events,
            [
                "send",
                "mark",
            ],
        )


if __name__ == "__main__":
    unittest.main()
