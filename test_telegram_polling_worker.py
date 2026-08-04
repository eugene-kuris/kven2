import unittest
from typing import Any

from telegram_workers import run_polling_once


class FakePollingStore:
    def __init__(
        self,
        *,
        offset: int = 0,
        error: Exception | None = None,
    ):
        self.offset = offset
        self.error = error
        self.calls = 0

    async def get_next_update_offset(self) -> int:
        self.calls += 1

        if self.error is not None:
            raise self.error

        return self.offset


class FakeTelegramPollingBot:
    def __init__(
        self,
        *,
        updates: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ):
        self.updates = list(updates or [])
        self.error = error
        self.calls: list[dict[str, int]] = []

    async def get_updates(
        self,
        *,
        offset: int,
        timeout: int = 50,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "offset": offset,
                "timeout": timeout,
            }
        )

        if self.error is not None:
            raise self.error

        return list(self.updates)


class FakeUpdateIngestor:
    def __init__(
        self,
        *,
        error_on_update_id: int | None = None,
    ):
        self.error_on_update_id = (
            error_on_update_id
        )
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        store: Any,
        update: dict[str, Any],
        *,
        allowed_user_id: int,
    ) -> bool:
        self.calls.append(
            {
                "store": store,
                "update": update,
                "allowed_user_id": (
                    allowed_user_id
                ),
            }
        )

        if (
            update.get("update_id")
            == self.error_on_update_id
        ):
            raise RuntimeError(
                "update ingestion failed"
            )

        return True


class TelegramPollingWorkerTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_empty_poll_does_nothing(
        self,
    ):
        store = FakePollingStore(offset=101)
        bot = FakeTelegramPollingBot()
        ingestor = FakeUpdateIngestor()

        processed = await run_polling_once(
            store,
            bot,
            allowed_user_id=202,
            timeout=35,
            update_ingestor=ingestor,
        )

        self.assertEqual(processed, 0)
        self.assertEqual(store.calls, 1)
        self.assertEqual(
            bot.calls,
            [
                {
                    "offset": 101,
                    "timeout": 35,
                }
            ],
        )
        self.assertEqual(ingestor.calls, [])

    async def test_batch_is_ingested_in_order(
        self,
    ):
        updates = [
            {
                "update_id": 110,
                "message": {
                    "text": "first",
                },
            },
            {
                "update_id": 111,
                "message": {
                    "text": "second",
                },
            },
        ]
        store = FakePollingStore(offset=110)
        bot = FakeTelegramPollingBot(
            updates=updates
        )
        ingestor = FakeUpdateIngestor()

        processed = await run_polling_once(
            store,
            bot,
            allowed_user_id=303,
            timeout=50,
            update_ingestor=ingestor,
        )

        self.assertEqual(processed, 2)
        self.assertEqual(
            [
                call["update"]["update_id"]
                for call in ingestor.calls
            ],
            [
                110,
                111,
            ],
        )
        self.assertTrue(
            all(
                call["store"] is store
                for call in ingestor.calls
            )
        )
        self.assertTrue(
            all(
                call["allowed_user_id"] == 303
                for call in ingestor.calls
            )
        )

    async def test_store_failure_skips_poll(
        self,
    ):
        store = FakePollingStore(
            error=RuntimeError(
                "offset database unavailable"
            )
        )
        bot = FakeTelegramPollingBot()
        ingestor = FakeUpdateIngestor()

        with self.assertRaisesRegex(
            RuntimeError,
            "offset database unavailable",
        ):
            await run_polling_once(
                store,
                bot,
                allowed_user_id=404,
                update_ingestor=ingestor,
            )

        self.assertEqual(bot.calls, [])
        self.assertEqual(ingestor.calls, [])

    async def test_poll_failure_skips_ingestion(
        self,
    ):
        store = FakePollingStore(offset=10)
        bot = FakeTelegramPollingBot(
            error=RuntimeError(
                "Telegram unavailable"
            )
        )
        ingestor = FakeUpdateIngestor()

        with self.assertRaisesRegex(
            RuntimeError,
            "Telegram unavailable",
        ):
            await run_polling_once(
                store,
                bot,
                allowed_user_id=505,
                update_ingestor=ingestor,
            )

        self.assertEqual(ingestor.calls, [])

    async def test_ingestion_failure_stops_batch(
        self,
    ):
        updates = [
            {
                "update_id": 200,
            },
            {
                "update_id": 201,
            },
            {
                "update_id": 202,
            },
        ]
        store = FakePollingStore(offset=200)
        bot = FakeTelegramPollingBot(
            updates=updates
        )
        ingestor = FakeUpdateIngestor(
            error_on_update_id=201
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "update ingestion failed",
        ):
            await run_polling_once(
                store,
                bot,
                allowed_user_id=606,
                update_ingestor=ingestor,
            )

        self.assertEqual(
            [
                call["update"]["update_id"]
                for call in ingestor.calls
            ],
            [
                200,
                201,
            ],
        )

    async def test_invalid_allowed_user_is_rejected(
        self,
    ):
        for invalid_value in (
            0,
            -1,
            True,
            "123",
        ):
            with self.subTest(
                invalid_value=invalid_value
            ):
                store = FakePollingStore(
                    offset=10
                )
                bot = FakeTelegramPollingBot()
                ingestor = FakeUpdateIngestor()

                with self.assertRaises(
                    ValueError
                ):
                    await run_polling_once(
                        store,
                        bot,
                        allowed_user_id=(
                            invalid_value
                        ),
                        update_ingestor=(
                            ingestor
                        ),
                    )

                self.assertEqual(
                    store.calls,
                    0,
                )
                self.assertEqual(
                    bot.calls,
                    [],
                )
                self.assertEqual(
                    ingestor.calls,
                    [],
                )


if __name__ == "__main__":
    unittest.main()
