import asyncio
import unittest
from typing import Any

from telegram_runtime import TelegramGatewayRuntime


class FakeStore:
    def __init__(
        self,
        events: list[str],
        *,
        init_error: Exception | None = None,
        recovered: int = 0,
    ):
        self.events = events
        self.init_error = init_error
        self.recovered = recovered
        self.init_calls = 0
        self.recover_calls = 0

    async def init(self) -> None:
        self.init_calls += 1
        self.events.append("store.init")

        if self.init_error is not None:
            raise self.init_error

    async def recover_incomplete_jobs(self) -> int:
        self.recover_calls += 1
        self.events.append("store.recover")
        return self.recovered


class FakeCloseable:
    def __init__(
        self,
        name: str,
        events: list[str],
    ):
        self.name = name
        self.events = events
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.events.append(
            f"{self.name}.close"
        )


class TelegramGatewayRuntimeTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_startup_precedes_all_workers(
        self,
    ):
        events: list[str] = []
        store = FakeStore(
            events,
            recovered=2,
        )
        bot = FakeCloseable(
            "bot",
            events,
        )
        kven = FakeCloseable(
            "kven",
            events,
        )
        stop_event = asyncio.Event()
        blocker = asyncio.Event()
        seen: set[str] = set()
        calls: dict[str, Any] = {}

        async def record(
            name: str,
            call: dict[str, Any],
        ) -> bool:
            self.assertEqual(
                events[:2],
                [
                    "store.init",
                    "store.recover",
                ],
            )

            events.append(name)
            calls[name] = call
            seen.add(name)

            if seen == {
                "poll",
                "generation",
                "delivery",
            }:
                stop_event.set()

            await blocker.wait()
            return True

        async def polling_runner(
            received_store: Any,
            received_bot: Any,
            *,
            allowed_user_id: int,
            timeout: int,
        ) -> int:
            return int(
                await record(
                    "poll",
                    {
                        "store": received_store,
                        "bot": received_bot,
                        "allowed_user_id": (
                            allowed_user_id
                        ),
                        "timeout": timeout,
                    },
                )
            )

        async def generation_runner(
            received_store: Any,
            received_kven: Any,
        ) -> bool:
            return await record(
                "generation",
                {
                    "store": received_store,
                    "kven": received_kven,
                },
            )

        async def delivery_runner(
            received_store: Any,
            received_bot: Any,
        ) -> bool:
            return await record(
                "delivery",
                {
                    "store": received_store,
                    "bot": received_bot,
                },
            )

        runtime = TelegramGatewayRuntime(
            store=store,
            telegram_bot=bot,
            kven_client=kven,
            allowed_user_id=12345,
            polling_timeout=37,
            polling_runner=polling_runner,
            generation_runner=generation_runner,
            delivery_runner=delivery_runner,
        )

        await asyncio.wait_for(
            runtime.run(stop_event),
            timeout=1.0,
        )

        self.assertEqual(store.init_calls, 1)
        self.assertEqual(store.recover_calls, 1)
        self.assertEqual(
            seen,
            {
                "poll",
                "generation",
                "delivery",
            },
        )

        self.assertIs(
            calls["poll"]["store"],
            store,
        )
        self.assertIs(
            calls["poll"]["bot"],
            bot,
        )
        self.assertEqual(
            calls["poll"]["allowed_user_id"],
            12345,
        )
        self.assertEqual(
            calls["poll"]["timeout"],
            37,
        )
        self.assertIs(
            calls["generation"]["kven"],
            kven,
        )
        self.assertIs(
            calls["delivery"]["bot"],
            bot,
        )

        self.assertEqual(bot.close_calls, 1)
        self.assertEqual(kven.close_calls, 1)
        self.assertIn("bot.close", events)
        self.assertIn("kven.close", events)

    async def test_startup_failure_prevents_workers(
        self,
    ):
        events: list[str] = []
        store = FakeStore(
            events,
            init_error=RuntimeError(
                "database unavailable"
            ),
        )
        bot = FakeCloseable(
            "bot",
            events,
        )
        kven = FakeCloseable(
            "kven",
            events,
        )
        worker_calls: list[str] = []

        async def unexpected(*args, **kwargs):
            worker_calls.append("called")
            return False

        runtime = TelegramGatewayRuntime(
            store=store,
            telegram_bot=bot,
            kven_client=kven,
            allowed_user_id=12345,
            polling_runner=unexpected,
            generation_runner=unexpected,
            delivery_runner=unexpected,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "database unavailable",
        ):
            await runtime.run(
                asyncio.Event()
            )

        self.assertEqual(store.init_calls, 1)
        self.assertEqual(store.recover_calls, 0)
        self.assertEqual(worker_calls, [])
        self.assertEqual(bot.close_calls, 1)
        self.assertEqual(kven.close_calls, 1)

    async def test_worker_error_uses_error_delay_and_retries(
        self,
    ):
        events: list[str] = []
        store = FakeStore(events)
        bot = FakeCloseable(
            "bot",
            events,
        )
        kven = FakeCloseable(
            "kven",
            events,
        )
        stop_event = asyncio.Event()
        blocker = asyncio.Event()
        delays: list[float] = []
        generation_calls = 0

        async def fake_sleep(
            delay: float,
        ) -> None:
            delays.append(delay)
            await asyncio.sleep(0)

        async def blocking_poll(
            *args,
            **kwargs,
        ) -> int:
            await blocker.wait()
            return 0

        async def blocking_delivery(
            *args,
            **kwargs,
        ) -> bool:
            await blocker.wait()
            return False

        async def generation_runner(
            *args,
            **kwargs,
        ) -> bool:
            nonlocal generation_calls
            generation_calls += 1

            if generation_calls == 1:
                raise RuntimeError(
                    "model unavailable"
                )

            stop_event.set()
            await blocker.wait()
            return True

        runtime = TelegramGatewayRuntime(
            store=store,
            telegram_bot=bot,
            kven_client=kven,
            allowed_user_id=12345,
            error_delay=7.5,
            sleeper=fake_sleep,
            polling_runner=blocking_poll,
            generation_runner=generation_runner,
            delivery_runner=blocking_delivery,
        )

        await asyncio.wait_for(
            runtime.run(stop_event),
            timeout=1.0,
        )

        self.assertEqual(generation_calls, 2)
        self.assertEqual(delays, [7.5])

    async def test_idle_worker_uses_idle_delay(
        self,
    ):
        events: list[str] = []
        store = FakeStore(events)
        bot = FakeCloseable(
            "bot",
            events,
        )
        kven = FakeCloseable(
            "kven",
            events,
        )
        stop_event = asyncio.Event()
        blocker = asyncio.Event()
        delays: list[float] = []
        generation_calls = 0

        async def fake_sleep(
            delay: float,
        ) -> None:
            delays.append(delay)
            await asyncio.sleep(0)

        async def blocking_poll(
            *args,
            **kwargs,
        ) -> int:
            await blocker.wait()
            return 0

        async def blocking_delivery(
            *args,
            **kwargs,
        ) -> bool:
            await blocker.wait()
            return False

        async def generation_runner(
            *args,
            **kwargs,
        ) -> bool:
            nonlocal generation_calls
            generation_calls += 1

            if generation_calls == 1:
                return False

            stop_event.set()
            await blocker.wait()
            return True

        runtime = TelegramGatewayRuntime(
            store=store,
            telegram_bot=bot,
            kven_client=kven,
            allowed_user_id=12345,
            idle_delay=0.125,
            sleeper=fake_sleep,
            polling_runner=blocking_poll,
            generation_runner=generation_runner,
            delivery_runner=blocking_delivery,
        )

        await asyncio.wait_for(
            runtime.run(stop_event),
            timeout=1.0,
        )

        self.assertEqual(generation_calls, 2)
        self.assertEqual(delays, [0.125])

    async def test_invalid_configuration_is_rejected(
        self,
    ):
        events: list[str] = []
        store = FakeStore(events)
        bot = FakeCloseable(
            "bot",
            events,
        )
        kven = FakeCloseable(
            "kven",
            events,
        )

        invalid_cases = [
            {
                "allowed_user_id": 0,
            },
            {
                "allowed_user_id": True,
            },
            {
                "allowed_user_id": 123,
                "polling_timeout": 0,
            },
            {
                "allowed_user_id": 123,
                "idle_delay": -1,
            },
            {
                "allowed_user_id": 123,
                "error_delay": 0,
            },
        ]

        for overrides in invalid_cases:
            with self.subTest(
                overrides=overrides
            ):
                arguments = {
                    "store": store,
                    "telegram_bot": bot,
                    "kven_client": kven,
                    "allowed_user_id": 123,
                }
                arguments.update(overrides)

                with self.assertRaises(
                    ValueError
                ):
                    TelegramGatewayRuntime(
                        **arguments
                    )


if __name__ == "__main__":
    unittest.main()
