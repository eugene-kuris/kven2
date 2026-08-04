from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Protocol

from telegram_workers import (
    run_delivery_once,
    run_generation_once,
    run_polling_once,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RuntimeStore(Protocol):
    async def init(self) -> None:
        ...

    async def recover_incomplete_jobs(
        self,
    ) -> int:
        ...


class AsyncCloseable(Protocol):
    async def aclose(self) -> None:
        ...


PollingRunner = Callable[
    ...,
    Awaitable[int],
]
GenerationRunner = Callable[
    ...,
    Awaitable[bool],
]
DeliveryRunner = Callable[
    ...,
    Awaitable[bool],
]
Sleeper = Callable[
    [float],
    Awaitable[None],
]


class TelegramGatewayRuntime:
    def __init__(
        self,
        *,
        store: RuntimeStore,
        telegram_bot: AsyncCloseable,
        kven_client: AsyncCloseable,
        allowed_user_id: int,
        polling_timeout: int = 50,
        idle_delay: float = 0.1,
        error_delay: float = 5.0,
        sleeper: Sleeper = asyncio.sleep,
        polling_runner: PollingRunner = (
            run_polling_once
        ),
        generation_runner: GenerationRunner = (
            run_generation_once
        ),
        delivery_runner: DeliveryRunner = (
            run_delivery_once
        ),
    ):
        if (
            not isinstance(allowed_user_id, int)
            or isinstance(allowed_user_id, bool)
            or allowed_user_id <= 0
        ):
            raise ValueError(
                "Telegram allowed user ID must be "
                "a positive integer"
            )

        if (
            not isinstance(polling_timeout, int)
            or isinstance(polling_timeout, bool)
            or polling_timeout <= 0
        ):
            raise ValueError(
                "Telegram polling timeout must be "
                "a positive integer"
            )

        if (
            not isinstance(idle_delay, (int, float))
            or isinstance(idle_delay, bool)
            or idle_delay < 0
        ):
            raise ValueError(
                "Telegram idle delay must be "
                "non-negative"
            )

        if (
            not isinstance(error_delay, (int, float))
            or isinstance(error_delay, bool)
            or error_delay <= 0
        ):
            raise ValueError(
                "Telegram error delay must be positive"
            )

        for name, value in (
            ("sleeper", sleeper),
            ("polling runner", polling_runner),
            ("generation runner", generation_runner),
            ("delivery runner", delivery_runner),
        ):
            if not callable(value):
                raise TypeError(
                    f"Telegram {name} must be callable"
                )

        self._store = store
        self._telegram_bot = telegram_bot
        self._kven_client = kven_client
        self._allowed_user_id = allowed_user_id
        self._polling_timeout = polling_timeout
        self._idle_delay = float(idle_delay)
        self._error_delay = float(error_delay)
        self._sleeper = sleeper
        self._polling_runner = polling_runner
        self._generation_runner = (
            generation_runner
        )
        self._delivery_runner = delivery_runner

    async def run(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        tasks: list[asyncio.Task[None]] = []

        try:
            await self._store.init()
            recovered = (
                await self._store
                .recover_incomplete_jobs()
            )

            logger.info(
                "Telegram gateway startup recovery "
                "completed: recovered=%s",
                recovered,
            )

            if stop_event.is_set():
                return

            tasks = [
                asyncio.create_task(
                    self._polling_loop(stop_event),
                    name="telegram-polling",
                ),
                asyncio.create_task(
                    self._generation_loop(stop_event),
                    name="telegram-generation",
                ),
                asyncio.create_task(
                    self._delivery_loop(stop_event),
                    name="telegram-delivery",
                ),
            ]

            await stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()

            if tasks:
                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

            await self._close_resources()

    async def _polling_loop(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                processed = (
                    await self._polling_runner(
                        self._store,
                        self._telegram_bot,
                        allowed_user_id=(
                            self._allowed_user_id
                        ),
                        timeout=self._polling_timeout,
                    )
                )
            except Exception:
                logger.exception(
                    "Telegram polling iteration failed"
                )
                await self._sleeper(
                    self._error_delay
                )
                continue

            if not processed:
                await self._sleeper(
                    self._idle_delay
                )

    async def _generation_loop(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                processed = (
                    await self._generation_runner(
                        self._store,
                        self._kven_client,
                    )
                )
            except Exception:
                logger.exception(
                    "Telegram generation iteration "
                    "failed"
                )
                await self._sleeper(
                    self._error_delay
                )
                continue

            if not processed:
                await self._sleeper(
                    self._idle_delay
                )

    async def _delivery_loop(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                processed = (
                    await self._delivery_runner(
                        self._store,
                        self._telegram_bot,
                    )
                )
            except Exception:
                logger.exception(
                    "Telegram delivery iteration failed"
                )
                await self._sleeper(
                    self._error_delay
                )
                continue

            if not processed:
                await self._sleeper(
                    self._idle_delay
                )

    async def _close_resources(self) -> None:
        for name, resource in (
            (
                "Telegram Bot API client",
                self._telegram_bot,
            ),
            (
                "Kven API client",
                self._kven_client,
            ),
        ):
            try:
                await resource.aclose()
            except Exception:
                logger.exception(
                    "%s close failed",
                    name,
                )
