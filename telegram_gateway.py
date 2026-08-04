from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from config import settings
from telegram_bot_api import TelegramBotApi
from telegram_kven_client import (
    DEFAULT_KVEN_CHAT_URL,
    DEFAULT_KVEN_TIMEOUT_SECONDS,
    TelegramKvenClient,
)
from telegram_runtime import TelegramGatewayRuntime
from telegram_store import TelegramStore


DEFAULT_TELEGRAM_DB_PATH = (
    "/agent/data/kven2/telegram_gateway.db"
)
DEFAULT_TELEGRAM_KVEN_MODEL = settings.MAIN_MODEL
DEFAULT_TELEGRAM_POLLING_TIMEOUT = 50
DEFAULT_TELEGRAM_IDLE_DELAY = 0.1
DEFAULT_TELEGRAM_ERROR_DELAY = 5.0
DEFAULT_TELEGRAM_LOG_LEVEL = "INFO"

_ALLOWED_LOG_LEVELS = {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
}


@dataclass(frozen=True, repr=False)
class TelegramGatewayConfig:
    bot_token: str
    allowed_user_id: int
    db_path: str = DEFAULT_TELEGRAM_DB_PATH
    kven_model: str = DEFAULT_TELEGRAM_KVEN_MODEL
    kven_chat_url: str = DEFAULT_KVEN_CHAT_URL
    kven_timeout_seconds: float = (
        DEFAULT_KVEN_TIMEOUT_SECONDS
    )
    polling_timeout: int = (
        DEFAULT_TELEGRAM_POLLING_TIMEOUT
    )
    idle_delay: float = DEFAULT_TELEGRAM_IDLE_DELAY
    error_delay: float = (
        DEFAULT_TELEGRAM_ERROR_DELAY
    )
    log_level: str = DEFAULT_TELEGRAM_LOG_LEVEL

    def __repr__(self) -> str:
        return (
            "TelegramGatewayConfig("
            "bot_token=<redacted>, "
            f"allowed_user_id={self.allowed_user_id!r}, "
            f"db_path={self.db_path!r}, "
            f"kven_model={self.kven_model!r}, "
            f"kven_chat_url={self.kven_chat_url!r}, "
            "kven_timeout_seconds="
            f"{self.kven_timeout_seconds!r}, "
            f"polling_timeout={self.polling_timeout!r}, "
            f"idle_delay={self.idle_delay!r}, "
            f"error_delay={self.error_delay!r}, "
            f"log_level={self.log_level!r}"
            ")"
        )


def _required_text(
    environment: Mapping[str, str],
    name: str,
) -> str:
    raw_value = environment.get(name)

    if raw_value is None:
        raise ValueError(
            f"Required environment variable is missing: "
            f"{name}"
        )

    value = str(raw_value).strip()

    if not value:
        raise ValueError(
            f"Required environment variable is blank: "
            f"{name}"
        )

    return value


def _optional_text(
    environment: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    raw_value = environment.get(name)

    if raw_value is None:
        return default

    value = str(raw_value).strip()

    if not value:
        raise ValueError(
            f"Environment variable must not be blank: "
            f"{name}"
        )

    return value


def _positive_integer(
    environment: Mapping[str, str],
    name: str,
    *,
    default: int | None = None,
) -> int:
    raw_value = environment.get(name)

    if raw_value is None:
        if default is None:
            raise ValueError(
                "Required environment variable is "
                f"missing: {name}"
            )

        return default

    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"Environment variable must be a positive "
            f"integer: {name}"
        ) from None

    if value <= 0:
        raise ValueError(
            f"Environment variable must be a positive "
            f"integer: {name}"
        )

    return value


def _finite_float(
    environment: Mapping[str, str],
    name: str,
    *,
    default: float,
    allow_zero: bool,
) -> float:
    raw_value = environment.get(name)

    if raw_value is None:
        return default

    try:
        value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"Environment variable must be a finite "
            f"number: {name}"
        ) from None

    if not math.isfinite(value):
        raise ValueError(
            f"Environment variable must be a finite "
            f"number: {name}"
        )

    if value < 0 or (
        value == 0 and not allow_zero
    ):
        qualifier = (
            "non-negative"
            if allow_zero
            else "positive"
        )
        raise ValueError(
            f"Environment variable must be "
            f"{qualifier}: {name}"
        )

    return value


def load_telegram_gateway_config(
    environ: Mapping[str, str] | None = None,
) -> TelegramGatewayConfig:
    environment = (
        os.environ
        if environ is None
        else environ
    )

    bot_token = _required_text(
        environment,
        "TELEGRAM_BOT_TOKEN",
    )
    allowed_user_id = _positive_integer(
        environment,
        "TELEGRAM_ALLOWED_USER_ID",
    )
    db_path = _optional_text(
        environment,
        "TELEGRAM_DB_PATH",
        DEFAULT_TELEGRAM_DB_PATH,
    )
    kven_model = _optional_text(
        environment,
        "TELEGRAM_KVEN_MODEL",
        DEFAULT_TELEGRAM_KVEN_MODEL,
    )
    kven_chat_url = _optional_text(
        environment,
        "TELEGRAM_KVEN_CHAT_URL",
        DEFAULT_KVEN_CHAT_URL,
    )
    kven_timeout_seconds = _finite_float(
        environment,
        "TELEGRAM_KVEN_TIMEOUT_SECONDS",
        default=DEFAULT_KVEN_TIMEOUT_SECONDS,
        allow_zero=False,
    )
    polling_timeout = _positive_integer(
        environment,
        "TELEGRAM_POLLING_TIMEOUT",
        default=DEFAULT_TELEGRAM_POLLING_TIMEOUT,
    )
    idle_delay = _finite_float(
        environment,
        "TELEGRAM_IDLE_DELAY",
        default=DEFAULT_TELEGRAM_IDLE_DELAY,
        allow_zero=True,
    )
    error_delay = _finite_float(
        environment,
        "TELEGRAM_ERROR_DELAY",
        default=DEFAULT_TELEGRAM_ERROR_DELAY,
        allow_zero=False,
    )
    log_level = _optional_text(
        environment,
        "TELEGRAM_LOG_LEVEL",
        DEFAULT_TELEGRAM_LOG_LEVEL,
    ).upper()

    if log_level not in _ALLOWED_LOG_LEVELS:
        raise ValueError(
            "Unsupported TELEGRAM_LOG_LEVEL: "
            f"{log_level}"
        )

    return TelegramGatewayConfig(
        bot_token=bot_token,
        allowed_user_id=allowed_user_id,
        db_path=db_path,
        kven_model=kven_model,
        kven_chat_url=kven_chat_url,
        kven_timeout_seconds=kven_timeout_seconds,
        polling_timeout=polling_timeout,
        idle_delay=idle_delay,
        error_delay=error_delay,
        log_level=log_level,
    )


def build_telegram_gateway_runtime(
    config: TelegramGatewayConfig,
    *,
    store_factory: Callable[[str], Any] = (
        TelegramStore
    ),
    bot_factory: Callable[[str], Any] = (
        TelegramBotApi
    ),
    kven_factory: Callable[..., Any] = (
        TelegramKvenClient
    ),
    runtime_factory: Callable[..., Any] = (
        TelegramGatewayRuntime
    ),
) -> Any:
    store = store_factory(config.db_path)
    telegram_bot = bot_factory(config.bot_token)
    kven_client = kven_factory(
        model=config.kven_model,
        chat_url=config.kven_chat_url,
        timeout_seconds=(
            config.kven_timeout_seconds
        ),
    )

    return runtime_factory(
        store=store,
        telegram_bot=telegram_bot,
        kven_client=kven_client,
        allowed_user_id=config.allowed_user_id,
        polling_timeout=config.polling_timeout,
        idle_delay=config.idle_delay,
        error_delay=config.error_delay,
    )


def install_shutdown_signal_handlers(
    stop_event: asyncio.Event,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    event_loop = (
        asyncio.get_running_loop()
        if loop is None
        else loop
    )

    def request_shutdown() -> None:
        stop_event.set()

    for sig in (
        signal.SIGTERM,
        signal.SIGINT,
    ):
        event_loop.add_signal_handler(
            sig,
            request_shutdown,
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s %(message)s"
        ),
    )


async def run_telegram_gateway(
    *,
    config: TelegramGatewayConfig | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_builder: Callable[
        [TelegramGatewayConfig],
        Any,
    ] = build_telegram_gateway_runtime,
    signal_installer: Callable[
        [asyncio.Event],
        None,
    ] = install_shutdown_signal_handlers,
    logging_configurator: Callable[
        [str],
        None,
    ] = configure_logging,
) -> None:
    resolved_config = (
        load_telegram_gateway_config(environ)
        if config is None
        else config
    )

    logging_configurator(
        resolved_config.log_level
    )

    stop_event = asyncio.Event()
    signal_installer(stop_event)

    runtime = runtime_builder(resolved_config)
    await runtime.run(stop_event)


def main() -> None:
    asyncio.run(run_telegram_gateway())


if __name__ == "__main__":
    main()
