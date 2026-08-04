import asyncio
import signal
import unittest
from typing import Any

from telegram_gateway import (
    DEFAULT_TELEGRAM_DB_PATH,
    TelegramGatewayConfig,
    build_telegram_gateway_runtime,
    install_shutdown_signal_handlers,
    load_telegram_gateway_config,
    run_telegram_gateway,
)


REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "123456:secret-token",
    "TELEGRAM_ALLOWED_USER_ID": "987654321",
}


class TelegramGatewayConfigTests(unittest.TestCase):
    def test_required_values_and_defaults_are_loaded(self):
        config = load_telegram_gateway_config(
            REQUIRED_ENV
        )

        self.assertEqual(
            config.bot_token,
            "123456:secret-token",
        )
        self.assertEqual(
            config.allowed_user_id,
            987654321,
        )
        self.assertEqual(
            config.db_path,
            DEFAULT_TELEGRAM_DB_PATH,
        )
        self.assertTrue(config.kven_model)
        self.assertEqual(
            config.kven_chat_url,
            (
                "http://127.0.0.1:10000/"
                "v1/chat/completions"
            ),
        )
        self.assertEqual(
            config.kven_timeout_seconds,
            1200.0,
        )
        self.assertEqual(
            config.polling_timeout,
            50,
        )
        self.assertEqual(
            config.idle_delay,
            0.1,
        )
        self.assertEqual(
            config.error_delay,
            5.0,
        )
        self.assertEqual(
            config.log_level,
            "INFO",
        )

    def test_optional_values_are_parsed(self):
        environment = {
            **REQUIRED_ENV,
            "TELEGRAM_DB_PATH": (
                "/tmp/telegram-test.db"
            ),
            "TELEGRAM_KVEN_MODEL": "TEST_MODEL",
            "TELEGRAM_KVEN_CHAT_URL": (
                "http://127.0.0.1:19999/v1/chat/"
                "completions"
            ),
            "TELEGRAM_KVEN_TIMEOUT_SECONDS": (
                "321.5"
            ),
            "TELEGRAM_POLLING_TIMEOUT": "42",
            "TELEGRAM_IDLE_DELAY": "0",
            "TELEGRAM_ERROR_DELAY": "8.25",
            "TELEGRAM_LOG_LEVEL": "debug",
        }

        config = load_telegram_gateway_config(
            environment
        )

        self.assertEqual(
            config.db_path,
            "/tmp/telegram-test.db",
        )
        self.assertEqual(
            config.kven_model,
            "TEST_MODEL",
        )
        self.assertEqual(
            config.kven_chat_url,
            (
                "http://127.0.0.1:19999/v1/chat/"
                "completions"
            ),
        )
        self.assertEqual(
            config.kven_timeout_seconds,
            321.5,
        )
        self.assertEqual(
            config.polling_timeout,
            42,
        )
        self.assertEqual(
            config.idle_delay,
            0.0,
        )
        self.assertEqual(
            config.error_delay,
            8.25,
        )
        self.assertEqual(
            config.log_level,
            "DEBUG",
        )

    def test_missing_or_blank_secret_is_rejected(self):
        for environment in (
            {
                "TELEGRAM_ALLOWED_USER_ID": "1",
            },
            {
                "TELEGRAM_BOT_TOKEN": "   ",
                "TELEGRAM_ALLOWED_USER_ID": "1",
            },
        ):
            with self.subTest(
                environment=environment
            ):
                with self.assertRaises(
                    ValueError
                ):
                    load_telegram_gateway_config(
                        environment
                    )

    def test_invalid_allowed_user_id_is_rejected(self):
        for value in (
            "",
            "0",
            "-1",
            "1.5",
            "abc",
        ):
            with self.subTest(value=value):
                environment = {
                    **REQUIRED_ENV,
                    "TELEGRAM_ALLOWED_USER_ID": (
                        value
                    ),
                }

                with self.assertRaises(
                    ValueError
                ):
                    load_telegram_gateway_config(
                        environment
                    )

    def test_invalid_optional_values_are_rejected(self):
        invalid_cases = [
            (
                "TELEGRAM_DB_PATH",
                "   ",
            ),
            (
                "TELEGRAM_KVEN_MODEL",
                "   ",
            ),
            (
                "TELEGRAM_KVEN_CHAT_URL",
                "   ",
            ),
            (
                "TELEGRAM_KVEN_TIMEOUT_SECONDS",
                "0",
            ),
            (
                "TELEGRAM_KVEN_TIMEOUT_SECONDS",
                "nan",
            ),
            (
                "TELEGRAM_POLLING_TIMEOUT",
                "0",
            ),
            (
                "TELEGRAM_POLLING_TIMEOUT",
                "1.5",
            ),
            (
                "TELEGRAM_IDLE_DELAY",
                "-0.1",
            ),
            (
                "TELEGRAM_ERROR_DELAY",
                "0",
            ),
            (
                "TELEGRAM_LOG_LEVEL",
                "VERBOSE",
            ),
        ]

        for name, value in invalid_cases:
            with self.subTest(
                name=name,
                value=value,
            ):
                environment = {
                    **REQUIRED_ENV,
                    name: value,
                }

                with self.assertRaises(
                    ValueError
                ):
                    load_telegram_gateway_config(
                        environment
                    )

    def test_config_repr_does_not_expose_token(self):
        config = load_telegram_gateway_config(
            REQUIRED_ENV
        )

        representation = repr(config)

        self.assertNotIn(
            REQUIRED_ENV["TELEGRAM_BOT_TOKEN"],
            representation,
        )
        self.assertIn(
            "bot_token=<redacted>",
            representation,
        )


class TelegramGatewayBuildTests(
    unittest.TestCase
):
    def test_runtime_is_wired_from_config(self):
        config = TelegramGatewayConfig(
            bot_token="secret-token",
            allowed_user_id=111,
            db_path="/tmp/gateway.db",
            kven_model="MODEL",
            kven_chat_url=(
                "http://127.0.0.1:10000/"
                "v1/chat/completions"
            ),
            kven_timeout_seconds=222.0,
            polling_timeout=33,
            idle_delay=0.25,
            error_delay=4.5,
            log_level="WARNING",
        )
        calls: dict[str, Any] = {}

        def store_factory(
            db_path: str,
        ) -> object:
            calls["store"] = {
                "db_path": db_path,
            }
            return "STORE"

        def bot_factory(
            token: str,
        ) -> object:
            calls["bot"] = {
                "token": token,
            }
            return "BOT"

        def kven_factory(
            *,
            model: str,
            chat_url: str,
            timeout_seconds: float,
        ) -> object:
            calls["kven"] = {
                "model": model,
                "chat_url": chat_url,
                "timeout_seconds": (
                    timeout_seconds
                ),
            }
            return "KVEN"

        def runtime_factory(
            **kwargs: Any,
        ) -> object:
            calls["runtime"] = kwargs
            return "RUNTIME"

        runtime = build_telegram_gateway_runtime(
            config,
            store_factory=store_factory,
            bot_factory=bot_factory,
            kven_factory=kven_factory,
            runtime_factory=runtime_factory,
        )

        self.assertEqual(runtime, "RUNTIME")
        self.assertEqual(
            calls["store"],
            {
                "db_path": "/tmp/gateway.db",
            },
        )
        self.assertEqual(
            calls["bot"],
            {
                "token": "secret-token",
            },
        )
        self.assertEqual(
            calls["kven"],
            {
                "model": "MODEL",
                "chat_url": (
                    "http://127.0.0.1:10000/"
                    "v1/chat/completions"
                ),
                "timeout_seconds": 222.0,
            },
        )
        self.assertEqual(
            calls["runtime"],
            {
                "store": "STORE",
                "telegram_bot": "BOT",
                "kven_client": "KVEN",
                "allowed_user_id": 111,
                "polling_timeout": 33,
                "idle_delay": 0.25,
                "error_delay": 4.5,
            },
        )


class FakeSignalLoop:
    def __init__(self):
        self.handlers: list[
            tuple[Any, Any]
        ] = []

    def add_signal_handler(
        self,
        sig: Any,
        callback: Any,
    ) -> None:
        self.handlers.append(
            (
                sig,
                callback,
            )
        )


class TelegramGatewaySignalTests(
    unittest.TestCase
):
    def test_sigterm_and_sigint_set_stop_event(
        self,
    ):
        loop = FakeSignalLoop()
        stop_event = asyncio.Event()

        install_shutdown_signal_handlers(
            stop_event,
            loop=loop,
        )

        self.assertEqual(
            [sig for sig, _ in loop.handlers],
            [
                signal.SIGTERM,
                signal.SIGINT,
            ],
        )
        self.assertFalse(stop_event.is_set())

        loop.handlers[0][1]()

        self.assertTrue(stop_event.is_set())


class TelegramGatewayLoggingTests(
    unittest.TestCase
):
    def test_configure_logging_suppresses_http_clients(
        self,
    ) -> None:
        import logging
        from unittest.mock import patch

        from telegram_gateway import configure_logging

        httpx_logger = logging.getLogger("httpx")
        httpcore_logger = logging.getLogger("httpcore")

        previous_httpx_level = httpx_logger.level
        previous_httpcore_level = httpcore_logger.level

        try:
            with patch(
                "telegram_gateway.logging.basicConfig"
            ) as basic_config:
                configure_logging("INFO")

            self.assertEqual(
                basic_config.call_args.kwargs["level"],
                logging.INFO,
            )
            self.assertEqual(
                httpx_logger.level,
                logging.WARNING,
            )
            self.assertEqual(
                httpcore_logger.level,
                logging.WARNING,
            )
        finally:
            httpx_logger.setLevel(
                previous_httpx_level
            )
            httpcore_logger.setLevel(
                previous_httpcore_level
            )


class TelegramGatewayRunTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_run_installs_signals_builds_and_runs(
        self,
    ):
        events: list[str] = []
        captured: dict[str, Any] = {}
        config = TelegramGatewayConfig(
            bot_token="secret-token",
            allowed_user_id=111,
            db_path="/tmp/gateway.db",
            kven_model="MODEL",
            kven_chat_url=(
                "http://127.0.0.1:10000/"
                "v1/chat/completions"
            ),
            kven_timeout_seconds=222.0,
            polling_timeout=33,
            idle_delay=0.25,
            error_delay=4.5,
            log_level="INFO",
        )

        class FakeRuntime:
            async def run(
                self,
                stop_event: asyncio.Event,
            ) -> None:
                events.append("runtime.run")
                captured["runtime_event"] = (
                    stop_event
                )

        def signal_installer(
            stop_event: asyncio.Event,
        ) -> None:
            events.append("signals")
            captured["signal_event"] = (
                stop_event
            )

        def runtime_builder(
            received_config: (
                TelegramGatewayConfig
            ),
        ) -> FakeRuntime:
            events.append("build")
            captured["config"] = (
                received_config
            )
            return FakeRuntime()

        def logging_configurator(
            level: str,
        ) -> None:
            events.append("logging")
            captured["level"] = level

        await run_telegram_gateway(
            config=config,
            runtime_builder=runtime_builder,
            signal_installer=signal_installer,
            logging_configurator=(
                logging_configurator
            ),
        )

        self.assertEqual(
            events,
            [
                "logging",
                "signals",
                "build",
                "runtime.run",
            ],
        )
        self.assertIs(
            captured["config"],
            config,
        )
        self.assertEqual(
            captured["level"],
            "INFO",
        )
        self.assertIs(
            captured["signal_event"],
            captured["runtime_event"],
        )

    async def test_run_loads_environment_when_config_missing(
        self,
    ):
        captured: dict[str, Any] = {}

        class FakeRuntime:
            async def run(
                self,
                stop_event: asyncio.Event,
            ) -> None:
                captured["stop_event"] = (
                    stop_event
                )

        def runtime_builder(
            config: TelegramGatewayConfig,
        ) -> FakeRuntime:
            captured["config"] = config
            return FakeRuntime()

        await run_telegram_gateway(
            environ=REQUIRED_ENV,
            runtime_builder=runtime_builder,
            signal_installer=lambda event: None,
            logging_configurator=lambda level: None,
        )

        self.assertEqual(
            captured["config"].allowed_user_id,
            987654321,
        )
        self.assertEqual(
            captured["config"].bot_token,
            "123456:secret-token",
        )


if __name__ == "__main__":
    unittest.main()
