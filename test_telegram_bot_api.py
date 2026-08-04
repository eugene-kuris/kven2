import unittest
from typing import Any

from telegram_bot_api import (
    TELEGRAM_SAFE_TEXT_UNITS,
    TelegramBotApi,
    TelegramBotApiError,
    split_telegram_text,
    telegram_text_units,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
    ):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(
                f"HTTP status {self.status_code}"
            )

    def json(self) -> Any:
        return self.payload


class FakeClient:
    def __init__(self, *results: Any):
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        if not self.results:
            raise AssertionError(
                "Unexpected HTTP request"
            )

        result = self.results.pop(0)

        if isinstance(result, BaseException):
            raise result

        return result


class TelegramTextTests(unittest.TestCase):
    def test_utf16_units_count_as_telegram_entities_do(self):
        self.assertEqual(
            telegram_text_units("abc"),
            3,
        )
        self.assertEqual(
            telegram_text_units("😀"),
            2,
        )

    def test_split_preserves_text_and_safe_limit(self):
        text = (
            "First paragraph.\n\n"
            + ("Русский текст " * 500)
            + "\n"
            + ("😀" * 2500)
            + "\nEnd."
        )

        chunks = split_telegram_text(text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(chunks))
        self.assertTrue(
            all(
                telegram_text_units(chunk)
                <= TELEGRAM_SAFE_TEXT_UNITS
                for chunk in chunks
            )
        )

    def test_split_handles_one_unbroken_long_word(self):
        text = "x" * (
            TELEGRAM_SAFE_TEXT_UNITS * 2 + 17
        )

        chunks = split_telegram_text(text)

        self.assertEqual("".join(chunks), text)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(
            all(
                telegram_text_units(chunk)
                <= TELEGRAM_SAFE_TEXT_UNITS
                for chunk in chunks
            )
        )

    def test_split_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            split_telegram_text("")


class TelegramBotApiTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_get_updates_uses_offset_and_long_polling(
        self,
    ):
        update = {
            "update_id": 123,
            "message": {
                "message_id": 7,
                "text": "hello",
            },
        }
        client = FakeClient(
            FakeResponse(
                {
                    "ok": True,
                    "result": [update],
                }
            )
        )
        api = TelegramBotApi(
            "TEST_TOKEN",
            client=client,
        )

        updates = await api.get_updates(
            offset=123,
            timeout=50,
        )

        self.assertEqual(updates, [update])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            client.calls[0],
            {
                "url": (
                    "https://api.telegram.org/"
                    "botTEST_TOKEN/getUpdates"
                ),
                "json": {
                    "offset": 123,
                    "timeout": 50,
                    "allowed_updates": ["message"],
                },
                "timeout": 60.0,
            },
        )

    async def test_send_message_replies_and_returns_id(
        self,
    ):
        client = FakeClient(
            FakeResponse(
                {
                    "ok": True,
                    "result": {
                        "message_id": 900,
                        "chat": {
                            "id": 200,
                        },
                        "text": "answer",
                    },
                }
            )
        )
        api = TelegramBotApi(
            "TEST_TOKEN",
            client=client,
        )

        message_id = await api.send_message(
            chat_id=200,
            text="answer",
            reply_to_message_id=7,
        )

        self.assertEqual(message_id, 900)
        self.assertEqual(
            client.calls[0],
            {
                "url": (
                    "https://api.telegram.org/"
                    "botTEST_TOKEN/sendMessage"
                ),
                "json": {
                    "chat_id": 200,
                    "text": "answer",
                    "reply_parameters": {
                        "message_id": 7,
                        "allow_sending_without_reply": True,
                    },
                    "link_preview_options": {
                        "is_disabled": True,
                    },
                },
                "timeout": 30.0,
            },
        )

    async def test_api_error_exposes_retry_without_token(
        self,
    ):
        token = "SECRET_TOKEN_VALUE"
        client = FakeClient(
            FakeResponse(
                {
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests",
                    "parameters": {
                        "retry_after": 3,
                    },
                }
            )
        )
        api = TelegramBotApi(
            token,
            client=client,
        )

        with self.assertRaises(
            TelegramBotApiError
        ) as context:
            await api.send_message(
                chat_id=200,
                text="answer",
            )

        error = context.exception

        self.assertEqual(error.method, "sendMessage")
        self.assertEqual(error.error_code, 429)
        self.assertEqual(error.retry_after, 3)
        self.assertNotIn(token, str(error))

    async def test_transport_error_does_not_expose_token(
        self,
    ):
        token = "SECRET_TOKEN_VALUE"
        client = FakeClient(
            RuntimeError(
                "request failed for "
                f"https://api.telegram.org/bot{token}"
            )
        )
        api = TelegramBotApi(
            token,
            client=client,
        )

        with self.assertRaises(
            TelegramBotApiError
        ) as context:
            await api.get_updates(
                offset=0,
                timeout=50,
            )

        error = context.exception

        self.assertEqual(error.method, "getUpdates")
        self.assertIsNone(error.error_code)
        self.assertNotIn(token, str(error))

    async def test_malformed_result_is_rejected(self):
        client = FakeClient(
            FakeResponse(
                {
                    "ok": True,
                    "result": {
                        "unexpected": "object",
                    },
                }
            )
        )
        api = TelegramBotApi(
            "TEST_TOKEN",
            client=client,
        )

        with self.assertRaises(
            TelegramBotApiError
        ):
            await api.get_updates(
                offset=0,
                timeout=50,
            )

    async def test_send_rejects_empty_or_oversized_text(
        self,
    ):
        client = FakeClient()
        api = TelegramBotApi(
            "TEST_TOKEN",
            client=client,
        )

        with self.assertRaises(ValueError):
            await api.send_message(
                chat_id=200,
                text="",
            )

        with self.assertRaises(ValueError):
            await api.send_message(
                chat_id=200,
                text="x" * 4097,
            )

        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
