import unittest
from typing import Any

from telegram_updates import (
    TelegramTextUpdate,
    TelegramUpdateError,
    ingest_telegram_update,
    parse_authorized_text_update,
)


def make_text_update(
    *,
    update_id: int = 100,
    chat_id: int = 200,
    chat_type: str = "private",
    user_id: int = 300,
    message_id: int = 400,
    text: Any = "hello",
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": message_id,
        "chat": {
            "id": chat_id,
            "type": chat_type,
        },
        "from": {
            "id": user_id,
            "is_bot": False,
        },
    }

    if text is not None:
        message["text"] = text

    return {
        "update_id": update_id,
        "message": message,
    }


class FakeStore:
    def __init__(
        self,
        *,
        enqueue_result: bool = True,
    ):
        self.enqueue_result = enqueue_result
        self.enqueued: list[dict[str, Any]] = []
        self.offsets: list[int] = []

    async def enqueue_text_update(
        self,
        **values: Any,
    ) -> bool:
        self.enqueued.append(values)
        return self.enqueue_result

    async def advance_update_offset(
        self,
        next_offset: int,
    ) -> None:
        self.offsets.append(next_offset)


class TelegramUpdateParserTests(unittest.TestCase):
    def test_allowed_private_text_is_parsed(self):
        raw_update = make_text_update()

        parsed = parse_authorized_text_update(
            raw_update,
            allowed_user_id=300,
        )

        self.assertEqual(
            parsed,
            TelegramTextUpdate(
                update_id=100,
                chat_id=200,
                user_id=300,
                message_id=400,
                text="hello",
                raw_update=raw_update,
            ),
        )

    def test_text_is_preserved_without_stripping(self):
        raw_update = make_text_update(
            text="  line one\nline two  ",
        )

        parsed = parse_authorized_text_update(
            raw_update,
            allowed_user_id=300,
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed.text,
            "  line one\nline two  ",
        )

    def test_commands_are_normal_text(self):
        raw_update = make_text_update(
            text="/status",
        )

        parsed = parse_authorized_text_update(
            raw_update,
            allowed_user_id=300,
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.text, "/status")

    def test_unsupported_updates_are_ignored(self):
        cases = {
            "group chat": make_text_update(
                chat_type="group",
            ),
            "different user": make_text_update(
                user_id=301,
            ),
            "photo without text": make_text_update(
                text=None,
            ),
            "empty text": make_text_update(
                text="",
            ),
            "non-string text": make_text_update(
                text=123,
            ),
            "service update": {
                "update_id": 100,
                "my_chat_member": {
                    "chat": {
                        "id": 200,
                    }
                },
            },
            "edited message": {
                "update_id": 100,
                "edited_message": make_text_update()[
                    "message"
                ],
            },
        }

        for label, raw_update in cases.items():
            with self.subTest(label=label):
                self.assertIsNone(
                    parse_authorized_text_update(
                        raw_update,
                        allowed_user_id=300,
                    )
                )

    def test_malformed_update_id_is_rejected(self):
        cases = [
            {},
            {
                "update_id": True,
            },
            {
                "update_id": -1,
            },
            {
                "update_id": "100",
            },
        ]

        for raw_update in cases:
            with self.subTest(raw_update=raw_update):
                with self.assertRaises(
                    TelegramUpdateError
                ):
                    parse_authorized_text_update(
                        raw_update,
                        allowed_user_id=300,
                    )

    def test_invalid_allowed_user_id_is_rejected(self):
        raw_update = make_text_update()

        for allowed_user_id in (
            True,
            0,
            -1,
            "300",
        ):
            with self.subTest(
                allowed_user_id=allowed_user_id
            ):
                with self.assertRaises(ValueError):
                    parse_authorized_text_update(
                        raw_update,
                        allowed_user_id=allowed_user_id,
                    )


class TelegramUpdateIngestionTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_allowed_update_is_enqueued(self):
        raw_update = make_text_update()
        store = FakeStore()

        inserted = await ingest_telegram_update(
            store,
            raw_update,
            allowed_user_id=300,
        )

        self.assertTrue(inserted)
        self.assertEqual(store.offsets, [])
        self.assertEqual(
            store.enqueued,
            [
                {
                    "update_id": 100,
                    "chat_id": 200,
                    "user_id": 300,
                    "message_id": 400,
                    "text": "hello",
                    "raw_update": raw_update,
                }
            ],
        )

    async def test_duplicate_result_is_preserved(self):
        raw_update = make_text_update()
        store = FakeStore(
            enqueue_result=False,
        )

        inserted = await ingest_telegram_update(
            store,
            raw_update,
            allowed_user_id=300,
        )

        self.assertFalse(inserted)
        self.assertEqual(len(store.enqueued), 1)
        self.assertEqual(store.offsets, [])

    async def test_ignored_update_advances_offset(self):
        raw_update = make_text_update(
            chat_type="supergroup",
        )
        store = FakeStore()

        inserted = await ingest_telegram_update(
            store,
            raw_update,
            allowed_user_id=300,
        )

        self.assertFalse(inserted)
        self.assertEqual(store.enqueued, [])
        self.assertEqual(store.offsets, [101])

    async def test_malformed_update_does_not_touch_store(
        self,
    ):
        store = FakeStore()

        with self.assertRaises(TelegramUpdateError):
            await ingest_telegram_update(
                store,
                {
                    "message": {},
                },
                allowed_user_id=300,
            )

        self.assertEqual(store.enqueued, [])
        self.assertEqual(store.offsets, [])


if __name__ == "__main__":
    unittest.main()
