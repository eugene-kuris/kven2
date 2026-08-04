from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TelegramUpdateError(ValueError):
    pass


@dataclass(frozen=True)
class TelegramTextUpdate:
    update_id: int
    chat_id: int
    user_id: int
    message_id: int
    text: str
    raw_update: dict[str, Any]


class TelegramUpdateStore(Protocol):
    async def enqueue_text_update(
        self,
        *,
        update_id: int,
        chat_id: int,
        user_id: int,
        message_id: int,
        text: str,
        raw_update: dict[str, Any],
    ) -> bool:
        ...

    async def advance_update_offset(
        self,
        next_offset: int,
    ) -> None:
        ...


def _require_positive_int(
    value: object,
) -> int | None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        return None

    return value


def _require_update_id(
    raw_update: object,
) -> int:
    if not isinstance(raw_update, dict):
        raise TelegramUpdateError(
            "Telegram update must be an object"
        )

    update_id = raw_update.get("update_id")

    if (
        not isinstance(update_id, int)
        or isinstance(update_id, bool)
        or update_id < 0
    ):
        raise TelegramUpdateError(
            "Telegram update has no valid update_id"
        )

    return update_id


def _validate_allowed_user_id(
    allowed_user_id: object,
) -> int:
    if (
        not isinstance(allowed_user_id, int)
        or isinstance(allowed_user_id, bool)
        or allowed_user_id <= 0
    ):
        raise ValueError(
            "Allowed Telegram user ID must be "
            "a positive integer"
        )

    return allowed_user_id


def parse_authorized_text_update(
    raw_update: dict[str, Any],
    *,
    allowed_user_id: int,
) -> TelegramTextUpdate | None:
    allowed_user_id = _validate_allowed_user_id(
        allowed_user_id
    )
    update_id = _require_update_id(raw_update)

    message = raw_update.get("message")

    if not isinstance(message, dict):
        return None

    chat = message.get("chat")
    sender = message.get("from")

    if (
        not isinstance(chat, dict)
        or not isinstance(sender, dict)
    ):
        return None

    if chat.get("type") != "private":
        return None

    user_id = _require_positive_int(
        sender.get("id")
    )

    if user_id != allowed_user_id:
        return None

    if sender.get("is_bot") is not False:
        return None

    chat_id = _require_positive_int(
        chat.get("id")
    )
    message_id = _require_positive_int(
        message.get("message_id")
    )
    text = message.get("text")

    if chat_id is None or message_id is None:
        return None

    if not isinstance(text, str) or text == "":
        return None

    return TelegramTextUpdate(
        update_id=update_id,
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        text=text,
        raw_update=raw_update,
    )


async def ingest_telegram_update(
    store: TelegramUpdateStore,
    raw_update: dict[str, Any],
    *,
    allowed_user_id: int,
) -> bool:
    parsed = parse_authorized_text_update(
        raw_update,
        allowed_user_id=allowed_user_id,
    )

    if parsed is None:
        update_id = _require_update_id(raw_update)

        await store.advance_update_offset(
            update_id + 1
        )
        return False

    return await store.enqueue_text_update(
        update_id=parsed.update_id,
        chat_id=parsed.chat_id,
        user_id=parsed.user_id,
        message_id=parsed.message_id,
        text=parsed.text,
        raw_update=parsed.raw_update,
    )
