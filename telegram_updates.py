from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TelegramUpdateError(ValueError):
    pass


SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}
MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024


@dataclass(frozen=True)
class TelegramTextUpdate:
    update_id: int
    chat_id: int
    user_id: int
    message_id: int
    text: str
    raw_update: dict[str, Any]
    message_date: int | None = None
    reply_to_message_id: int | None = None
    media: "TelegramImageMedia | None" = None


@dataclass(frozen=True)
class TelegramImageMedia:
    kind: str
    file_id: str
    file_unique_id: str
    mime_type: str
    filename: str | None = None
    width: int | None = None
    height: int | None = None
    file_size: int | None = None


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
    caption = message.get("caption")
    media = _parse_image_media(message)
    if "document" in message and media is None:
        return None
    if text is None:
        text = caption

    if chat_id is None or message_id is None:
        return None

    if media is None and (not isinstance(text, str) or text == ""):
        return None
    if text is not None and (not isinstance(text, str) or text == ""):
        raise TelegramUpdateError("Telegram image caption is malformed")
    if text is None:
        text = "Image attachment."

    return TelegramTextUpdate(
        update_id=update_id,
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        text=text,
        message_date=(
            message.get("date")
            if isinstance(message.get("date"), int)
            and not isinstance(message.get("date"), bool)
            else None
        ),
        reply_to_message_id=(
            message.get("reply_to_message", {}).get("message_id")
            if isinstance(message.get("reply_to_message"), dict)
            and isinstance(message.get("reply_to_message", {}).get("message_id"), int)
            else None
        ),
        raw_update=raw_update,
        media=media,
    )


def _required_media_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TelegramUpdateError(f"Telegram image has no valid {label}")
    return value


def _optional_media_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TelegramUpdateError(f"Telegram image has invalid {label}")
    if label == "file_size" and value > MAX_IMAGE_FILE_SIZE:
        raise TelegramUpdateError("Telegram image exceeds the size limit")
    return value


def _parse_image_media(message: dict[str, Any]) -> TelegramImageMedia | None:
    photos = message.get("photo")
    if photos is not None:
        if not isinstance(photos, list) or not photos or any(not isinstance(p, dict) for p in photos):
            raise TelegramUpdateError("Telegram photo variants are malformed")
        parsed = []
        for photo in photos:
            width = _optional_media_int(photo.get("width"), "width")
            height = _optional_media_int(photo.get("height"), "height")
            if width is None or height is None:
                raise TelegramUpdateError("Telegram photo has no valid dimensions")
            parsed.append(TelegramImageMedia(
                kind="photo",
                file_id=_required_media_text(photo.get("file_id"), "file_id"),
                file_unique_id=_required_media_text(photo.get("file_unique_id"), "file_unique_id"),
                mime_type="image/jpeg",
                width=width,
                height=height,
                file_size=_optional_media_int(photo.get("file_size"), "file_size"),
            ))
        return max(parsed, key=lambda item: ((item.width or 0) * (item.height or 0), item.file_size or 0))
    document = message.get("document")
    if document is None:
        return None
    if not isinstance(document, dict):
        raise TelegramUpdateError("Telegram document metadata is malformed")
    mime_type = document.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type.lower().startswith("image/"):
        return None
    if mime_type.lower() not in SUPPORTED_IMAGE_MIME_TYPES:
        return None
    filename = document.get("file_name")
    if filename is not None and (not isinstance(filename, str) or not filename):
        raise TelegramUpdateError("Telegram image document has invalid filename")
    return TelegramImageMedia(
        kind="document",
        file_id=_required_media_text(document.get("file_id"), "file_id"),
        file_unique_id=_required_media_text(document.get("file_unique_id"), "file_unique_id"),
        mime_type=mime_type.lower(),
        filename=filename,
        width=_optional_media_int(document.get("width"), "width"),
        height=_optional_media_int(document.get("height"), "height"),
        file_size=_optional_media_int(document.get("file_size"), "file_size"),
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

    extra = {}
    if parsed.message_date is not None:
        extra["message_date"] = parsed.message_date
    if parsed.reply_to_message_id is not None:
        extra["reply_to_message_id"] = parsed.reply_to_message_id
    if parsed.media is not None:
        extra["media"] = parsed.media
    return await store.enqueue_text_update(
        update_id=parsed.update_id,
        chat_id=parsed.chat_id,
        user_id=parsed.user_id,
        message_id=parsed.message_id,
        text=parsed.text,
        raw_update=parsed.raw_update,
        **extra,
    )
