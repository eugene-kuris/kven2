from __future__ import annotations

from typing import Any, Protocol

import httpx


TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_MAX_TEXT_UNITS = 4096
TELEGRAM_SAFE_TEXT_UNITS = 4000


class AsyncHttpClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> Any:
        ...


class TelegramBotApiError(RuntimeError):
    def __init__(
        self,
        method: str,
        message: str,
        *,
        error_code: int | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.error_code = error_code
        self.retry_after = retry_after


def telegram_text_units(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return len(text.encode("utf-16-le")) // 2


def _maximum_prefix_length(
    text: str,
    limit: int,
) -> int:
    low = 0
    high = len(text)

    while low < high:
        middle = (low + high + 1) // 2

        if telegram_text_units(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1

    return low


def _preferred_split_position(
    text: str,
    maximum_position: int,
) -> int:
    minimum_preferred_position = maximum_position // 2
    prefix = text[:maximum_position]

    candidates = (
        prefix.rfind("\n\n"),
        prefix.rfind("\n"),
        prefix.rfind(" "),
    )

    for candidate in candidates:
        if candidate < minimum_preferred_position:
            continue

        separator_length = (
            2
            if prefix.startswith("\n\n", candidate)
            else 1
        )
        split_position = candidate + separator_length

        if split_position > 0:
            return split_position

    return maximum_position


def split_telegram_text(
    text: str,
    *,
    limit: int = TELEGRAM_SAFE_TEXT_UNITS,
) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text:
        raise ValueError("Telegram message text is empty")

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > TELEGRAM_MAX_TEXT_UNITS
    ):
        raise ValueError(
            "Telegram text limit must be between "
            f"1 and {TELEGRAM_MAX_TEXT_UNITS}"
        )

    chunks: list[str] = []
    remaining = text

    while telegram_text_units(remaining) > limit:
        maximum_position = _maximum_prefix_length(
            remaining,
            limit,
        )

        if maximum_position <= 0:
            raise ValueError(
                "Unable to split Telegram message text"
            )

        split_position = _preferred_split_position(
            remaining,
            maximum_position,
        )
        chunk = remaining[:split_position]

        if not chunk:
            raise ValueError(
                "Telegram text splitter produced "
                "an empty chunk"
            )

        chunks.append(chunk)
        remaining = remaining[split_position:]

    if remaining:
        chunks.append(remaining)

    if "".join(chunks) != text:
        raise RuntimeError(
            "Telegram text splitter changed content"
        )

    return chunks


class TelegramBotApi:
    def __init__(
        self,
        token: str,
        *,
        client: AsyncHttpClient | None = None,
        api_base: str = TELEGRAM_API_BASE,
    ):
        if not isinstance(token, str) or not token:
            raise ValueError("Telegram bot token is empty")

        self._token = token
        self._api_base = api_base.rstrip("/")
        self._owns_client = client is None
        self._client: AsyncHttpClient = (
            client
            if client is not None
            else httpx.AsyncClient()
        )

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close_method = getattr(
            self._client,
            "aclose",
            None,
        )

        if close_method is not None:
            await close_method()

    def _method_url(self, method: str) -> str:
        return (
            f"{self._api_base}/"
            f"bot{self._token}/{method}"
        )

    def _sanitize(self, value: object) -> str:
        return str(value).replace(
            self._token,
            "<redacted>",
        )

    async def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> Any:
        try:
            response = await self._client.post(
                self._method_url(method),
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:
            raise TelegramBotApiError(
                method,
                "transport request failed: "
                f"{self._sanitize(exc)}",
            ) from None

        try:
            body = response.json()
        except Exception as exc:
            try:
                response.raise_for_status()
            except Exception as status_exc:
                raise TelegramBotApiError(
                    method,
                    "HTTP request failed: "
                    f"{self._sanitize(status_exc)}",
                ) from None

            raise TelegramBotApiError(
                method,
                "response is not valid JSON: "
                f"{self._sanitize(exc)}",
            ) from None

        if isinstance(body, dict) and body.get("ok") is False:
            error_code = body.get("error_code")
            description = body.get(
                "description",
                "Telegram API request failed",
            )
            parameters = body.get("parameters")
            retry_after: int | None = None

            if isinstance(parameters, dict):
                raw_retry_after = parameters.get(
                    "retry_after"
                )

                if (
                    isinstance(raw_retry_after, int)
                    and not isinstance(
                        raw_retry_after,
                        bool,
                    )
                ):
                    retry_after = raw_retry_after

            raise TelegramBotApiError(
                method,
                self._sanitize(description),
                error_code=(
                    error_code
                    if isinstance(error_code, int)
                    and not isinstance(error_code, bool)
                    else None
                ),
                retry_after=retry_after,
            )

        try:
            response.raise_for_status()
        except Exception as exc:
            raise TelegramBotApiError(
                method,
                "HTTP request failed: "
                f"{self._sanitize(exc)}",
            ) from None

        if (
            not isinstance(body, dict)
            or body.get("ok") is not True
            or "result" not in body
        ):
            raise TelegramBotApiError(
                method,
                "malformed Telegram API response",
            )

        return body["result"]

    async def get_updates(
        self,
        *,
        offset: int,
        timeout: int = 50,
    ) -> list[dict[str, Any]]:
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
        ):
            raise ValueError(
                "Telegram update offset must be "
                "a non-negative integer"
            )

        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or timeout < 0
            or timeout > 50
        ):
            raise ValueError(
                "Telegram long-poll timeout must be "
                "between 0 and 50 seconds"
            )

        result = await self._call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message"],
            },
            timeout=float(timeout) + 10.0,
        )

        if (
            not isinstance(result, list)
            or any(
                not isinstance(update, dict)
                for update in result
            )
        ):
            raise TelegramBotApiError(
                "getUpdates",
                "result is not a list of updates",
            )

        return result

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> int:
        if (
            not isinstance(chat_id, int)
            or isinstance(chat_id, bool)
        ):
            raise TypeError(
                "Telegram chat ID must be an integer"
            )

        if not isinstance(text, str):
            raise TypeError(
                "Telegram message text must be a string"
            )

        text_units = telegram_text_units(text)

        if text_units == 0:
            raise ValueError(
                "Telegram message text is empty"
            )

        if text_units > TELEGRAM_MAX_TEXT_UNITS:
            raise ValueError(
                "Telegram message text exceeds "
                f"{TELEGRAM_MAX_TEXT_UNITS} UTF-16 units"
            )

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {
                "is_disabled": True,
            },
        }

        if reply_to_message_id is not None:
            if (
                not isinstance(
                    reply_to_message_id,
                    int,
                )
                or isinstance(
                    reply_to_message_id,
                    bool,
                )
            ):
                raise TypeError(
                    "Telegram reply message ID "
                    "must be an integer"
                )

            payload["reply_parameters"] = {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }

        result = await self._call(
            "sendMessage",
            payload,
            timeout=30.0,
        )

        if not isinstance(result, dict):
            raise TelegramBotApiError(
                "sendMessage",
                "result is not a message object",
            )

        message_id = result.get("message_id")

        if (
            not isinstance(message_id, int)
            or isinstance(message_id, bool)
        ):
            raise TelegramBotApiError(
                "sendMessage",
                "result has no valid message ID",
            )

        return message_id
