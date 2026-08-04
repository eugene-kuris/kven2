from __future__ import annotations

from typing import Any, Protocol

from telegram_store import (
    TelegramDelivery,
    TelegramJob,
)


class GenerationStore(Protocol):
    async def claim_next_job(
        self,
    ) -> TelegramJob | None:
        ...

    async def load_conversation(
        self,
        chat_id: int,
        *,
        through_update_id: int | None = None,
    ) -> list[dict[str, str]]:
        ...

    async def save_response(
        self,
        job_id: int,
        response_text: str,
    ) -> None:
        ...


class KvenReplyClient(Protocol):
    async def generate_reply(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        ...


async def run_generation_once(
    store: GenerationStore,
    kven_client: KvenReplyClient,
) -> bool:
    job = await store.claim_next_job()

    if job is None:
        return False

    messages = await store.load_conversation(
        job.chat_id,
        through_update_id=job.update_id,
    )

    response_text = await kven_client.generate_reply(
        messages
    )

    await store.save_response(
        job.id,
        response_text,
    )

    return True


class DeliveryStore(Protocol):
    async def claim_next_delivery(
        self,
    ) -> TelegramDelivery | None:
        ...

    async def mark_delivery_chunk_delivered(
        self,
        chunk_id: int,
        telegram_message_id: int,
    ) -> bool:
        ...


class TelegramBotClient(Protocol):
    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> int:
        ...


async def run_delivery_once(
    store: DeliveryStore,
    telegram_bot: TelegramBotClient,
) -> bool:
    delivery = await store.claim_next_delivery()

    if delivery is None:
        return False

    telegram_message_id = (
        await telegram_bot.send_message(
            chat_id=delivery.chat_id,
            text=delivery.text,
            reply_to_message_id=(
                delivery.reply_to_message_id
            ),
        )
    )

    await store.mark_delivery_chunk_delivered(
        delivery.chunk_id,
        telegram_message_id,
    )

    return True
