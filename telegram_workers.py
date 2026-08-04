from __future__ import annotations

from typing import Any, Protocol

from telegram_store import TelegramJob


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
