# /opt/kven2/consolidation.py

import logging
import json
import asyncio

from sqlite import (
    insert_memory,
    get_recent_episodic,
    query
)

from small_model import call_small_model
from embedder import get_embedding

import hnsw

logger = logging.getLogger("Kven.Memory")

# --------------------------------------------------
# Разрешённые типы знаний для semantic_memory
# --------------------------------------------------

ALLOWED_TYPES = {
    "Verified Invariant",
    "Procedure",
    "Decision",
    "Failure Pattern"
}

# --------------------------------------------------
# Промпт консолидации
# --------------------------------------------------

CONSOLIDATION_PROMPT = """
SYSTEM: You are a conservative memory consolidation engine.

TASK:
Convert episodic memories into reusable semantic knowledge.

STRICT RULES:

1. Extract ONLY information explicitly supported by the logs.
2. Never invent rules.
3. Never generalize from a single event.
4. Never output hypotheses.
5. Never output observations.
6. Never output temporary states.
7. Only output:

   - Verified Invariant
   - Procedure
   - Decision
   - Failure Pattern

OUTPUT FORMAT:

[
  {{
    "type": "Decision",
    "content": "Retrieval ranking switched to similarity+confidence scoring"
  }}
]

OUTPUT MUST BE VALID JSON.
NO markdown.
NO explanations.

LOGS:
{episodic_logs}
"""

# --------------------------------------------------
# Служебные функции
# --------------------------------------------------

def clean_json_response(text: str) -> str:
    """
    Удаляем markdown-обёртки.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


async def semantic_duplicate_exists(content: str) -> bool:
    """
    Проверка полного дубликата.
    """

    rows = await query(
        "SELECT id FROM semantic_memory WHERE content = ?",
        (content,)
    )

    return bool(rows)


# --------------------------------------------------
# Основная консолидация
# --------------------------------------------------

async def consolidate():

    logger.info(
        "[CONSOLIDATION] Starting strict consolidation..."
    )

    try:

        episodic_logs = await get_recent_episodic(limit=10)

        if not episodic_logs:
            logger.info(
                "[CONSOLIDATION] No episodic memories."
            )
            return

        logs_text = "\n".join(
            f"- {log}"
            for log in episodic_logs
        )

        prompt = CONSOLIDATION_PROMPT.format(
            episodic_logs=logs_text
        )

        raw_response = await call_small_model(
            prompt,
            grammar=None,
            max_tokens=1024
        )

        if not raw_response:
            logger.warning(
                "[CONSOLIDATION] Empty model response."
            )
            return

        cleaned = clean_json_response(raw_response)

        data = json.loads(cleaned)

        if not isinstance(data, list):
            logger.warning(
                "[CONSOLIDATION] Response is not a list."
            )
            return

        saved = 0
        skipped = 0

        for item in data:

            if not isinstance(item, dict):
                skipped += 1
                continue

            memory_type = item.get("type", "").strip()
            content = item.get("content", "").strip()

            if not content:
                skipped += 1
                continue

            content = content[:2500]

            # --------------------------------------
            # Тип должен быть разрешён
            # --------------------------------------

            if memory_type not in ALLOWED_TYPES:

                logger.debug(
                    f"[CONSOLIDATION] Blocked type: "
                    f"{memory_type}"
                )

                skipped += 1
                continue

            # --------------------------------------
            # Проверка дубликатов
            # --------------------------------------

            if await semantic_duplicate_exists(content):

                logger.debug(
                    "[CONSOLIDATION] Duplicate skipped."
                )

                skipped += 1
                continue

            # --------------------------------------
            # Важность по типу знания
            # --------------------------------------

            importance = {
                "Verified Invariant": 0.8,
                "Procedure": 0.75,
                "Decision": 0.7,
                "Failure Pattern": 0.75
            }.get(memory_type, 0.6)

            # --------------------------------------
            # Сохранение
            # --------------------------------------

            new_id = await insert_memory(
                content=content,
                kind="semantic",
                importance=importance,
                tags='["consolidated"]',
                decay_rate=0.99,
                table_name="semantic_memory",
                epistemic_type=memory_type,
                source="consolidation_verified"
            )

            if not new_id:
                continue

            saved += 1

            # --------------------------------------
            # Индексация в HNSW
            # --------------------------------------

            try:

                vector = await get_embedding(
                    content
                )

                await asyncio.to_thread(
                    hnsw.add_to_hnsw,
                    [new_id],
                    [vector]
                )

            except Exception as e:

                logger.error(
                    f"[HNSW] Failed to index "
                    f"{new_id}: {e}"
                )

        logger.info(
            f"[CONSOLIDATION] Saved={saved}, "
            f"Skipped={skipped}"
        )

    except Exception as e:

        logger.error(
            f"[CONSOLIDATION] Fatal error: {e}",
            exc_info=True
        )