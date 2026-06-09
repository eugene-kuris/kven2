# /opt/kven2/write_path.py

import re
import logging
import asyncio

from small_model import call_small_model
from sqlite import insert_memory
from consolidation import consolidate

logger = logging.getLogger("Kven.Memory")

EXTRACT_PROMPT = """
SYSTEM: You are a strict memory archivist.

TASK:
Extract key facts, decisions, observations, active problems and conclusions from the conversation.

OUTPUT RULES:
- Output ONLY lines beginning with FACT: or MEMORY:
- One fact per line
- No markdown
- No code blocks
- No explanations
- No numbering
- No JSON
- If nothing useful exists, output nothing
- Maximum 15 entries
- Prefer important information
- Ignore conversational filler

EXAMPLES:

FACT: Server runs Ubuntu 24.04
FACT: JVM heap increased to 2 GB
MEMORY: User is troubleshooting memory subsystem

Dialogue:
{dialogue}

Output:
"""


def strip_reasoning(text: str) -> str:
    """
    Удаляет reasoning-блоки различных моделей.
    """

    patterns = [
        r'<\|think_start\|>.*?<\|think_end\|>',
        r'<think>.*?</think>',
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def parse_memory_text(raw_text: str) -> list:
    """
    Извлекает строки FACT:/MEMORY: из ответа модели.
    """

    items = []

    pattern = re.compile(
        r'^(FACT|MEMORY):\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = pattern.findall(raw_text)

    if not matches:
        logger.warning(
            "[WRITE_PATH] No valid FACT/MEMORY entries extracted."
        )
        return []

    for entry_type, content in matches:

        content = content.strip()

        if len(content) < 3:
            continue

        content = re.sub(
            r'```.*?```',
            '',
            content,
            flags=re.DOTALL
        ).strip()

        if not content:
            continue

        items.append({
            "content": content[:2500],
            "kind": "episodic",
            "epistemic_type": (
                "Fact"
                if entry_type.upper() == "FACT"
                else "Observation"
            ),
            "source": "model_inference"
        })

    return items


async def process_episodic(
    messages: list,
    assistant_reply: str,
    active_state: dict
):
    """
    Основной pipeline записи эпизодической памяти.
    """

    if not assistant_reply:
        logger.info(
            "[WRITE_PATH] Empty assistant reply. Skipping."
        )
        return

    if len(assistant_reply.strip()) < 10:
        logger.info(
            "[WRITE_PATH] Assistant reply too short. Skipping."
        )
        return

    try:

        logger.info(
            f"[WRITE_PATH] Task started. Assistant reply length: "
            f"{len(assistant_reply)}"
        )

        cleaned_reply = strip_reasoning(assistant_reply)

        dialogue_lines = []

        for m in messages:

            role = m.get("role", "unknown")
            content = m.get("content", "")

            if role == "system":
                continue

            dialogue_lines.append(
                f"{role}: {content}"
            )

        dialogue_lines.append(
            f"assistant: {cleaned_reply}"
        )

        full_dialogue = "\n".join(dialogue_lines)

        # ограничение контекста для маленькой модели
        full_dialogue = full_dialogue[-4000:]

        prompt = EXTRACT_PROMPT.format(
            dialogue=full_dialogue
        )

        raw_response = await call_small_model(
            prompt,
            grammar=None,
            max_tokens=512
        )

        logger.info(
            "[WRITE_PATH] Raw Small Model response: "
            f"{raw_response[:300]}"
        )

        items = parse_memory_text(raw_response)

        if not items:
            logger.info(
                "[WRITE_PATH] Nothing useful extracted."
            )
            return

        saved = 0

        for item in items:

            await insert_memory(
                content=item["content"],
                kind=item["kind"],
                importance=0.7,
                tags="[]",
                decay_rate=0.95,
                table_name="episodic_memory",
                epistemic_type=item["epistemic_type"],
                source=item["source"]
            )

            saved += 1

        logger.info(
            f"[WRITE_PATH] Saved {saved} episodic memories."
        )

        asyncio.create_task(consolidate())

    except Exception as e:
        logger.error(
            f"[WRITE_PATH] Fatal Error: {e}",
            exc_info=True
        )