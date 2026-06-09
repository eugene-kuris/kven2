# /opt/kven2/session_manager.py
import asyncio
import logging
import sqlite3
from small_model import call_small_model
from sqlite import get_connection, get_recent_episodic

logger = logging.getLogger("Kven.Gateway")
MAX_SYS_BLOCK_LENGTH = 3500  # байт, не символы
COMPRESSION_THRESHOLD = 20000  # символов в диалоге, триггер сжатия

async def compress_history(messages: list) -> list:
    """
    Сжимает историю диалога при достижении порога.
    Заменяет старые сообщения на краткий running summary.
    """
    full_text = " ".join([m.get("content", "") for m in messages])
    if len(full_text) < COMPRESSION_THRESHOLD:
        return messages

    logger.info("[SESSION] Triggering history compression...")
    old_memories = await get_recent_episodic(limit=15)
    if not old_memories:
        return messages

    prompt = f"""
SYSTEM: You are a session archivist.
TASK: Summarize the following past interactions into a concise running summary.
RULES:
- Max 250 characters.
- Keep only active problems, key decisions, and persistent facts.
- Output ONLY the summary text. No markdown.
PAST INTERACTIONS:
{chr(10).join(old_memories)[:3000]}
Summary:
"""
    try:
        summary = await call_small_model(prompt, grammar=None, max_tokens=200)
        summary = summary.strip()
        # Сохраняем только системный блок и последние 8 сообщений
        compressed = [messages[0]] if messages else []
        compressed.append({"role": "system", "content": f"SESSION SUMMARY: {summary}"})
        compressed.extend(messages[-8:])
        logger.info(f"[SESSION] History compressed. New length: {sum(len(m.get('content','')) for m in compressed)}")
        return compressed
    except Exception as e:
        logger.error(f"[SESSION] Compression failed: {e}")
        return messages

def optimize_sys_block(sys_block: str) -> str:
    """
    Приоритетное усечение sys_block до MAX_SYS_BLOCK_LENGTH.
    Сохраняет ACTIVE STATE и проекты, обрезает семантику и identity при необходимости.
    """
    if len(sys_block.encode('utf-8')) <= MAX_SYS_BLOCK_LENGTH:
        return sys_block

    parts = sys_block.split("---")
    kept = []
    current_bytes = 0

    # 1. ACTIVE STATE (критично)
    for part in parts:
        if "ACTIVE STATE" in part or "Current Project" in part:
            kept.append(part)
            current_bytes += len(part.encode('utf-8')) + 4

    # 2. Semantic Memory (только последние 3 факта)
    for part in parts:
        if "SEMANTIC MEMORY" in part:
            lines = part.split("\n")
            header = lines[0] if lines else ""
            facts = lines[3:6] if len(lines) > 6 else lines[3:]
            kept.append(f"{header}\n" + "\n".join(facts))
            current_bytes += len(part.encode('utf-8')) + 4

    # 3. Projects, Identity, прочее
    for part in parts:
        if any(k in part for k in ["ACTIVE STATE", "SEMANTIC MEMORY"]):
            continue
        if current_bytes < MAX_SYS_BLOCK_LENGTH - 50:
            kept.append(part)
            current_bytes += len(part.encode('utf-8')) + 4

    final = "---".join(kept)
    if len(final.encode('utf-8')) > MAX_SYS_BLOCK_LENGTH:
        final = final.encode('utf-8')[:MAX_SYS_BLOCK_LENGTH].decode('utf-8', errors='ignore')
    return final

async def get_debug_state() -> dict:
    """Возвращает метрики для /debug/state"""
    conn = get_connection()
    try:
        ep_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        sem_count = conn.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0]
        active = conn.execute("SELECT state_json FROM active_state WHERE id=1").fetchone()
        return {
            "episodic_count": ep_count,
            "semantic_count": sem_count,
            "active_state": active[0] if active else "{}",
            "max_sys_block": MAX_SYS_BLOCK_LENGTH,
            "compression_trigger": COMPRESSION_THRESHOLD
        }
    finally:
        conn.close()