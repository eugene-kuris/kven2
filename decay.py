import logging
import asyncio
# Исправлен импорт: теперь из того же каталога, без 'agent.storage'
from sqlite import get_connection

logger = logging.getLogger("Kven.Memory")

OLD_RECORD_THRESHOLD = 3600 

# Синохронная функция для выполнения внутри потока (чтобы не блокировать event loop)
def _sync_decay():
    conn = get_connection()
    try:
        logger.info("[DECAY] Lowering importance of old episodic memories...")
        conn.execute("UPDATE episodic_memory SET importance = importance * 0.5 WHERE created_at < datetime('now', '-1 hour')")
        
        logger.info("[DECAY] Lowering importance of old semantic memories...")
        conn.execute("UPDATE semantic_memory SET importance = importance * 0.9 WHERE created_at < datetime('now', '-24 hours')")
        
        conn.commit()
        logger.info("[DECAY] Hygiene check complete.")
    except Exception as e:
        logger.error(f"[DECAY] Error: {e}")
    finally:
        conn.close()

async def run_decay():
    logger.info("[DECAY] Starting memory hygiene check...")
    # Запускаем синхронный код в отдельном потоке
    await asyncio.to_thread(_sync_decay)
