import asyncio
import sys
import os
import logging

sys.path.append('/opt/kven2')

from config import settings
from sqlite import insert_memory, get_connection
import hnsw
from embedder import get_embedding
from retrieval import retrieve_context

# Настройка логирования для теста
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase3.TestRetrieval")

TEST_CONTENT = "The Kven II memory system uses HNSW for fast vector retrieval of semantic facts."
TEST_QUERY = "How does Kven II retrieve semantic memories?"
TEST_IMPORTANCE = 0.9

async def main() -> bool:
    logger.info("🚀 Starting Phase 3: Retrieval Test Harness")
    conn = get_connection()
    test_id = None

    try:
        # 1. Вставка тестовой записи в semantic_memory
        logger.info(f"📝 Inserting test memory: '{TEST_CONTENT[:60]}...'")
        test_id = await insert_memory(
            content=TEST_CONTENT,
            kind="semantic",
            importance=TEST_IMPORTANCE,
            tags='["test", "phase3"]',
            decay_rate=0.98,
            table_name="semantic_memory",
            epistemic_type="Fact",
            source="test_harness"
        )
        if not test_id:
            logger.error("❌ FAIL: insert_memory returned None or 0.")
            return False

        # 2. Генерация эмбеддинга и добавление в HNSW
        logger.info("🧠 Generating embedding for test memory...")
        test_vector = await get_embedding(TEST_CONTENT)
        if not test_vector:
            logger.error("❌ FAIL: Embedding generation returned empty vector.")
            return False

        logger.info("📡 Adding vector to HNSW index...")
        hnsw.init_hnsw()  # Гарантируем загрузку индекса
        # add_to_hnsw синхронный, оборачиваем в поток
        added = await asyncio.to_thread(hnsw.add_to_hnsw, [test_id], [test_vector])
        if not added:
            logger.error("❌ FAIL: add_to_hnsw returned False.")
            return False

        hnsw.save_hnsw()  # Сохраняем на диск, чтобы retrieve_context увидел запись

        # 3. Запрос к retrieval pipeline
        logger.info(f"🔍 Testing retrieval with query: '{TEST_QUERY}'")
        results = await retrieve_context(TEST_QUERY, top_k_raw=20, top_k_final=5)
        
        if not results:
            logger.error("❌ FAIL: retrieve_context returned empty list.")
            return False

        # 4. Валидация качества
        found = False
        for res in results:
            logger.info(f"   → ID: {res['id']} | Conf: {res['confidence']:.2f} | Content: {res['content'][:70]}...")
            if res['id'] == test_id:
                found = True
                logger.info(f"✅ SUCCESS: Test memory (ID {test_id}) found in Top-5 results!")
                break

        if not found:
            logger.warning("⚠️ FAIL: Test memory not in Top-5. Vector space mismatch or ranking issue.")
            return False

        logger.info("✅ Phase 3 PASSED: Retrieval is functional, indexed, and relevant.")
        return True

    except Exception as e:
        logger.error(f"❌ Phase 3 FAILED with exception: {e}", exc_info=True)
        return False
    finally:
        # 5. Очистка (только БД для безопасности HNSW)
        logger.info("🧹 Cleaning up test data...")
        if test_id:
            try:
                conn.execute("DELETE FROM semantic_memory WHERE id = ?", (test_id,))
                conn.commit()
                logger.info(f"✅ Deleted test record ID {test_id} from semantic_memory.")
                # Примечание: удаление из HNSW без перестройки индекса нестабильно в текущей версии hnswlib.
                # Тестовый вектор останется в индексе, но не будет влиять на семантику из-за отсутствия записи в БД.
            except Exception as e:
                logger.error(f"⚠️ Cleanup warning: {e}")
        conn.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)