import sys
import logging
import asyncio

sys.path.append('/opt/kven2')

import hnsw
import embedder
from sqlite import query

logger = logging.getLogger("Kven.Retrieval")

async def retrieve_context(query_text: str, top_k_raw=20, top_k_final=5):
    """
    Асинхронное извлечение контекста из векторной памяти.
    1. Embed user query
    2. HNSW top_k_raw
    3. Fetch details via async query
    4. Re-rank by confidence & return top_k_final
    """
    if not query_text or not query_text.strip():
        return []
        
    try:
        logger.debug(f"[RETRIEVAL] Generating embedding for query: '{query_text[:50]}...'")
        # 1. Асинхронная генерация эмбеддинга
        query_vector = await embedder.get_embedding(query_text)
        if not query_vector:
            logger.warning("[RETRIEVAL] Embedding generation returned empty vector.")
            return []
            
        # Безопасная инициализация HNSW для изолированных тестов
        if hnsw.hnsw_index is None:
            logger.debug("[RETRIEVAL] Initializing HNSW for isolated test...")
            hnsw.init_hnsw()
            
        # 2. Поиск в HNSW
        # hnsw.get_nearest_neighbors возвращает список кортежей: [(id, distance), ...]
        candidates = hnsw.get_nearest_neighbors(query_vector, k=top_k_raw)
        if not candidates:
            logger.debug("[RETRIEVAL] No HNSW neighbors found.")
            return []
            
        # 3. Сбор контекста и ранжирование
        retrieved_items = []
        
        # ИСПРАВЛЕНИЕ: Распаковка кортежа (db_id, distance)
        for db_id, distance in candidates:
            db_id = int(db_id)
            
            # 4. Асинхронный запрос к БД
            row = await query(
                "SELECT id, content, epistemic_type, importance as confidence FROM semantic_memory WHERE id = ?", 
                (db_id,)
            )
            if not row:
                row = await query(
                    "SELECT id, content, epistemic_type, importance as confidence FROM episodic_memory WHERE id = ?", 
                    (db_id,)
                )
                
            if row:
                row = row[0]
                confidence = row[3] if row[3] is not None else 0.5
                retrieved_items.append({
                    "id": row[0],
                    "content": row[1],
                    "type": row[2],
                    "confidence": confidence
                })
                
        # 5. Сортировка по confidence и возврат top_k_final
        retrieved_items.sort(key=lambda x: x["confidence"], reverse=True)
        final_results = retrieved_items[:top_k_final]
        
        logger.debug(f"[RETRIEVAL] Retrieved {len(final_results)} items from HNSW.")
        return final_results
        
    except Exception as e:
        logger.error(f"[RETRIEVAL] Error during retrieval: {e}", exc_info=True)
        return []