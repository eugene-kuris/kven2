import os
import sys
import atexit
import threading
import numpy as np
import hnswlib
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

logger = logging.getLogger("Kven.Memory")

# Путь к индексу из config.py [1]
INDEX_PATH = settings.INDEX_PATH
MAX_DIM = 768  # Размерность nomic-ai/nomic-embed-text-v1 [4]
_SAVE_THRESHOLD = 100

hnsw_index = None
next_hnsw_id = {}
_SAVE_COUNTER = 0
hnsw_lock = threading.Lock()

def init_hnsw():
    global hnsw_index
    if hnsw_index is not None:
        return
    
    if not os.path.exists(INDEX_PATH):
        logger.warning(f"[HNSW] Index file not found: {INDEX_PATH}. Creating new index.")
        hnsw_index = hnswlib.Index(space='cosine', dim=MAX_DIM)
        hnsw_index.init_index(max_elements=1000, ef_construction=64, M=16)
        return

    logger.info(f"[HNSW] Loading index from {INDEX_PATH}...")
    try:
        hnsw_index = hnswlib.Index(space='cosine', dim=MAX_DIM)
        hnsw_index.load_index(INDEX_PATH)
        logger.info(f"[HNSW] Index loaded successfully. Elements: {hnsw_index.get_current_count()}")
    except Exception as e:
        logger.error(f"[HNSW] Failed to load index: {e}. Creating new index.")
        hnsw_index = hnswlib.Index(space='cosine', dim=MAX_DIM)
        hnsw_index.init_index(max_elements=1000, ef_construction=64, M=16)

def add_to_hnsw(labels, vectors):
    global hnsw_index, next_hnsw_id, _SAVE_COUNTER
    if hnsw_index is None:
        logger.debug(f"[HNSW] add_to_hnsw пропущен: index is None. labels={labels}")
        return False
    with hnsw_lock:
        try:
            if not isinstance(labels, (list, np.ndarray)):
                labels = [labels]
            labels = list(labels)
            
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)

            logger.debug(f"[HNSW] Готово к вставке: labels={labels}, vecs_shape={vectors.shape}")  # 👈 ОТЛАДКА
            
            hnsw_ids = hnsw_index.add_items(vectors, labels)
            
            logger.debug(f"[HNSW] Успешно вставлено {len(hnsw_ids)} векторов.")          # 👈 ОТЛАДКА
            
            hnsw_ids = [int(i) for i in hnsw_ids]
            for i, label in enumerate(hnsw_ids):
                next_hnsw_id[label] = int(labels[i])
                
            _SAVE_COUNTER += len(labels)
            if _SAVE_COUNTER >= _SAVE_THRESHOLD:
                hnsw_index.save_index(INDEX_PATH)
                logger.info(f"[HNSW] Auto-saved index to disk (threshold reached: {_SAVE_COUNTER} items).")
                _SAVE_COUNTER = 0
                
            return True
        except Exception as e:
            logger.error(f"[HNSW] Add error: {e}")
            return False

def save_hnsw():
    """Принудительное сохранение индекса и ID-карты на диск."""
    global _SAVE_COUNTER
    if hnsw_index is None:
        return True
    with hnsw_lock:
        try:
            if _SAVE_COUNTER > 0:
                hnsw_index.save_index(INDEX_PATH)
                logger.info("[HNSW] Forced flush to disk (remainder saved).")
                _SAVE_COUNTER = 0
            return True
        except Exception as e:
            logger.error(f"[HNSW] Save error: {e}")
            return False

def get_nearest_neighbors(query_vector, k=5):
    global hnsw_index
    if hnsw_index is None:
        return []
    with hnsw_lock:
        try:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
            labels, distances = hnsw_index.knn_query(query_vector, k=k)
            return list(zip(labels[0], distances[0]))
        except Exception as e:
            logger.error(f"[HNSW] Search error: {e}")
            return []

def _register_atexit():
    try:
        atexit.register(save_hnsw)
        logger.info("[HNSW] Registered graceful shutdown handler.")
    except Exception as e:
        logger.warning(f"[HNSW] Failed to register atexit handler: {e}")

_register_atexit()