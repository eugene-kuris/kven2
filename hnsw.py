import hnswlib
import os
import json
import logging
import numpy as np
import threading
import sys
import atexit

logger = logging.getLogger(__name__)

sys.path.append('/opt/kven2')
from config import settings

index_path = settings.INDEX_PATH
id_map_path = index_path + ".id_map.json"

# Увеличиваем лимит до 2 миллионов элементов
MAX_ELEMENTS = 2000000
DIMENSION = 768
SPACE = 'cosine'

hnsw_index = None
id_to_hnsw = {}  # db_id -> hnsw_id
hnsw_to_id = {}  # hnsw_id -> db_id
next_hnsw_id = 0
hnsw_lock = threading.Lock()

# 🔹 Настройки периодического сохранения на диск
_SAVE_COUNTER = 0
_SAVE_THRESHOLD = 100  # Сохранять бинарный индекс каждые N добавленных векторов

def _load_id_map():
    global id_to_hnsw, hnsw_to_id, next_hnsw_id
    if os.path.exists(id_map_path):
        try:
            with open(id_map_path, 'r') as f:
                data = json.load(f)
            id_to_hnsw = {int(k): v for k, v in data.items()}
            hnsw_to_id = {v: int(k) for k, v in id_to_hnsw.items()}
            next_hnsw_id = max(hnsw_to_id.keys(), default=0)
            logger.info(f"[HNSW] ID map loaded. {len(id_to_hnsw)} entries.")
        except Exception as e:
            logger.error(f"[HNSW] Load ID map failed: {e}")
            id_to_hnsw = {}
            hnsw_to_id = {}

def _save_id_map():
    try:
        with open(id_map_path, 'w') as f:
            json.dump(id_to_hnsw, f)
    except Exception as e:
        logger.error(f"[HNSW] Save ID map failed: {e}")

def init_hnsw():
    global hnsw_index
    logger.info(f"[HNSW] Initializing index...")
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    _load_id_map()

    if os.path.exists(index_path):
        try:
            hnsw_index = hnswlib.Index(space=SPACE, dim=DIMENSION)
            hnsw_index.load_index(index_path)
            loaded_max = hnsw_index.get_max_elements()
            loaded_count = hnsw_index.get_current_count()
            logger.info(f"[HNSW] Index loaded. Current: {loaded_count}, Max Limit: {loaded_max}")

            # КРИТИЧЕСКАЯ ПРОВЕРКА: Если загруженный индекс имеет лимит меньше нашего,
            # то нужно создать новый индекс с правильным лимитом.
            if loaded_max < MAX_ELEMENTS:
                logger.warning(f"[HNSW] Loaded index limit ({loaded_max}) is smaller than required ({MAX_ELEMENTS}). Recreating index.")
                hnsw_index = None
            else:
                return
        except Exception as e:
            logger.error(f"[HNSW] Load failed: {e}. Creating new.")
            hnsw_index = None

    hnsw_index = hnswlib.Index(space=SPACE, dim=DIMENSION)
    hnsw_index.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16)
    hnsw_index.set_ef(50)
    logger.info(f"[HNSW] Index initialized (new). Max Limit: {MAX_ELEMENTS}")

def add_to_hnsw(labels, vectors):
    global hnsw_index, next_hnsw_id, _SAVE_COUNTER
    if hnsw_index is None: return False
    with hnsw_lock:
        try:
            if not isinstance(labels, (list, np.ndarray)):
                labels = [labels]
            labels = list(labels)

            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)

            hnsw_ids = []
            for label in labels:
                label = int(label)
                if label not in id_to_hnsw:
                    next_hnsw_id += 1
                    id_to_hnsw[label] = next_hnsw_id
                    hnsw_to_id[next_hnsw_id] = label
                hnsw_ids.append(id_to_hnsw[label])
            hnsw_ids = np.array(hnsw_ids, dtype=np.int32)

            hnsw_index.add_items(data=vectors, ids=hnsw_ids)
            _save_id_map()

            # 🔹 ПЕРИОДИЧЕСКОЕ СОХРАНЕНИЕ ИНДЕКСА НА ДИСК
            _SAVE_COUNTER += len(labels)
            if _SAVE_COUNTER >= _SAVE_THRESHOLD:
                hnsw_index.save_index(index_path)
                logger.info(f"[HNSW] Auto-saved index to disk (threshold reached: {_SAVE_COUNTER} items).")
                _SAVE_COUNTER = 0

            return True
        except Exception as e:
            logger.error(f"[HNSW] Add error: {e}")
            return False

def save_hnsw():
    """
    Принудительное сохранение индекса и ID-карты на диск.
    Вызывать при завершении сессии, graceful shutdown или явном запросе.
    """
    global _SAVE_COUNTER
    if hnsw_index is None:
        _save_id_map()
        return True

    with hnsw_lock:
        try:
            if _SAVE_COUNTER > 0:
                hnsw_index.save_index(index_path)
                logger.info("[HNSW] Forced flush to disk (remainder saved).")
                _SAVE_COUNTER = 0
            _save_id_map()
            logger.info("[HNSW] Index and ID map saved to disk.")
            return True
        except Exception as e:
            logger.error(f"[HNSW] Save error: {e}")
            return False

def get_nearest_neighbors(query_vector, k=5):
    global hnsw_index
    if hnsw_index is None: return []

    with hnsw_lock:
        try:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array([query_vector], dtype=np.float32)
            labels, distances = hnsw_index.knn_query(data=query_vector, k=k)

            db_ids = [hnsw_to_id.get(lbl, lbl) for lbl in labels[0]]
            return list(zip(db_ids, distances[0]))
        except Exception as e:
            logger.error(f"[HNSW] Search error: {e}")
            return []

# 🔹 Автоматический сброс при graceful shutdown Python (Ctrl+C, exit())
def _register_atexit():
    try:
        atexit.register(save_hnsw)
        logger.info("[HNSW] Registered graceful shutdown handler.")
    except Exception as e:
        logger.warning(f"[HNSW] Failed to register atexit handler: {e}")

_register_atexit()