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

DIMENSION = 768
SPACE = 'cosine'

# This is an initial allocation size, not a lifetime record limit.
# The index grows geometrically before new items exceed its capacity.
DEFAULT_INITIAL_CAPACITY = 10_000
GROWTH_FACTOR = 2


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "[HNSW] Invalid %s=%r; using default=%s.",
            name,
            raw_value,
            default,
        )
        return default

    if value < 1:
        logger.warning(
            "[HNSW] Non-positive %s=%r; using default=%s.",
            name,
            raw_value,
            default,
        )
        return default

    return value


INITIAL_CAPACITY = _read_positive_int_env(
    "KVEN2_HNSW_INITIAL_CAPACITY",
    DEFAULT_INITIAL_CAPACITY,
)

hnsw_index = None
id_to_hnsw = {}   # db_id -> hnsw_id
hnsw_to_id = {}   # hnsw_id -> db_id
next_hnsw_id = 0
hnsw_lock = threading.Lock()

# Для текущих объёмов важнее надёжность, чем экономия дисковых операций.
# Старое значение 100 приводило к состоянию: id_map уже сохранён, а index.bin ещё нет.
_SAVE_COUNTER = 0
_SAVE_THRESHOLD = 1


def _reset_id_map(save_empty: bool = True):
    """Сбрасывает карту соответствия SQLite id <-> HNSW internal id."""
    global id_to_hnsw, hnsw_to_id, next_hnsw_id
    id_to_hnsw = {}
    hnsw_to_id = {}
    next_hnsw_id = 0
    logger.warning("[HNSW] ID map reset.")

    if save_empty:
        _save_id_map()


def _load_id_map() -> bool:
    """
    Загружает id_map только после успешной загрузки бинарного HNSW index.
    Возвращает True, если файл карты найден и успешно прочитан.
    """
    global id_to_hnsw, hnsw_to_id, next_hnsw_id

    if not os.path.exists(id_map_path):
        logger.warning(f"[HNSW] ID map file not found: {id_map_path}")
        _reset_id_map(save_empty=False)
        return False

    try:
        with open(id_map_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("ID map JSON must be an object/dict")

        id_to_hnsw = {int(k): int(v) for k, v in data.items()}
        hnsw_to_id = {int(v): int(k) for k, v in id_to_hnsw.items()}
        next_hnsw_id = max(hnsw_to_id.keys(), default=0)
        logger.info(f"[HNSW] ID map loaded. {len(id_to_hnsw)} entries.")
        return True

    except Exception as e:
        logger.error(f"[HNSW] Load ID map failed: {e}", exc_info=True)
        _reset_id_map(save_empty=False)
        return False


def _save_id_map() -> bool:
    try:
        os.makedirs(os.path.dirname(id_map_path) or ".", exist_ok=True)
        with open(id_map_path, 'w') as f:
            json.dump(id_to_hnsw, f)
        return True
    except Exception as e:
        logger.error(f"[HNSW] Save ID map failed: {e}", exc_info=True)
        return False


def _next_capacity(
    required_count: int,
    current_capacity: int,
) -> int:
    """Return a geometrically grown capacity with no application cap."""
    required = max(0, int(required_count))
    current = max(0, int(current_capacity))

    if required <= current:
        return current

    capacity = max(
        current,
        INITIAL_CAPACITY,
        1,
    )

    while capacity < required:
        capacity = max(
            capacity + 1,
            capacity * GROWTH_FACTOR,
        )

    return capacity


def _ensure_capacity(required_count: int) -> int:
    """Expand the active index before adding new records."""
    if hnsw_index is None:
        raise RuntimeError(
            "Cannot resize an uninitialized HNSW index"
        )

    current_capacity = int(
        hnsw_index.get_max_elements()
    )
    target_capacity = _next_capacity(
        required_count,
        current_capacity,
    )

    if target_capacity > current_capacity:
        hnsw_index.resize_index(target_capacity)
        logger.info(
            "[HNSW] Capacity expanded: old=%s new=%s required=%s",
            current_capacity,
            target_capacity,
            required_count,
        )

    return target_capacity


def _create_new_index(persist_empty: bool = True):
    """Создаёт новый пустой HNSW index с текущими параметрами."""
    global hnsw_index, _SAVE_COUNTER

    hnsw_index = hnswlib.Index(space=SPACE, dim=DIMENSION)
    hnsw_index.init_index(
        max_elements=INITIAL_CAPACITY,
        ef_construction=200,
        M=16,
    )
    hnsw_index.set_ef(50)
    _SAVE_COUNTER = 0
    logger.info(
        "[HNSW] Index initialized (new). Initial Capacity: %s",
        INITIAL_CAPACITY,
    )

    if persist_empty:
        try:
            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            hnsw_index.save_index(index_path)
            logger.info(f"[HNSW] Empty index saved to disk: {index_path}")
        except Exception as e:
            logger.error(f"[HNSW] Failed to save empty index: {e}", exc_info=True)


def _check_integrity() -> bool:
    """
    Проверяет базовую целостность пары index.bin + id_map.
    Для текущей схемы количество элементов в HNSW должно совпадать с числом записей id_map.
    """
    if hnsw_index is None:
        logger.warning("[HNSW] Integrity check failed: index is None.")
        return False

    try:
        index_count = int(hnsw_index.get_current_count())
        map_count = int(len(id_to_hnsw))
        logger.info(f"[HNSW] Integrity: index_count={index_count}, id_map_entries={map_count}")

        if index_count != map_count:
            logger.warning(
                f"[HNSW] Integrity mismatch: index_count={index_count}, "
                f"id_map_entries={map_count}. Index/map pair is unsafe."
            )
            return False

        return True

    except Exception as e:
        logger.error(f"[HNSW] Integrity check error: {e}", exc_info=True)
        return False


def init_hnsw():
    """
    Инициализация HNSW.

    Важное правило целостности:
    id_map имеет смысл только вместе с успешно загруженным бинарным index.bin.
    Если index.bin отсутствует, повреждён или несовместим — создаём новый index и сбрасываем id_map.
    """
    global hnsw_index

    logger.info("[HNSW] Initializing index...")
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

    # В случае symlink на отсутствующий target os.path.exists вернёт False — это нужное поведение.
    if os.path.exists(index_path):
        try:
            hnsw_index = hnswlib.Index(space=SPACE, dim=DIMENSION)
            hnsw_index.load_index(index_path)
            hnsw_index.set_ef(50)

            loaded_capacity = int(
                hnsw_index.get_max_elements()
            )
            loaded_count = int(
                hnsw_index.get_current_count()
            )

            logger.info(
                "[HNSW] Index loaded. Current: %s, Capacity: %s",
                loaded_count,
                loaded_capacity,
            )

            # hnswlib 0.8.0 does not persist max_elements for an
            # empty index. Reload it with an explicit runtime capacity.
            if loaded_count == 0 and loaded_capacity == 0:
                logger.info(
                    "[HNSW] Empty index has no persisted capacity; "
                    "reloading with initial capacity=%s.",
                    INITIAL_CAPACITY,
                )

                hnsw_index = hnswlib.Index(
                    space=SPACE,
                    dim=DIMENSION,
                )
                hnsw_index.load_index(
                    index_path,
                    max_elements=INITIAL_CAPACITY,
                )
                hnsw_index.set_ef(50)

                loaded_capacity = int(
                    hnsw_index.get_max_elements()
                )

            if loaded_capacity < loaded_count:
                raise RuntimeError(
                    "Loaded HNSW capacity is smaller than its "
                    f"record count: capacity={loaded_capacity}, "
                    f"count={loaded_count}"
                )

            # Preserve every existing record and enlarge the index
            # in place when an older index has a smaller capacity.
            if loaded_capacity < INITIAL_CAPACITY:
                previous_capacity = loaded_capacity

                hnsw_index.resize_index(
                    INITIAL_CAPACITY
                )
                loaded_capacity = int(
                    hnsw_index.get_max_elements()
                )

                if loaded_count > 0:
                    hnsw_index.save_index(index_path)

                logger.info(
                    "[HNSW] Loaded index capacity expanded: "
                    "old=%s new=%s count=%s",
                    previous_capacity,
                    loaded_capacity,
                    loaded_count,
                )

            map_loaded = _load_id_map()
            if not map_loaded:
                logger.warning("[HNSW] Index exists but ID map is missing/broken. Recreating index to avoid wrong DB mappings.")
                _reset_id_map(save_empty=True)
                _create_new_index(persist_empty=True)
                _check_integrity()
                return

            if not _check_integrity():
                logger.warning("[HNSW] Recreating index and resetting ID map due to integrity mismatch.")
                _reset_id_map(save_empty=True)
                _create_new_index(persist_empty=True)
                _check_integrity()
                return

            logger.info("[HNSW] Index/id_map pair loaded successfully.")
            return

        except Exception as e:
            logger.error(f"[HNSW] Load failed: {e}. Creating new index and resetting ID map.", exc_info=True)
            hnsw_index = None
            _reset_id_map(save_empty=True)
            _create_new_index(persist_empty=True)
            _check_integrity()
            return

    logger.warning(f"[HNSW] Index file not found: {index_path}. Creating new empty index and resetting ID map.")
    _reset_id_map(save_empty=True)
    _create_new_index(persist_empty=True)
    _check_integrity()


def add_to_hnsw(labels, vectors):
    global hnsw_index, next_hnsw_id, _SAVE_COUNTER
    if hnsw_index is None:
        logger.warning("[HNSW] Add skipped: index is not initialized.")
        return False

    with hnsw_lock:
        try:
            if not isinstance(labels, (list, tuple, np.ndarray)):
                labels = [labels]
            labels = [int(label) for label in list(labels)]

            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            if vectors.shape[0] != len(labels):
                raise ValueError(
                    f"labels/vectors count mismatch: labels={len(labels)}, vectors={vectors.shape[0]}"
                )

            if vectors.shape[1] != DIMENSION:
                raise ValueError(
                    f"vector dimension mismatch: got={vectors.shape[1]}, expected={DIMENSION}"
                )

            new_labels = {
                label
                for label in labels
                if label not in id_to_hnsw
            }
            required_count = (
                int(hnsw_index.get_current_count())
                + len(new_labels)
            )

            _ensure_capacity(required_count)

            hnsw_ids = []

            for label in labels:
                if label not in id_to_hnsw:
                    next_hnsw_id += 1
                    id_to_hnsw[label] = next_hnsw_id
                    hnsw_to_id[next_hnsw_id] = label

                hnsw_ids.append(
                    id_to_hnsw[label]
                )

            # hnswlib accepts 64-bit labels. Avoid introducing an
            # artificial signed 32-bit identifier ceiling.
            hnsw_ids = np.asarray(
                hnsw_ids,
                dtype=np.int64,
            )

            hnsw_index.add_items(data=vectors, ids=hnsw_ids)
            _SAVE_COUNTER += len(labels)

            _save_id_map()

            # В текущих объёмах сохраняем индекс сразу, чтобы не было orphan id_map без index.bin.
            if _SAVE_COUNTER >= _SAVE_THRESHOLD:
                hnsw_index.save_index(index_path)
                logger.info(f"[HNSW] Auto-saved index to disk ({_SAVE_COUNTER} new item(s)).")
                _SAVE_COUNTER = 0

            _check_integrity()
            return True

        except Exception as e:
            logger.error(f"[HNSW] Add error: {e}", exc_info=True)
            return False


def save_hnsw():
    """
    Принудительное сохранение индекса и ID-карты на диск.
    Вызывать при завершении сессии, graceful shutdown или явном запросе.
    """
    global _SAVE_COUNTER

    with hnsw_lock:
        try:
            if hnsw_index is None:
                logger.warning(
                    "[HNSW] save_hnsw skipped: index is None; "
                    "refusing to write an ID map without its index."
                )
                return False

            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            hnsw_index.save_index(index_path)
            _SAVE_COUNTER = 0
            logger.info("[HNSW] Index saved to disk.")

            if not _save_id_map():
                logger.error("[HNSW] ID map save failed.")
                return False

            logger.info("[HNSW] Index and ID map saved to disk.")
            return True

        except Exception as e:
            logger.error(f"[HNSW] Save error: {e}", exc_info=True)
            return False


def get_nearest_neighbors(query_vector, k=5):
    """
    Возвращает до k ближайших соседей как список пар (db_id, distance).

    Важно: hnswlib падает, если запросить k больше, чем элементов в индексе.
    Поэтому k ограничивается фактическим размером индекса.
    Фильтрация по смысловой близости выполняется выше, в retrieval.py.
    """
    global hnsw_index
    if hnsw_index is None:
        return []

    with hnsw_lock:
        try:
            current_count = int(hnsw_index.get_current_count())
            if current_count <= 0:
                logger.debug("[HNSW] Search skipped: index is empty.")
                return []

            safe_k = min(int(k), current_count)
            if safe_k <= 0:
                return []

            query_vector = np.asarray(query_vector, dtype=np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            if query_vector.shape[1] != DIMENSION:
                raise ValueError(
                    f"query vector dimension mismatch: got={query_vector.shape[1]}, expected={DIMENSION}"
                )

            labels, distances = hnsw_index.knn_query(data=query_vector, k=safe_k)

            results = []
            unmapped = 0
            for lbl, dist in zip(labels[0], distances[0]):
                hnsw_id = int(lbl)
                db_id = hnsw_to_id.get(hnsw_id)
                if db_id is None:
                    unmapped += 1
                    logger.warning(f"[HNSW] Search result skipped: hnsw_id={hnsw_id} has no DB mapping.")
                    continue
                results.append((int(db_id), float(dist)))

            if unmapped:
                logger.warning(f"[HNSW] Search returned {unmapped} unmapped neighbor(s).")

            return results

        except Exception as e:
            logger.error(f"[HNSW] Search error: {e}", exc_info=True)
            return []


# Автоматический сброс при graceful shutdown Python (Ctrl+C, exit())
def _register_atexit():
    try:
        atexit.register(save_hnsw)
        logger.info("[HNSW] Registered graceful shutdown handler.")
    except Exception as e:
        logger.warning(f"[HNSW] Failed to register atexit handler: {e}")


_register_atexit()
