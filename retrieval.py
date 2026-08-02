import sys
import os
import logging
import asyncio

sys.path.append('/opt/kven2')
import hnsw
import embedder
from sqlite import query
from memory_reranker import select_relevant_memories

logger = logging.getLogger("Kven.Retrieval")

# --- НАСТРОЙКИ ФИЛЬТРАЦИИ И ВЕСОВ ---
# Для hnswlib SPACE='cosine' distance обычно соответствует 1 - cosine_similarity.
# Чем меньше distance, тем ближе запись.
#
# ВАЖНО: один глобальный порог плохо работает для разреженной памяти.
# Поэтому используем детерминированную классификацию запроса:
#   skip         -> retrieval не нужен, возвращаем []
#   normal       -> строгий порог, чтобы не загрязнять prompt случайной памятью
#   memory_query -> ослабленный порог для явных вопросов "что ты помнишь?"
STRICT_MAX_DISTANCE = 0.42
MEMORY_QUERY_MAX_DISTANCE = 0.62

TOP_K_RAW = 20
TOP_K_FINAL = 5
SIMILARITY_WEIGHT = 0.8
CONFIDENCE_WEIGHT = 0.2


# Короткие технические тесты. Для них retrieval чаще вреден, чем полезен.
_SKIP_MARKERS = (
    "ответь одним словом",
    "ответь только одним словом",
    "одним словом",
    "работает",
    "проверка",
    "тест",
    "test",
    "ping",
    "pong",
    "hello",
)

# Явные запросы к памяти / текущему состоянию проекта.
# Это простая детерминированная эвристика, без вызова малой модели.
_MEMORY_QUERY_MARKERS = (
    "помнишь",
    "вспомни",
    "что ты помнишь",
    "что известно",
    "что мы решили",
    "что мы зафиксировали",
    "что зафиксировано",
    "точка отсч",
    "чистая точка",
    "baseline",
    "память",
    "memory",
    "owui",
    "routes.py",
    "hnsw",
)

_TRUE_ENV_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}


def _read_bool_env(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return (
        raw_value.strip().lower()
        in _TRUE_ENV_VALUES
    )


def _read_int_env(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name)

    try:
        value = (
            int(raw_value)
            if raw_value is not None
            else int(default)
        )
    except (TypeError, ValueError):
        value = int(default)

    return max(
        minimum,
        min(maximum, value),
    )


def _read_float_env(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(name)

    try:
        value = (
            float(raw_value)
            if raw_value is not None
            else float(default)
        )
    except (TypeError, ValueError):
        value = float(default)

    return max(
        minimum,
        min(maximum, value),
    )



def classify_retrieval_query(query_text: str) -> str:
    """
    Возвращает один из режимов retrieval:
      - skip: не искать память вообще;
      - memory_query: явный запрос к памяти, можно ослабить distance threshold;
      - normal: обычный запрос, используем строгий threshold.

    Это намеренно не LLM-классификация, а быстрые правила по строке.
    Поведение должно быть предсказуемым и легко отлаживаемым по логам.
    """
    q = (query_text or "").strip().lower()
    if not q:
        return "skip"

    # Короткие smoke-тесты не должны тянуть память в prompt.
    if len(q) < 100 and any(marker in q for marker in _SKIP_MARKERS):
        return "skip"

    if any(marker in q for marker in _MEMORY_QUERY_MARKERS):
        return "memory_query"

    return "normal"


def max_distance_for_mode(mode: str) -> float:
    if mode == "memory_query":
        return MEMORY_QUERY_MAX_DISTANCE
    return STRICT_MAX_DISTANCE


async def _fetch_memory_row(db_id: int):
    """
    Возвращает строку памяти по db_id.
    Основной индекс должен указывать на semantic_memory.
    Fallback на episodic_memory оставлен для совместимости со старой схемой/тестами.
    Формат результата: (id, content, epistemic_type, confidence)
    """
    row = await query(
        "SELECT id, content, epistemic_type, importance as confidence FROM semantic_memory WHERE id = ? AND deleted = 0",
        (db_id,)
    )
    if row:
        return row[0]

    # Fallback: если episodic_memory не содержит epistemic_type, используем литерал.
    try:
        row = await query(
            "SELECT id, content, 'episodic' as epistemic_type, importance as confidence FROM episodic_memory WHERE id = ?",
            (db_id,)
        )
        if row:
            return row[0]
    except Exception as e:
        logger.debug(f"[RETRIEVAL] Episodic fallback failed for id={db_id}: {e}")

    return None


async def retrieve_context(query_text: str, top_k_raw=TOP_K_RAW, top_k_final=TOP_K_FINAL):
    """
    Асинхронное извлечение контекста с динамическим distance gate.

    Поток:
    1. Детерминированно классифицируем запрос: skip / normal / memory_query.
    2. Для skip сразу возвращаем [].
    3. Генерируем embedding запроса.
    4. Берём HNSW top_k_raw.
    5. Фильтруем по threshold, зависящему от режима.
    6. Достаём строки из SQLite.
    7. When enabled, let the planner select directly relevant IDs.
    8. Otherwise preserve the legacy similarity/confidence ranking.
    """
    if not query_text or not query_text.strip():
        logger.info("[RETRIEVAL_MODE] mode=skip reason=empty_query")
        return []

    mode = classify_retrieval_query(query_text)
    if mode == "skip":
        logger.info(
            "[RETRIEVAL_MODE] mode=skip reason=technical_or_short_test "
            f"query='{query_text[:80]}'"
        )
        return []

    max_distance = max_distance_for_mode(mode)
    logger.info(
        f"[RETRIEVAL_MODE] mode={mode} max_distance={max_distance} "
        f"query='{query_text[:120]}'"
    )

    try:
        logger.debug(f"[RETRIEVAL] Generating embedding for query: '{query_text[:80]}...'")

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
        candidates = hnsw.get_nearest_neighbors(query_vector, k=top_k_raw)
        if not candidates:
            logger.info(
                f"[RETRIEVAL_SUMMARY] mode={mode} | "
                "No HNSW neighbors found. Memory injected: 0"
            )
            return []

        # 3. Сбор контекста и комбинированное ранжирование
        retrieved_items = []
        rejected_by_distance = 0
        missing_db_rows = 0
        best_distance = None
        rejected_preview = []

        for db_id, distance in candidates:
            db_id = int(db_id)
            distance = float(distance)
            best_distance = distance if best_distance is None else min(best_distance, distance)

            row = None
            if distance > max_distance:
                rejected_by_distance += 1

                # Для отладки показываем несколько ближайших отвергнутых кандидатов.
                # Это помогает калибровать threshold без blind guessing.
                if len(rejected_preview) < 5:
                    row = await _fetch_memory_row(db_id)
                    if row:
                        rejected_preview.append(
                            f"ID:{db_id} DIST:{distance:.4f} TEXT:{str(row[1])[:70]}..."
                        )
                    else:
                        rejected_preview.append(
                            f"ID:{db_id} DIST:{distance:.4f} TEXT:<missing db row>"
                        )

                logger.debug(
                    f"[RETRIEVAL] Record {db_id} skipped "
                    f"(distance={distance:.4f} > {max_distance})"
                )
                continue

            # 4. Асинхронный запрос к БД
            row = await _fetch_memory_row(db_id)
            if not row:
                missing_db_rows += 1
                logger.debug(f"[RETRIEVAL] Record {db_id} skipped: no matching DB row.")
                continue

            confidence = row[3] if row[3] is not None else 0.5

            # --- РАСЧЁТ КОМБИНИРОВАННОГО СКОРА ---
            # Нормализуем cosine distance в similarity: distance≈0 => similarity≈1, distance≈1 => similarity≈0.
            similarity = max(0.0, 1.0 - distance)
            # Взвешенная сумма: 80% релевантности, 20% важности.
            score = (similarity * SIMILARITY_WEIGHT) + (confidence * CONFIDENCE_WEIGHT)

            token_est = len(row[1]) // 4
            logger.info(
                f"[RETRIEVAL_AUDIT] "
                f"MODE:{mode} | "
                f"ID:{db_id} | "
                f"DIST:{distance:.4f} | "
                f"SIM:{similarity:.4f} | "
                f"CONF:{confidence:.2f} | "
                f"SCORE:{score:.4f} | "
                f"TOKENS:{token_est} | "
                f"TEXT:{row[1][:80]}..."
            )

            retrieved_items.append({
                "id": row[0],
                "content": row[1],
                "type": row[2],
                "confidence": confidence,
                "distance": distance,
                "similarity": similarity,
                "score": score,
                "tokens": token_est,
                "retrieval_mode": mode,
            })

        rerank_enabled = _read_bool_env(
            "KVEN2_RAG_PLANNER_RERANK_ENABLED",
            False,
        )
        rerank_status = "disabled"
        rerank_error = ""
        rerank_meta = {}
        selected_ids = []

        if rerank_enabled:
            if not retrieved_items:
                rerank_status = "none"
                final_results = []
            else:
                candidate_limit = _read_int_env(
                    "KVEN2_RAG_PLANNER_CANDIDATES",
                    12,
                    1,
                    max(1, int(top_k_raw)),
                )
                # This is a safety ceiling for the experimental
                # reranker, not an acceptable latency target.
                rerank_timeout = _read_float_env(
                    "KVEN2_RAG_PLANNER_RERANK_TIMEOUT",
                    15.0,
                    0.5,
                    60.0,
                )
                planner_candidates = retrieved_items[
                    :candidate_limit
                ]

                rerank_result = (
                    await select_relevant_memories(
                        query_text,
                        planner_candidates,
                        max_items=top_k_final,
                        timeout_seconds=rerank_timeout,
                    )
                )

                rerank_status = str(
                    rerank_result.get("status")
                    or "error"
                )
                rerank_error = str(
                    rerank_result.get("error")
                    or ""
                )[:500]
                rerank_meta = (
                    rerank_result.get("meta")
                    if isinstance(
                        rerank_result.get("meta"),
                        dict,
                    )
                    else {}
                )

                try:
                    selected_ids = [
                        int(memory_id)
                        for memory_id
                        in (
                            rerank_result.get(
                                "selected_ids"
                            )
                            or []
                        )
                    ]
                except (TypeError, ValueError):
                    selected_ids = []
                    rerank_status = "error"
                    rerank_error = (
                        "planner returned invalid "
                        "memory identifiers"
                    )

                candidates_by_id = {
                    int(item["id"]): item
                    for item in planner_candidates
                }
                invalid_selection = (
                    len(selected_ids)
                    > int(top_k_final)
                    or len(selected_ids)
                    != len(set(selected_ids))
                    or any(
                        memory_id
                        not in candidates_by_id
                        for memory_id
                        in selected_ids
                    )
                )

                if (
                    rerank_status == "selected"
                    and not invalid_selection
                ):
                    final_results = [
                        candidates_by_id[memory_id]
                        for memory_id in selected_ids
                    ]
                elif rerank_status == "none":
                    final_results = []
                else:
                    final_results = []

                    if invalid_selection:
                        rerank_status = "error"
                        rerank_error = (
                            "planner selection failed "
                            "defensive validation"
                        )

                logger.info(
                    "[RETRIEVAL_RERANK] "
                    "enabled=True status=%s "
                    "candidates=%s selected_ids=%s "
                    "timeout=%s error=%s meta=%s",
                    rerank_status,
                    len(planner_candidates),
                    selected_ids,
                    rerank_timeout,
                    rerank_error,
                    rerank_meta,
                )
        else:
            # Preserve the previous behavior until the
            # planner reranker is explicitly enabled.
            retrieved_items.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            final_results = retrieved_items[
                :top_k_final
            ]

        total_mem_tokens = sum(
            item.get("tokens", 0)
            for item in final_results
        )

        logger.info(
            f"[RETRIEVAL_SUMMARY] "
            f"mode={mode} | "
            f"Candidates: {len(candidates)} | "
            f"Accepted: {len(retrieved_items)} | "
            f"Injected: {len(final_results)} | "
            f"Rejected by distance: {rejected_by_distance} | "
            f"Missing DB rows: {missing_db_rows} | "
            f"Best distance: {best_distance if best_distance is not None else 'n/a'} | "
            f"Max distance: {max_distance} | "
            f"Memory tokens injected: {total_mem_tokens} | "
            f"Planner rerank: {rerank_status} | "
            f"Planner error: {rerank_error}"
        )

        if rejected_preview:
            logger.info(
                "[RETRIEVAL_REJECTED_PREVIEW] "
                + " || ".join(rejected_preview)
            )

        return final_results

    except Exception as e:
        logger.error(f"[RETRIEVAL] Error during retrieval: {e}", exc_info=True)
        return []
