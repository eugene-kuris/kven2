import atexit
import os
import unittest
from unittest.mock import AsyncMock, patch

import hnsw
import retrieval


atexit.unregister(hnsw.save_hnsw)


ROWS = {
    1: (
        1,
        "Nearest by vector but not always preferred.",
        "Verified Invariant",
        0.1,
    ),
    2: (
        2,
        "A directly relevant durable decision.",
        "Decision",
        1.0,
    ),
    3: (
        3,
        "Another candidate.",
        "Procedure",
        0.5,
    ),
}


class RetrievalPlannerRerankTests(
    unittest.IsolatedAsyncioTestCase
):
    async def _retrieve(
        self,
        *,
        enabled: bool,
        rerank_result: dict | None = None,
        candidate_limit: int = 12,
    ):
        selector = AsyncMock(
            return_value=(
                rerank_result
                or {
                    "status": "none",
                    "selected_ids": [],
                    "meta": {},
                    "error": "",
                }
            )
        )

        async def fetch_row(memory_id):
            return ROWS[int(memory_id)]

        environment = {
            "KVEN2_RAG_PLANNER_RERANK_ENABLED": (
                "1" if enabled else "0"
            ),
            "KVEN2_RAG_PLANNER_CANDIDATES": str(
                candidate_limit
            ),
            "KVEN2_RAG_PLANNER_RERANK_TIMEOUT": "7",
        }

        with patch.dict(
            os.environ,
            environment,
            clear=False,
        ), patch.object(
            retrieval.embedder,
            "get_embedding",
            AsyncMock(
                return_value=[0.0] * 768
            ),
        ), patch.object(
            retrieval.hnsw,
            "hnsw_index",
            object(),
        ), patch.object(
            retrieval.hnsw,
            "get_nearest_neighbors",
            return_value=[
                (1, 0.10),
                (2, 0.20),
                (3, 0.30),
            ],
        ), patch.object(
            retrieval,
            "_fetch_memory_row",
            side_effect=fetch_row,
        ), patch.object(
            retrieval,
            "select_relevant_memories",
            selector,
        ):
            result = await retrieval.retrieve_context(
                "Что мы решили об архитектуре?",
                top_k_raw=3,
                top_k_final=2,
            )

        return result, selector

    async def test_selected_ids_define_final_order(self):
        result, selector = await self._retrieve(
            enabled=True,
            rerank_result={
                "status": "selected",
                "selected_ids": [2, 1],
                "meta": {},
                "error": "",
            },
        )

        self.assertEqual(
            [item["id"] for item in result],
            [2, 1],
        )
        selector.assert_awaited_once()

    async def test_none_injects_no_memory(self):
        result, _ = await self._retrieve(
            enabled=True,
            rerank_result={
                "status": "none",
                "selected_ids": [],
                "meta": {},
                "error": "",
            },
        )

        self.assertEqual(result, [])

    async def test_error_fails_closed(self):
        result, _ = await self._retrieve(
            enabled=True,
            rerank_result={
                "status": "error",
                "selected_ids": [],
                "meta": {},
                "error": "planner unavailable",
            },
        )

        self.assertEqual(result, [])

    async def test_unknown_selected_id_fails_closed(self):
        result, _ = await self._retrieve(
            enabled=True,
            rerank_result={
                "status": "selected",
                "selected_ids": [99],
                "meta": {},
                "error": "",
            },
        )

        self.assertEqual(result, [])

    async def test_candidate_limit_is_enforced(self):
        _, selector = await self._retrieve(
            enabled=True,
            rerank_result={
                "status": "none",
                "selected_ids": [],
                "meta": {},
                "error": "",
            },
            candidate_limit=2,
        )

        candidates = (
            selector.await_args.args[1]
        )

        self.assertEqual(
            [item["id"] for item in candidates],
            [1, 2],
        )

    async def test_disabled_flag_preserves_legacy_score(self):
        result, selector = await self._retrieve(
            enabled=False,
        )

        selector.assert_not_awaited()

        # Candidate 2 has a lower vector similarity but
        # wins under the legacy confidence-weighted score.
        self.assertEqual(
            [item["id"] for item in result],
            [2, 1],
        )


if __name__ == "__main__":
    unittest.main()
