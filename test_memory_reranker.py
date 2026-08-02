import json
import unittest
from unittest.mock import AsyncMock, patch

import memory_reranker
import planner_router


CANDIDATES = [
    {
        "id": 1,
        "content": "The project uses a sliding context window.",
        "type": "Decision",
        "confidence": 0.9,
    },
    {
        "id": 2,
        "content": "Copper does not react in the tested conditions.",
        "type": "Verified Invariant",
        "confidence": 0.8,
    },
]


class MemoryRerankerTests(
    unittest.IsolatedAsyncioTestCase
):
    def test_protocol_accepts_none(self):
        self.assertEqual(
            memory_reranker.parse_memory_selection_protocol(
                "NONE",
                allowed_ids={1, 2},
                max_items=2,
            ),
            [],
        )

    def test_protocol_accepts_ordered_ids(self):
        self.assertEqual(
            memory_reranker.parse_memory_selection_protocol(
                "MEMORY 2,1",
                allowed_ids={1, 2},
                max_items=2,
            ),
            [2, 1],
        )

    def test_protocol_rejects_explanation(self):
        with self.assertRaises(
            planner_router.PlannerRouterError
        ):
            memory_reranker.parse_memory_selection_protocol(
                "MEMORY 1\nBecause it is relevant",
                allowed_ids={1, 2},
                max_items=2,
            )

    def test_protocol_rejects_unknown_id(self):
        with self.assertRaises(
            planner_router.PlannerRouterError
        ):
            memory_reranker.parse_memory_selection_protocol(
                "MEMORY 3",
                allowed_ids={1, 2},
                max_items=2,
            )

    def test_protocol_rejects_duplicates(self):
        with self.assertRaises(
            planner_router.PlannerRouterError
        ):
            memory_reranker.parse_memory_selection_protocol(
                "MEMORY 1,1",
                allowed_ids={1, 2},
                max_items=2,
            )

    def test_protocol_rejects_too_many_ids(self):
        with self.assertRaises(
            planner_router.PlannerRouterError
        ):
            memory_reranker.parse_memory_selection_protocol(
                "MEMORY 1,2",
                allowed_ids={1, 2},
                max_items=1,
            )

    def test_prompt_keeps_query_last_and_omits_confidence(
        self,
    ):
        query = "What did we decide about context?"
        prompt = (
            memory_reranker.build_memory_rerank_prompt(
                query,
                CANDIDATES,
                max_items=2,
            )
        )

        self.assertTrue(
            prompt.endswith(
                json.dumps(
                    query,
                    ensure_ascii=False,
                )
            )
        )
        self.assertNotIn(
            '"confidence"',
            prompt,
        )
        self.assertIn(
            '"id":1',
            prompt,
        )
        self.assertIn(
            '"id":2',
            prompt,
        )

    async def test_selects_planner_ids(self):
        post = AsyncMock(
            return_value=(
                "MEMORY 2,1",
                {
                    "elapsed_seconds": 0.1,
                },
            )
        )

        with patch.object(
            memory_reranker.planner_router,
            "_post_planner_text",
            post,
        ):
            result = await (
                memory_reranker.select_relevant_memories(
                    "query",
                    CANDIDATES,
                    max_items=2,
                    timeout_seconds=7.0,
                )
            )

        self.assertEqual(
            result["status"],
            "selected",
        )
        self.assertEqual(
            result["selected_ids"],
            [2, 1],
        )
        self.assertEqual(
            post.await_args.kwargs["max_tokens"],
            memory_reranker.RERANK_MAX_TOKENS,
        )
        self.assertEqual(
            post.await_args.kwargs[
                "timeout_seconds"
            ],
            7.0,
        )

    async def test_invalid_output_fails_closed(self):
        post = AsyncMock(
            return_value=(
                "MEMORY 99",
                {},
            )
        )

        with patch.object(
            memory_reranker.planner_router,
            "_post_planner_text",
            post,
        ):
            result = await (
                memory_reranker.select_relevant_memories(
                    "query",
                    CANDIDATES,
                    max_items=2,
                    timeout_seconds=7.0,
                )
            )

        self.assertEqual(
            result["status"],
            "error",
        )
        self.assertEqual(
            result["selected_ids"],
            [],
        )

    async def test_empty_candidates_skip_planner(self):
        post = AsyncMock()

        with patch.object(
            memory_reranker.planner_router,
            "_post_planner_text",
            post,
        ):
            result = await (
                memory_reranker.select_relevant_memories(
                    "query",
                    [],
                    max_items=2,
                    timeout_seconds=7.0,
                )
            )

        post.assert_not_awaited()
        self.assertEqual(
            result["status"],
            "none",
        )


if __name__ == "__main__":
    unittest.main()
