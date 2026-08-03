import os
import unittest
from unittest.mock import patch

import routes


class SystemContextBudgetTests(unittest.TestCase):
    def test_token_estimator_rounds_up(self):
        self.assertEqual(
            routes._estimate_text_tokens(
                "abcde",
                chars_per_token=4.0,
            ),
            2,
        )

    def test_core_only_is_preserved_verbatim(self):
        core = "PROFILE\nTOOLS\nACTIVE STATE\n"

        with patch.dict(
            os.environ,
            {
                "KVEN2_SYSTEM_CONTEXT_MAX_TOKENS": "256",
                "KVEN2_SYSTEM_CONTEXT_CHARS_PER_TOKEN": "1",
            },
            clear=False,
        ):
            result, meta = routes._append_bounded_vector_context(
                core,
                "",
            )

        self.assertEqual(result, core)
        self.assertEqual(meta["status"], "core_only")
        self.assertFalse(meta["retrieval_truncated"])
        self.assertFalse(meta["retrieval_omitted"])

    def test_short_retrieval_is_preserved(self):
        core = "PROFILE\nTOOLS\nACTIVE STATE\n"
        retrieval = "Relevant verified memory."

        with patch.dict(
            os.environ,
            {
                "KVEN2_SYSTEM_CONTEXT_MAX_TOKENS": "256",
                "KVEN2_SYSTEM_CONTEXT_CHARS_PER_TOKEN": "1",
            },
            clear=False,
        ):
            result, meta = routes._append_bounded_vector_context(
                core,
                retrieval,
            )

        self.assertTrue(result.startswith(core))
        self.assertIn("VECTOR RETRIEVAL CONTEXT:", result)
        self.assertIn(retrieval, result)
        self.assertEqual(meta["status"], "retrieval_full")
        self.assertEqual(
            meta["retrieval_included_chars"],
            len(retrieval),
        )

    def test_long_retrieval_is_truncated_without_touching_core(self):
        core = (
            "PROFILE-V2\n"
            "CURRENT DATE AND TIME POLICY\n"
            "MEMORY OWNERSHIP POLICY\n"
            "GATEWAY TOOL AVAILABILITY\n"
            "PROJECT CONTEXT\n"
            "ACTIVE STATE\n"
        )
        retrieval = "R" * 500

        with patch.dict(
            os.environ,
            {
                "KVEN2_SYSTEM_CONTEXT_MAX_TOKENS": "256",
                "KVEN2_SYSTEM_CONTEXT_CHARS_PER_TOKEN": "1",
            },
            clear=False,
        ):
            result, meta = routes._append_bounded_vector_context(
                core,
                retrieval,
            )

        self.assertTrue(result.startswith(core))
        self.assertIn("GATEWAY TOOL AVAILABILITY", result)
        self.assertIn("ACTIVE STATE", result)
        self.assertIn(
            "[TRUNCATED: VECTOR RETRIEVAL CONTEXT BUDGET]",
            result,
        )
        self.assertEqual(meta["status"], "retrieval_truncated")
        self.assertTrue(meta["retrieval_truncated"])
        self.assertLessEqual(meta["final_chars"], 256)

    def test_oversized_core_is_preserved_and_retrieval_is_omitted(self):
        core = "C" * 300
        retrieval = "R" * 100

        with patch.dict(
            os.environ,
            {
                "KVEN2_SYSTEM_CONTEXT_MAX_TOKENS": "256",
                "KVEN2_SYSTEM_CONTEXT_CHARS_PER_TOKEN": "1",
            },
            clear=False,
        ):
            result, meta = routes._append_bounded_vector_context(
                core,
                retrieval,
            )

        self.assertEqual(result, core)
        self.assertTrue(meta["core_over_budget"])
        self.assertTrue(meta["retrieval_omitted"])
        self.assertEqual(
            meta["status"],
            "retrieval_omitted_core_budget",
        )


if __name__ == "__main__":
    unittest.main()
