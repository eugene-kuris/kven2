import atexit
import os
import unittest
from unittest.mock import patch

import hnsw
import routes


try:
    atexit.unregister(hnsw.save_hnsw)
except ValueError:
    pass


class FinalAnswerBudgetTests(unittest.TestCase):
    def apply_guard(
        self,
        payload,
        *,
        default_tokens="1024",
        min_tokens_override=None,
        max_tokens_cap=None,
    ):
        with patch.dict(
            os.environ,
            {
                "KVEN2_FINAL_MIN_TOKENS": (
                    default_tokens
                )
            },
            clear=False,
        ):
            return routes._apply_final_answer_safeguards(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "test",
                        }
                    ],
                    **payload,
                },
                route_label="budget_test",
                enable_thinking=False,
                min_tokens_override=(
                    min_tokens_override
                ),
                max_tokens_cap=max_tokens_cap,
            )

    def test_explicit_max_tokens_is_preserved(self):
        guarded = self.apply_guard(
            {"max_tokens": 16}
        )

        self.assertEqual(guarded["max_tokens"], 16)

    def test_explicit_max_completion_tokens_is_preserved(
        self,
    ):
        guarded = self.apply_guard(
            {"max_completion_tokens": 32}
        )

        self.assertEqual(guarded["max_tokens"], 32)
        self.assertNotIn(
            "max_completion_tokens",
            guarded,
        )

    def test_missing_budget_uses_configured_default(self):
        guarded = self.apply_guard({})

        self.assertEqual(
            guarded["max_tokens"],
            1024,
        )

    def test_cap_reduces_large_requested_budget(self):
        guarded = self.apply_guard(
            {"max_tokens": 24687},
            min_tokens_override=256,
            max_tokens_cap=2048,
        )

        self.assertEqual(
            guarded["max_tokens"],
            2048,
        )

    def test_cap_does_not_raise_small_client_budget(self):
        guarded = self.apply_guard(
            {"max_tokens": 16},
            min_tokens_override=256,
            max_tokens_cap=2048,
        )

        self.assertEqual(guarded["max_tokens"], 16)

    def test_budget_source_reports_only_real_capping(
        self,
    ):
        with self.assertLogs(
            routes.logger,
            level="INFO",
        ) as captured:
            guarded = self.apply_guard(
                {"max_tokens": 16},
                max_tokens_cap=2048,
            )

        self.assertEqual(guarded["max_tokens"], 16)
        uncapped_log = "\n".join(captured.output)
        self.assertIn(
            "budget_source=client ",
            uncapped_log,
        )
        self.assertNotIn(
            "budget_source=client_capped",
            uncapped_log,
        )

        with self.assertLogs(
            routes.logger,
            level="INFO",
        ) as captured:
            guarded = self.apply_guard(
                {"max_tokens": 24687},
                max_tokens_cap=2048,
            )

        self.assertEqual(guarded["max_tokens"], 2048)
        capped_log = "\n".join(captured.output)
        self.assertIn(
            "budget_source=client_capped",
            capped_log,
        )

    def test_missing_default_can_be_reduced_by_cap(
        self,
    ):
        guarded = self.apply_guard(
            {},
            default_tokens="4096",
            max_tokens_cap=2048,
        )

        self.assertEqual(guarded["max_tokens"], 2048)

    def test_invalid_or_non_positive_budget_uses_default(
        self,
    ):
        for value in (
            0,
            -1,
            "invalid",
            True,
        ):
            with self.subTest(value=value):
                guarded = self.apply_guard(
                    {"max_tokens": value}
                )

                self.assertEqual(
                    guarded["max_tokens"],
                    1024,
                )


if __name__ == "__main__":
    unittest.main()
