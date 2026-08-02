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


class FinalAnswerRouteBudgetTests(
    unittest.IsolatedAsyncioTestCase
):
    def test_default_route_policies(self):
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            self.assertEqual(
                routes._resolve_final_answer_token_budget(
                    "FAST"
                ),
                (2048, 4096),
            )
            self.assertEqual(
                routes._resolve_final_answer_token_budget(
                    "THINK"
                ),
                (8192, 12288),
            )
            self.assertEqual(
                routes._resolve_final_answer_token_budget(
                    "CONTINUATION"
                ),
                (2048, 4096),
            )

    def test_policy_environment_override_is_bounded(
        self,
    ):
        with patch.dict(
            os.environ,
            {
                "KVEN2_FINAL_FAST_DEFAULT_TOKENS": (
                    "9000"
                ),
                "KVEN2_FINAL_FAST_MAX_TOKENS": (
                    "3000"
                ),
            },
            clear=True,
        ):
            self.assertEqual(
                routes._resolve_final_answer_token_budget(
                    "FAST"
                ),
                (3000, 3000),
            )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            routes._resolve_final_answer_token_budget(
                "UNKNOWN"
            )

    async def test_fast_route_uses_fast_default(self):
        expected = object()

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=True,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected,
        ) as stream:
            result = await (
                routes
                ._proxy_hybrid_no_tool_final_response(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "test",
                            }
                        ],
                        "stream": True,
                    },
                    "http://backend.invalid",
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                    answer_mode="FAST",
                )
            )

        self.assertIs(result, expected)
        forwarded = stream.call_args.args[0]
        self.assertEqual(
            forwarded["max_tokens"],
            2048,
        )

    async def test_think_route_caps_large_client_budget(
        self,
    ):
        expected = object()

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=True,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected,
        ) as stream:
            result = await (
                routes
                ._proxy_hybrid_no_tool_final_response(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "test",
                            }
                        ],
                        "stream": True,
                        "max_tokens": 24687,
                    },
                    "http://backend.invalid",
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                    answer_mode="THINK",
                )
            )

        self.assertIs(result, expected)
        forwarded = stream.call_args.args[0]
        self.assertEqual(
            forwarded["max_tokens"],
            12288,
        )
        self.assertTrue(
            forwarded["chat_template_kwargs"][
                "enable_thinking"
            ]
        )

    async def test_small_client_budget_remains_small(
        self,
    ):
        expected = object()

        with patch.dict(
            os.environ,
            {
                "KVEN2_MAIN_ENABLE_THINKING": "1",
                "KVEN2_CONDITIONAL_MAIN_THINKING_ENABLED": "1",
            },
            clear=True,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected,
        ) as stream:
            result = await (
                routes
                ._proxy_hybrid_no_tool_final_response(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "test",
                            }
                        ],
                        "stream": True,
                        "max_tokens": 16,
                    },
                    "http://backend.invalid",
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                    answer_mode="FAST",
                )
            )

        self.assertIs(result, expected)
        forwarded = stream.call_args.args[0]
        self.assertEqual(
            forwarded["max_tokens"],
            16,
        )

    def test_main_final_policy_composition(self):
        cases = (
            (
                "FAST",
                False,
                None,
                2048,
            ),
            (
                "THINK",
                True,
                24687,
                12288,
            ),
        )

        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            for (
                policy,
                enable_thinking,
                requested_tokens,
                expected_tokens,
            ) in cases:
                with self.subTest(policy=policy):
                    (
                        default_tokens,
                        max_tokens_cap,
                    ) = (
                        routes
                        ._resolve_final_answer_token_budget(
                            policy
                        )
                    )

                    payload = {
                        "messages": [
                            {
                                "role": "user",
                                "content": "test",
                            }
                        ],
                    }

                    if requested_tokens is not None:
                        payload["max_tokens"] = (
                            requested_tokens
                        )

                    guarded = (
                        routes
                        ._apply_final_answer_safeguards(
                            payload,
                            route_label="main_final",
                            enable_thinking=(
                                enable_thinking
                            ),
                            min_tokens_override=(
                                default_tokens
                            ),
                            max_tokens_cap=(
                                max_tokens_cap
                            ),
                        )
                    )

                    self.assertEqual(
                        guarded["max_tokens"],
                        expected_tokens,
                    )
                    self.assertEqual(
                        guarded[
                            "chat_template_kwargs"
                        ]["enable_thinking"],
                        enable_thinking,
                    )

    def test_speculative_thinking_uses_think_budget(
        self,
    ):
        expected_source = object()
        expected_speculative = object()

        with patch.dict(
            os.environ,
            {},
            clear=True,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected_source,
        ) as stream, patch.object(
            routes,
            "SpeculativeStream",
            return_value=expected_speculative,
        ):
            result = (
                routes
                ._start_speculative_no_tool_stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "test",
                            }
                        ],
                        "stream": True,
                    },
                    "http://backend.invalid",
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                    enable_thinking=True,
                    timeout_seconds=10.0,
                    max_chunks=4,
                )
            )

        self.assertIs(
            result,
            expected_speculative,
        )

        forwarded = stream.call_args.args[0]

        self.assertEqual(
            forwarded["max_tokens"],
            8192,
        )
        self.assertTrue(
            forwarded["chat_template_kwargs"][
                "enable_thinking"
            ]
        )

    async def test_continuation_uses_short_default(
        self,
    ):
        expected = object()

        with patch.dict(
            os.environ,
            {},
            clear=True,
        ), patch.object(
            routes,
            "_stream_main_chat_response",
            return_value=expected,
        ) as stream:
            result = await (
                routes
                ._proxy_hybrid_continuation_final_response(
                    {
                        "messages": [
                            {
                                "role": "tool",
                                "content": "result",
                            }
                        ],
                        "stream": True,
                    },
                    "http://backend.invalid",
                    model_adapter=None,
                    write_path_messages=[],
                    active_state={},
                    owui_rag_meta={},
                    skip_write_path=True,
                )
            )

        self.assertIs(result, expected)
        forwarded = stream.call_args.args[0]
        self.assertEqual(
            forwarded["max_tokens"],
            2048,
        )
        self.assertFalse(
            forwarded["chat_template_kwargs"][
                "enable_thinking"
            ]
        )


if __name__ == "__main__":
    unittest.main()
