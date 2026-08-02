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


class ContextBudgetTelemetryTests(
    unittest.TestCase
):
    @staticmethod
    def _guard_payload():
        return routes._apply_final_answer_safeguards(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "SECRET-SYSTEM-CONTENT"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "SECRET-USER-CONTENT"
                        ),
                    },
                ],
                "max_tokens": 50000,
            },
            route_label="main_final",
            enable_thinking=True,
            min_tokens_override=8192,
            max_tokens_cap=4096,
        )

    def test_budget_report_is_disabled_by_default(
        self,
    ):
        with patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            for name in (
                "KVEN2_CONTEXT_BUDGET_REPORT_ENABLED",
                "KVEN2_CONTEXT_BUDGET_TOKENS",
                "KVEN2_CONTEXT_BUDGET_SUMMARY_TARGET_TOKENS",
                "KVEN2_CONTEXT_BUDGET_CHARS_PER_TOKEN",
            ):
                os.environ.pop(name, None)

            with patch.object(
                routes,
                "build_context_budget_report",
            ) as reporter:
                guarded = self._guard_payload()

        reporter.assert_not_called()
        self.assertEqual(
            guarded["max_tokens"],
            4096,
        )

    def test_enabled_report_uses_final_budget_without_payload_changes(
        self,
    ):
        environment = {
            "KVEN2_CONTEXT_BUDGET_REPORT_ENABLED": "1",
            "KVEN2_CONTEXT_WINDOW_TAIL_MESSAGES": "2",
            "KVEN2_CONTEXT_BUDGET_TOKENS": "10000",
            "KVEN2_CONTEXT_BUDGET_SUMMARY_TARGET_TOKENS": "300",
            "KVEN2_CONTEXT_BUDGET_CHARS_PER_TOKEN": "4.0",
        }

        with patch.dict(
            os.environ,
            environment,
            clear=False,
        ):
            with self.assertLogs(
                routes.logger,
                level="INFO",
            ) as captured:
                enabled_payload = (
                    self._guard_payload()
                )

        with patch.dict(
            os.environ,
            {
                "KVEN2_CONTEXT_BUDGET_REPORT_ENABLED": "0",
            },
            clear=False,
        ):
            disabled_payload = self._guard_payload()

        self.assertEqual(
            enabled_payload,
            disabled_payload,
        )
        self.assertEqual(
            enabled_payload["max_tokens"],
            4096,
        )

        output = "\n".join(captured.output)

        self.assertIn(
            "[CONTEXT_BUDGET_REPORT]",
            output,
        )
        self.assertIn(
            '"route_label":"main_final"',
            output,
        )
        self.assertIn(
            '"budget_report_version":'
            '"kven2-context-budget-v1"',
            output,
        )
        self.assertIn(
            '"reserved_completion_tokens":4096',
            output,
        )
        self.assertIn(
            '"context_token_budget":10000',
            output,
        )
        self.assertIn(
            '"configured_tail_messages":2',
            output,
        )
        self.assertNotIn(
            "SECRET-SYSTEM-CONTENT",
            output,
        )
        self.assertNotIn(
            "SECRET-USER-CONTENT",
            output,
        )

    def test_budget_report_failure_does_not_break_guard(
        self,
    ):
        with patch.dict(
            os.environ,
            {
                "KVEN2_CONTEXT_BUDGET_REPORT_ENABLED": "1",
            },
            clear=False,
        ), patch.object(
            routes,
            "build_context_budget_report",
            side_effect=RuntimeError(
                "telemetry failure"
            ),
        ):
            with self.assertLogs(
                routes.logger,
                level="WARNING",
            ) as captured:
                guarded = self._guard_payload()

        self.assertEqual(
            guarded["max_tokens"],
            4096,
        )
        self.assertIn(
            "[CONTEXT_BUDGET_REPORT] failed",
            "\n".join(captured.output),
        )


if __name__ == "__main__":
    unittest.main()
