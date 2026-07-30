import unittest

from model_adapters.qwen36_llamacpp import (
    Qwen36LlamaCppAdapter,
)


class Qwen36StreamNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen36LlamaCppAdapter()

    def normalize(
        self,
        *,
        phase: str,
        thinking_enabled: bool,
        pieces: list[str],
    ) -> str:
        normalizer = (
            self.adapter.create_stream_content_normalizer(
                phase=phase,
                thinking_enabled=thinking_enabled,
            )
        )

        output = "".join(
            normalizer.feed(piece)
            for piece in pieces
        )
        output += normalizer.finish()
        return output

    def test_main_fast_strips_empty_leading_wrapper(self):
        result = self.normalize(
            phase="main",
            thinking_enabled=False,
            pieces=[
                "<thi",
                "nk>\n\n",
                "</think>\n\n",
                "B",
            ],
        )

        self.assertEqual(result, "B")

    def test_continuation_fast_still_strips_wrapper(self):
        result = self.normalize(
            phase="continuation",
            thinking_enabled=False,
            pieces=[
                "<think>\n",
                "\n</think>",
                "\n\nAnswer",
            ],
        )

        self.assertEqual(result, "Answer")

    def test_main_thinking_preserves_backend_content(self):
        source = "<think>\nreasoning\n</think>\n\nB"

        result = self.normalize(
            phase="main",
            thinking_enabled=True,
            pieces=[source],
        )

        self.assertEqual(result, source)

    def test_nonempty_think_wrapper_is_not_suppressed(self):
        source = "<think>reasoning</think>B"

        result = self.normalize(
            phase="main",
            thinking_enabled=False,
            pieces=[source],
        )

        self.assertEqual(result, source)

    def test_plain_fast_content_is_unchanged(self):
        result = self.normalize(
            phase="main",
            thinking_enabled=False,
            pieces=["Ord", "inary answer"],
        )

        self.assertEqual(result, "Ordinary answer")


if __name__ == "__main__":
    unittest.main()
