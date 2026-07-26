from __future__ import annotations

import re

from .base import (
    ModelBackendAdapter,
    StreamContentNormalizer,
)


class _EmptyLeadingThinkWrapperNormalizer(StreamContentNormalizer):
    """
    Suppress only an empty leading Qwen think wrapper.

    Accepted shape:

        optional whitespace
        <think>
        whitespace only
        </think>
        optional whitespace
        visible answer

    Anything else is returned unchanged so the gateway's generic continuation
    guard can reject non-empty, malformed, unclosed, or later think markup.
    """

    _OPENING = "<think>"
    _CLOSING = "</think>"
    _MAX_PREFIX_CHARS = 4096

    def __init__(self) -> None:
        self._buffer = ""
        self._resolved = False

    @classmethod
    def _classify(
        cls,
        text: str,
    ) -> tuple[str, int | None]:
        index = 0
        length = len(text)

        while index < length and text[index].isspace():
            index += 1

        opening_fragment = text[
            index:index + len(cls._OPENING)
        ].lower()

        if len(opening_fragment) < len(cls._OPENING):
            if cls._OPENING.startswith(opening_fragment):
                return "pending", None

            return "reject", None

        if opening_fragment != cls._OPENING:
            return "reject", None

        index += len(cls._OPENING)

        while index < length and text[index].isspace():
            index += 1

        closing_fragment = text[
            index:index + len(cls._CLOSING)
        ].lower()

        if len(closing_fragment) < len(cls._CLOSING):
            if cls._CLOSING.startswith(closing_fragment):
                return "pending", None

            return "reject", None

        if closing_fragment != cls._CLOSING:
            return "reject", None

        index += len(cls._CLOSING)
        remainder = text[index:]

        if not remainder or remainder.isspace():
            return "matched_waiting", index

        return "matched", index

    def feed(self, text: str) -> str:
        piece = str(text or "")

        if not piece:
            return ""

        if self._resolved:
            return piece

        self._buffer += piece

        if len(self._buffer) > self._MAX_PREFIX_CHARS:
            self._resolved = True
            output = self._buffer
            self._buffer = ""
            return output

        status, wrapper_end = self._classify(self._buffer)

        if status in {"pending", "matched_waiting"}:
            return ""

        self._resolved = True

        if status == "matched" and wrapper_end is not None:
            output = self._buffer[wrapper_end:].lstrip()
        else:
            output = self._buffer

        self._buffer = ""
        return output

    def finish(self) -> str:
        if self._resolved or not self._buffer:
            return ""

        status, wrapper_end = self._classify(self._buffer)
        buffered = self._buffer

        self._buffer = ""
        self._resolved = True

        if (
            status in {"matched", "matched_waiting"}
            and wrapper_end is not None
        ):
            return buffered[wrapper_end:].lstrip()

        return buffered


class Qwen36LlamaCppAdapter(ModelBackendAdapter):
    """Compatibility adapter for Qwen 3.6 served by llama.cpp."""

    adapter_id = "qwen36_llamacpp"

    def matches(
        self,
        *,
        backend_model: str,
        backend_url: str,
    ) -> bool:
        normalized_model = re.sub(
            r"[^a-z0-9]+",
            "",
            str(backend_model or "").lower(),
        )

        return "qwen36" in normalized_model

    def create_stream_content_normalizer(
        self,
        *,
        phase: str,
        thinking_enabled: bool,
    ) -> StreamContentNormalizer:
        if phase == "continuation" and not thinking_enabled:
            return _EmptyLeadingThinkWrapperNormalizer()

        return super().create_stream_content_normalizer(
            phase=phase,
            thinking_enabled=thinking_enabled,
        )
