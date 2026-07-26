from __future__ import annotations

from .base import ModelBackendAdapter


class GenericOpenAIAdapter(ModelBackendAdapter):
    """Fallback adapter for an OpenAI-compatible backend."""

    adapter_id = "generic_openai"

    def matches(
        self,
        *,
        backend_model: str,
        backend_url: str,
    ) -> bool:
        return True
