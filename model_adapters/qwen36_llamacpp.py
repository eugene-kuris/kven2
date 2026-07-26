from __future__ import annotations

import re

from .base import ModelBackendAdapter


class Qwen36LlamaCppAdapter(ModelBackendAdapter):
    """
    Adapter for Qwen 3.6 served by llama.cpp.

    This first revision only identifies the model/backend combination.
    It intentionally performs no request or response transformations yet.
    """

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
