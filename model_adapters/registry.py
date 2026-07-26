from __future__ import annotations

from .base import ModelBackendAdapter
from .generic_openai import GenericOpenAIAdapter
from .qwen36_llamacpp import Qwen36LlamaCppAdapter


_GENERIC_ADAPTER = GenericOpenAIAdapter()

_MODEL_ADAPTERS: tuple[ModelBackendAdapter, ...] = (
    Qwen36LlamaCppAdapter(),
)

_ADAPTERS_BY_ID: dict[str, ModelBackendAdapter] = {
    adapter.adapter_id: adapter
    for adapter in (
        *_MODEL_ADAPTERS,
        _GENERIC_ADAPTER,
    )
}


def available_adapter_ids() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS_BY_ID))


def resolve_model_adapter(
    *,
    backend_model: str,
    backend_url: str,
    adapter_id: str | None = None,
) -> ModelBackendAdapter:
    """
    Select an adapter explicitly or detect it from the active backend model.

    Explicit adapter_id support is intended for the future LMM integration.
    Automatic detection remains available as a safe fallback.
    """
    if adapter_id:
        selected = _ADAPTERS_BY_ID.get(str(adapter_id).strip())

        if selected is None:
            raise ValueError(
                f"Unknown model adapter: {adapter_id!r}"
            )

        return selected

    for adapter in _MODEL_ADAPTERS:
        if adapter.matches(
            backend_model=backend_model,
            backend_url=backend_url,
        ):
            return adapter

    return _GENERIC_ADAPTER
