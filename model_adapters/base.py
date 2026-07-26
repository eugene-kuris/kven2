from __future__ import annotations

from typing import Any, Mapping


class ModelBackendAdapter:
    """
    Compatibility boundary between Kven II and a concrete model/backend pair.

    The initial implementation is intentionally passive: every transformation
    returns the input unchanged. Model-specific behavior will be moved here
    incrementally after the adapter boundary is integrated and tested.
    """

    adapter_id = "base"

    def matches(
        self,
        *,
        backend_model: str,
        backend_url: str,
    ) -> bool:
        return False

    def prepare_request(
        self,
        payload: Mapping[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        return dict(payload)

    def normalize_complete_content(
        self,
        content: str,
        *,
        phase: str,
    ) -> str:
        return content

    def forbidden_stream_markers(
        self,
        *,
        phase: str,
    ) -> tuple[str, ...]:
        return ()
