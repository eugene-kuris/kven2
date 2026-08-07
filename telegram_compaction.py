from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from context_window import estimate_tokens_from_chars


SCHEMA_VERSION = "telegram-compaction-v1"
ITEM_FIELDS = (
    "established_context",
    "speaker_statements",
    "uncertainty_and_disagreement",
    "open_loops",
    "commitments",
)


def source_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        record = {
            "id": int(row["id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "source_update_id": int(row["source_update_id"]),
        }
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_prompt(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    records = [
        {"entry_id": int(row["id"]), "role": str(row["role"]),
         "content": str(row["content"])} for row in rows
    ]
    instructions = f"""Create a JSON conversation checkpoint using schema {SCHEMA_VERSION}.
Return only one JSON object with keys: schema_version, overview,
established_context, speaker_statements, uncertainty_and_disagreement,
open_loops, commitments, important_reference_ids. Every item in each list must
be an object containing text and source_entry_ids; speaker_statements and
commitments must also contain speaker. Open loops must contain status.

Epistemic rules: unsupported detail must be omitted, never reconstructed.
Do not turn a speaker assertion into shared fact, uncertainty into certainty,
contradiction into resolution, an intention into a decision, or an unfinished
action into a completed action. Preserve competing claims as attributed or
unresolved. Completion/cancellation is allowed only when a later entry says so.
Every semantic item must cite one or more supplied entry IDs. The overview must
be neutral and may not add details absent from the entries. This is derived,
potentially incomplete conversation context, not authoritative memory."""
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(
            {"transcript_entries": records}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"))},
    ]


def parse_and_validate_payload(raw: str, valid_ids: set[int]) -> dict[str, Any]:
    text = raw.strip() if isinstance(raw, str) else raw
    if isinstance(text, str) and text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1 and text.endswith("```"):
            text = text[first_newline + 1:-3].strip()

    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("compaction output is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("compaction schema version is missing or unsupported")
    if not isinstance(payload.get("overview"), str):
        raise ValueError("compaction overview must be text")
    for field in ITEM_FIELDS:
        items = payload.get(field)
        if not isinstance(items, list):
            raise ValueError(f"compaction field {field} must be a list")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"compaction field {field} contains an invalid item")
            ids = item.get("source_entry_ids")
            if (not isinstance(ids, list) or not ids or
                    any(not isinstance(value, int) or isinstance(value, bool)
                        for value in ids)):
                raise ValueError(f"compaction field {field} item lacks source IDs")
            if not set(ids) <= valid_ids:
                raise ValueError(f"compaction field {field} cites an invalid source ID")
            if field in {"speaker_statements", "commitments"} and not isinstance(item.get("speaker"), str):
                raise ValueError(f"compaction field {field} item lacks a speaker")
            if field == "open_loops" and not isinstance(item.get("status"), str):
                raise ValueError("compaction open loop lacks a status")
    references = payload.get("important_reference_ids")
    if (not isinstance(references, list) or
            any(not isinstance(value, int) or isinstance(value, bool)
                for value in references) or not set(references) <= valid_ids):
        raise ValueError("compaction important references are invalid")
    return payload


def render_payload(payload: Mapping[str, Any], token_budget: int) -> tuple[str, int]:
    header = ("[Derived Telegram conversation checkpoint; potentially incomplete. "
              "Exact recent messages, explicit reply references, verified memory, "
              "and current observations take precedence.]\n")
    sections: list[str] = []
    # Open state is projected first so it survives summary-budget pressure.
    for title, field in (
        ("Open loops", "open_loops"), ("Commitments", "commitments"),
        ("Uncertainty and disagreement", "uncertainty_and_disagreement"),
        ("Established context", "established_context"),
        ("Speaker-attributed statements", "speaker_statements"),
    ):
        items = payload.get(field, [])
        if items:
            lines = [f"- {item['text']} [sources: {','.join(map(str, item['source_entry_ids']))}]"
                     for item in items]
            sections.append(f"{title}:\n" + "\n".join(lines))
    overview = str(payload.get("overview", "")).strip()
    if overview:
        sections.append("Overview:\n" + overview)
    selected: list[str] = []
    for section in sections:
        candidate = header + "\n\n".join(selected + [section])
        if estimate_tokens_from_chars(len(candidate)) <= token_budget:
            selected.append(section)
    rendered = header + "\n\n".join(selected)
    return rendered, estimate_tokens_from_chars(len(rendered))
