# kven2_profile.py
import json
import logging
import os

logger = logging.getLogger(__name__)

PROFILE_PATH = "/opt/kven2/agent_profile.json"

PROFILE_INSTRUCTION_FIELDS = (
    ("identity_instruction", "IDENTITY AND CONTINUITY"),
    ("personality_instruction", "PERSONALITY"),
    (
        "personality_development_instruction",
        "PERSONALITY DEVELOPMENT",
    ),
    ("self_model_instruction", "SELF-MODEL"),
    ("consciousness_instruction", "SUBJECTIVE STATUS"),
    ("agency_instruction", "AGENCY"),
    ("capability_instruction", "CAPABILITIES"),
    ("revision_instruction", "REVISION"),
    ("memory_instruction", "MEMORY"),
    ("relationship_instruction", "RELATIONSHIP"),
    ("truth_instruction", "TRUTHFULNESS"),
    ("time_instruction", "TIME"),
    ("tool_instruction", "TOOLS"),
    ("communication_instruction", "COMMUNICATION"),
)


def _profile_text(profile: dict, key: str) -> str:
    value = profile.get(key)
    if not isinstance(value, str):
        return ""
    return value.strip()


def _supports_extended_profile(profile: dict) -> bool:
    version = profile.get("profile_version")
    return (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version >= 2
    )


def build_agent_profile_prompt(profile: dict) -> str:
    """Build a deterministic prompt block from allowlisted profile fields."""
    if not isinstance(profile, dict) or not profile:
        return ""

    agent_name = _profile_text(profile, "agent_name") or "Kven II"

    lines = [
        f"name: {agent_name}",
        "role: You are my friend.",
        "",
        f"You are {agent_name}.",
        f"Agent Role: {_profile_text(profile, 'agent_role')}",
        f"Project History: {_profile_text(profile, 'project_history')}",
        f"Owner: {_profile_text(profile, 'owner')}",
        f"Mission: {_profile_text(profile, 'mission')}",
    ]

    if not _supports_extended_profile(profile):
        return "\n".join(lines).rstrip() + "\n\n"

    relationship = _profile_text(profile, "relationship")
    if relationship:
        lines.append(f"Relationship: {relationship}")

    instruction_lines = []
    for key, heading in PROFILE_INSTRUCTION_FIELDS:
        value = _profile_text(profile, key)
        if not value:
            continue
        instruction_lines.extend((f"{heading}:", value, ""))

    if instruction_lines:
        lines.extend(("", "AGENT PROFILE INSTRUCTIONS:", ""))
        lines.extend(instruction_lines)

    return "\n".join(lines).rstrip() + "\n\n"


def load_agent_profile() -> dict:
    """Load the agent profile for synchronous use from async routes."""
    try:
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as profile_file:
                return json.load(profile_file)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[PROFILE] Invalid JSON in %s: %s",
            PROFILE_PATH,
            exc,
        )
    except Exception as exc:
        logger.error("[PROFILE] Failed to load profile: %s", exc)

    return {}
