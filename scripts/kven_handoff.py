"""Validation primitives for Codex reviewer handoffs and correction findings."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


HANDOFF_SCHEMA_VERSION = "1.0"
REVIEW_FINDINGS_SCHEMA_VERSION = "1.0"
SHA_RE = re.compile(r"[0-9a-f]{40}")

HANDOFF_SECTIONS = {
    "schema_version", "task_identity", "task_understood_as", "implementation_map",
    "important_design_decisions", "rejected_alternatives", "architecture_deviations",
    "data_schema_migration_changes", "runtime_behavior_changed",
    "existing_behavior_intentionally_preserved", "known_weak_points",
    "uncertainties", "requirement_evidence_map", "tests", "exact_git_state",
    "recommended_reviewer_checks", "do_not_spend_time_rediscovering",
    "unresolved_issues", "correction_routing_metadata",
    "existing_architecture_reused", "runtime_path_changes",
    "persistent_state_changes", "security_privacy_impact", "deployment_impact",
    "things_not_tested",
}
EXPLICIT_LIST_SECTIONS = {
    "implementation_map", "important_design_decisions", "rejected_alternatives",
    "data_schema_migration_changes", "runtime_behavior_changed",
    "existing_behavior_intentionally_preserved", "known_weak_points", "uncertainties",
    "requirement_evidence_map", "tests", "recommended_reviewer_checks",
    "do_not_spend_time_rediscovering", "unresolved_issues",
    "existing_architecture_reused", "runtime_path_changes",
    "persistent_state_changes", "security_privacy_impact", "deployment_impact",
    "things_not_tested",
}

REVIEWER_CONTEXT_FIELDS = {
    "schema_version", "task_id", "original_task_path", "original_task_sha256",
    "codex_run_id", "codex_session_id_if_available", "start_time", "finish_time",
    "model", "baseline_sha", "feature_branch", "feature_sha", "worktree",
    "commit_list", "changed_paths", "migration_paths", "config_paths",
    "affected_services", "tests", "test_results", "secret_scan", "git_diff_check",
    "network_used", "developer_handoff_path", "previous_run_id",
    "previous_feature_sha", "correction_sequence", "review_status", "known_risks",
    "architecture_deviations", "findings_received", "findings_closed",
    "findings_open", "developer_run_number", "runtime_seconds", "token_usage",
    "session_resumable", "correction_context_path", "delta_handoff_path",
    "review_bundle_path",
}

DELTA_HANDOFF_FIELDS = {
    "schema_version", "task_id", "previous_run_id", "previous_feature_sha",
    "feature_sha", "correction_sequence", "finding_results",
    "changed_paths_since_previous", "tests_added", "tests_run",
    "decisions_preserved", "assumptions_invalidated", "deployment_impact_delta",
    "migration_impact_delta", "restart_scope_delta", "new_risks",
    "open_findings", "closed_findings",
}


class HandoffError(ValueError):
    """A handoff or reviewer-findings artifact violates its durable schema."""


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise HandoffError(f"{label} must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{label} must be a non-empty string")
    return value


def _keys(value: Any, required: set[str], label: str) -> dict:
    value = _object(value, label)
    missing = required - set(value)
    if missing:
        raise HandoffError(f"{label} missing required fields: {', '.join(sorted(missing))}")
    return value


def load_json(path: Path, label: str) -> dict:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise HandoffError(f"malformed {label}: {exc}") from exc


def validate_review_findings(value: Any) -> dict:
    data = _keys(value, {"schema_version", "task_id", "findings"}, "review-findings.json")
    if data["schema_version"] != REVIEW_FINDINGS_SCHEMA_VERSION:
        raise HandoffError("unsupported review-findings schema_version")
    _nonempty(data["task_id"], "review findings task_id")
    if not isinstance(data["findings"], list):
        raise HandoffError("review findings must be a list")
    ids = []
    required = {
        "finding_id", "severity", "status", "claim_or_requirement", "observed",
        "evidence", "required_correction", "must_preserve", "verification_required",
        "expected_behavior", "suspected_component", "reproduction_exists",
        "required_regression_test", "reviewer_confidence",
    }
    for index, finding in enumerate(data["findings"]):
        finding = _keys(finding, required, f"findings[{index}]")
        finding_id = _nonempty(finding["finding_id"], f"findings[{index}].finding_id")
        if not re.fullmatch(r"REV-[0-9]{3,}", finding_id):
            raise HandoffError(f"findings[{index}].finding_id is invalid")
        if finding["severity"] not in {"blocking", "major", "minor", "note"}:
            raise HandoffError(f"findings[{index}].severity is invalid")
        for field in required - {"finding_id", "severity"}:
            if not isinstance(finding[field], (str, list, dict)):
                raise HandoffError(f"findings[{index}].{field} has an invalid type")
        ids.append(finding_id)
    if len(ids) != len(set(ids)):
        raise HandoffError("review findings contain duplicate finding IDs")
    return data


def validate_reviewer_context(value: Any) -> dict:
    data = _keys(value, REVIEWER_CONTEXT_FIELDS, "reviewer-context.json")
    if data["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise HandoffError("unsupported reviewer-context schema_version")
    for field in ("task_id", "original_task_path", "original_task_sha256", "codex_run_id",
                  "start_time", "finish_time", "baseline_sha", "feature_branch", "feature_sha",
                  "worktree", "developer_handoff_path", "review_status"):
        _nonempty(data[field], f"reviewer context {field}")
    if not re.fullmatch(r"[0-9a-f]{64}", data["original_task_sha256"]):
        raise HandoffError("reviewer context original_task_sha256 is malformed")
    for field in ("baseline_sha", "feature_sha"):
        if not SHA_RE.fullmatch(data[field]):
            raise HandoffError(f"reviewer context {field} is malformed")
    if data["previous_feature_sha"] is not None and not SHA_RE.fullmatch(data["previous_feature_sha"]):
        raise HandoffError("reviewer context previous_feature_sha is malformed")
    for field in ("commit_list", "changed_paths", "migration_paths", "config_paths",
                  "affected_services", "tests", "test_results", "known_risks",
                  "architecture_deviations", "findings_received", "findings_closed", "findings_open"):
        if not isinstance(data[field], list):
            raise HandoffError(f"reviewer context {field} must be a list")
    if type(data["correction_sequence"]) is not int or data["correction_sequence"] < 0:
        raise HandoffError("reviewer context correction_sequence is invalid")
    if type(data["developer_run_number"]) is not int or data["developer_run_number"] < 1:
        raise HandoffError("reviewer context developer_run_number is invalid")
    if type(data["session_resumable"]) is not bool:
        raise HandoffError("reviewer context session_resumable must be boolean")
    return data


def validate_delta_handoff(value: Any, finding_ids: list[str]) -> dict:
    data = _keys(value, DELTA_HANDOFF_FIELDS, "delta-handoff.json")
    if data["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise HandoffError("unsupported delta-handoff schema_version")
    if not SHA_RE.fullmatch(data["previous_feature_sha"]) or not SHA_RE.fullmatch(data["feature_sha"]):
        raise HandoffError("delta handoff feature SHA is malformed")
    list_fields = DELTA_HANDOFF_FIELDS - {
        "schema_version", "task_id", "previous_run_id", "previous_feature_sha",
        "feature_sha", "correction_sequence", "deployment_impact_delta",
        "migration_impact_delta", "restart_scope_delta",
    }
    for field in list_fields:
        if not isinstance(data[field], list):
            raise HandoffError(f"delta handoff {field} must be a list")
    result_ids = [item.get("finding_id") for item in data["finding_results"] if isinstance(item, dict)]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(finding_ids):
        raise HandoffError("delta finding results do not exactly match supplied findings")
    open_ids = set(data["open_findings"])
    closed_ids = set(data["closed_findings"])
    if open_ids & closed_ids or open_ids | closed_ids != set(finding_ids):
        raise HandoffError("delta open/closed findings do not exactly cover supplied findings")
    return data


def validate_handoff(value: Any, markdown: str, *, task_id: str, baseline: str,
                     branch: str, feature_head: str, commits: list[dict[str, str]],
                     changed_files: list[str], result_package: str,
                     finding_ids: list[str] | None = None) -> dict:
    data = _object(value, "handoff-to-reviewer.json")
    missing = HANDOFF_SECTIONS - set(data)
    if missing:
        raise HandoffError("handoff missing required sections: " + ", ".join(sorted(missing)))
    if data["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise HandoffError("unsupported handoff schema_version")
    for section in EXPLICIT_LIST_SECTIONS:
        if not isinstance(data[section], list):
            raise HandoffError(f"handoff section {section} must be an explicit list")
    identity = _keys(data["task_identity"], {
        "task_id", "task_title", "baseline_sha", "feature_branch", "feature_head",
        "commits", "result_package", "codex_model", "started_at", "finished_at",
    }, "task_identity")
    expected = {
        "task_id": task_id, "baseline_sha": baseline, "feature_branch": branch,
        "feature_head": feature_head, "result_package": result_package,
    }
    for field, wanted in expected.items():
        if identity[field] != wanted:
            raise HandoffError(f"handoff {field} mismatch")
    if identity["commits"] != commits:
        raise HandoffError("handoff commit list mismatch")
    _keys(data["task_understood_as"], {"implementation", "why", "non_goals", "constraints_preserved"}, "task_understood_as")
    if not isinstance(data["task_understood_as"]["non_goals"], list) or not isinstance(data["task_understood_as"]["constraints_preserved"], list):
        raise HandoffError("task_understood_as list fields are malformed")
    implementation_paths = []
    for index, item in enumerate(data["implementation_map"]):
        item = _keys(item, {"path", "why_changed", "symbols", "responsibility", "interaction", "requirement_ids"}, f"implementation_map[{index}]")
        implementation_paths.append(_nonempty(item["path"], f"implementation_map[{index}].path"))
        if not item["why_changed"] or not item["responsibility"] or not isinstance(item["requirement_ids"], list):
            raise HandoffError(f"implementation_map[{index}] lacks substantive detail")
    omitted = sorted(set(changed_files) - set(implementation_paths))
    if omitted:
        raise HandoffError("changed files omitted from implementation map: " + ", ".join(omitted))
    structured_lists = {
        "important_design_decisions": {"decision_id", "decision", "reason", "alternatives_considered", "evidence_code_location", "consequence"},
        "rejected_alternatives": {"alternative", "why_rejected", "tradeoff"},
        "data_schema_migration_changes": {"object", "change", "compatibility", "idempotency", "rollback_implications"},
        "runtime_behavior_changed": {"before", "after", "trigger", "affected_component", "observable_symptom_telemetry"},
        "existing_behavior_intentionally_preserved": {"behavior", "why_preservation_matters", "evidence_test"},
        "known_weak_points": {"risk", "likely_symptom", "affected_location", "reviewer_check", "severity"},
        "uncertainties": {"statement", "reason", "suggested_verification", "blocking"},
        "requirement_evidence_map": {"requirement_id", "requirement", "implementation_locations", "exact_test_names", "log_result_artifact", "result", "notes"},
        "tests": {"command", "result", "duration", "requirement_ids_proven", "artifact_path"},
        "recommended_reviewer_checks": {"priority", "target", "why_it_matters", "failure_appearance"},
        "do_not_spend_time_rediscovering": {"fact", "evidence", "confidence"},
    }
    for section, fields in structured_lists.items():
        for index, item in enumerate(data[section]):
            item = _keys(item, fields, f"{section}[{index}]")
            if section == "requirement_evidence_map" and item["result"] not in {"PASS", "PARTIAL", "NOT_TESTED"}:
                raise HandoffError(f"{section}[{index}].result is invalid")
    if not data["requirement_evidence_map"]:
        raise HandoffError("requirement_evidence_map must cover every task requirement")
    _keys(data["architecture_deviations"], {"status", "reason", "consequence", "approval_required"}, "architecture_deviations")
    git_state = _keys(data["exact_git_state"], {"baseline", "commits", "feature_head", "changed_files", "diff_check", "worktree_status", "feature_worktree_clean"}, "exact_git_state")
    if (git_state["baseline"] != baseline or git_state["commits"] != commits
            or git_state["feature_head"] != feature_head
            or sorted(git_state["changed_files"]) != sorted(changed_files)
            or git_state["feature_worktree_clean"] is not True):
        raise HandoffError("handoff exact Git state mismatch")
    if task_id not in markdown or feature_head not in markdown:
        raise HandoffError("handoff Markdown task/head disagreement")
    supplied_ids = finding_ids or []
    if supplied_ids:
        results = data.get("correction_results")
        if not isinstance(results, list):
            raise HandoffError("corrective handoff requires correction_results")
        result_ids = []
        required = {"finding_id", "root_cause", "exact_correction", "files_symbols_changed", "tests_added_run", "verification_result", "remaining_risk", "status"}
        for index, item in enumerate(results):
            item = _keys(item, required, f"correction_results[{index}]")
            if item["status"] not in {"FIXED", "PARTIAL", "NOT_FIXED", "REJECTED_WITH_REASON"}:
                raise HandoffError(f"correction_results[{index}].status is invalid")
            result_ids.append(item["finding_id"])
        if set(result_ids) != set(supplied_ids) or len(result_ids) != len(set(result_ids)):
            raise HandoffError("correction_results finding IDs do not exactly match supplied findings")
    return data
