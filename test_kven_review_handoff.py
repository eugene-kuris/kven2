import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
import kven_handoff as handoff


BASE = "1" * 40
HEAD = "2" * 40
COMMITS = [{"sha": HEAD, "subject": "change"}]


def valid_handoff():
    return {
        "schema_version": "1.0",
        "task_identity": {"task_id": "TASK", "task_title": "Title", "baseline_sha": BASE,
                          "feature_branch": "feature", "feature_head": HEAD, "commits": COMMITS,
                          "result_package": "/result", "codex_model": "model",
                          "started_at": "start", "finished_at": "finish"},
        "task_understood_as": {"implementation": "Implement handoff", "why": "Review continuity",
                               "non_goals": [], "constraints_preserved": []},
        "implementation_map": [{"path": "file.py", "why_changed": "Add validation",
                                "symbols": ["validate"], "responsibility": "Validate evidence",
                                "interaction": "Called by runner", "requirement_ids": ["REQ-1"]}],
        "important_design_decisions": [], "rejected_alternatives": [],
        "architecture_deviations": {"status": "none", "reason": "Not applicable",
                                    "consequence": "None", "approval_required": False},
        "data_schema_migration_changes": [],
        "runtime_behavior_changed": [], "existing_behavior_intentionally_preserved": [],
        "known_weak_points": [], "uncertainties": [],
        "requirement_evidence_map": [{"requirement_id": "REQ-1", "requirement": "Validate",
                                      "implementation_locations": ["file.py:validate"],
                                      "exact_test_names": ["test_valid"],
                                      "log_result_artifact": "test.log", "result": "PASS", "notes": "covered"}],
        "tests": [],
        "exact_git_state": {"baseline": BASE, "commits": COMMITS, "feature_head": HEAD,
                            "changed_files": ["file.py"], "diff_check": "passed",
                            "worktree_status": "clean", "feature_worktree_clean": True},
        "recommended_reviewer_checks": [], "do_not_spend_time_rediscovering": [],
        "unresolved_issues": [], "correction_routing_metadata": {},
    }


def validate(value, finding_ids=None):
    return handoff.validate_handoff(value, f"TASK\n{HEAD}\n", task_id="TASK", baseline=BASE,
                                    branch="feature", feature_head=HEAD, commits=COMMITS,
                                    changed_files=["file.py"], result_package="/result",
                                    finding_ids=finding_ids)


class HandoffValidationTests(unittest.TestCase):
    def test_valid_and_explicit_empty_lists(self):
        self.assertEqual(validate(valid_handoff())["known_weak_points"], [])

    def test_identity_mismatches_and_omitted_file_or_section(self):
        mutations = [("task_identity", "task_id", "wrong"), ("task_identity", "baseline_sha", "0" * 40),
                     ("task_identity", "feature_branch", "wrong"), ("task_identity", "feature_head", "0" * 40),
                     ("task_identity", "commits", [])]
        for section, field, value in mutations:
            with self.subTest(field=field):
                data = valid_handoff(); data[section][field] = value
                with self.assertRaises(handoff.HandoffError): validate(data)
        data = valid_handoff(); data["implementation_map"] = []
        with self.assertRaisesRegex(handoff.HandoffError, "omitted"): validate(data)
        data = valid_handoff(); del data["known_weak_points"]
        with self.assertRaisesRegex(handoff.HandoffError, "missing required sections"): validate(data)

    def test_markdown_disagreement(self):
        with self.assertRaisesRegex(handoff.HandoffError, "Markdown"):
            handoff.validate_handoff(valid_handoff(), "TASK wrong", task_id="TASK", baseline=BASE,
                                     branch="feature", feature_head=HEAD, commits=COMMITS,
                                     changed_files=["file.py"], result_package="/result")

    def test_correction_results_exactly_cover_findings(self):
        item = {"finding_id": "REV-001", "root_cause": "cause", "exact_correction": "fix",
                "files_symbols_changed": ["file.py:validate"], "tests_added_run": ["test"],
                "verification_result": "PASS", "remaining_risk": "none", "status": "FIXED"}
        data = valid_handoff(); data["correction_results"] = [item]
        self.assertEqual(validate(data, ["REV-001"])["correction_results"][0]["status"], "FIXED")
        for results in ([], [dict(item, finding_id="REV-999")]):
            data = valid_handoff(); data["correction_results"] = results
            with self.assertRaisesRegex(handoff.HandoffError, "exactly match"): validate(data, ["REV-001"])


class FindingsAndHelperTests(unittest.TestCase):
    def findings(self):
        return {"schema_version": "1.0", "task_id": "TASK", "findings": [{
            "finding_id": "REV-001", "severity": "major", "status": "open",
            "claim_or_requirement": "REQ-1", "observed": "bad", "evidence": ["file.py:1"],
            "required_correction": "fix", "must_preserve": ["compatibility"],
            "verification_required": ["test"], "reviewer_notes": "note"}]}

    def test_valid_malformed_and_duplicate_findings(self):
        self.assertEqual(handoff.validate_review_findings(self.findings())["findings"][0]["finding_id"], "REV-001")
        duplicate = self.findings(); duplicate["findings"].append(copy.deepcopy(duplicate["findings"][0]))
        with self.assertRaisesRegex(handoff.HandoffError, "duplicate"): handoff.validate_review_findings(duplicate)
        with self.assertRaises(handoff.HandoffError): handoff.validate_review_findings({"schema_version": "1.0"})

    def test_helper_all_views_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory); data = valid_handoff()
            data["correction_results"] = [{"finding_id": "REV-001"}]
            (package / "handoff-to-reviewer.json").write_text(json.dumps(data), encoding="utf-8")
            helper = ROOT / "scripts" / "kven-review-handoff"
            for view in ("summary", "recommended-review-checks", "requirement-evidence",
                         "uncertainties-weak-points", "correction-results"):
                result = subprocess.run([sys.executable, str(helper), str(package), view, "--json"],
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)


if __name__ == "__main__":
    unittest.main()
