import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).parent / "scripts" / "kven-integrate-task"
loader = importlib.machinery.SourceFileLoader("kven_integrate_task", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
integration = importlib.util.module_from_spec(spec)
loader.exec_module(integration)


# Machine-readable traceability for section 11 of WORK-KVEN-OPS-002A.
SCENARIO_COVERAGE = {
    1: ["test_valid_manifest_parsing_and_inspection"],
    2: ["test_missing_malformed_and_strict_type_rejection"],
    3: ["test_default_repository_boundary_rejects_manifest_redirect"],
    4: ["test_baseline_head_mismatch_rejected"],
    5: ["test_dirty_main_rejected"],
    6: ["test_origin_main_mismatch_rejected"],
    7: ["test_missing_feature_branch_rejected"],
    8: ["test_feature_head_mismatch_rejected"],
    9: ["test_failed_codex_status_rejected"],
    10: ["test_empty_and_failed_test_evidence_rejected"],
    11: ["test_hidden_actual_changed_path_and_allowed_scope_rejected"],
    12: ["test_missing_contract_rejected"],
    13: ["test_dry_run_is_read_only"],
    14: ["test_end_to_end_stage_record_no_push_rollback"],
    15: ["test_end_to_end_stage_record_no_push_rollback"],
    16: ["test_wal_online_backup_integrity_and_connection_close"],
    17: ["test_wal_online_backup_integrity_and_connection_close", "test_stale_wal_cannot_override_restore"],
    18: ["test_wal_online_backup_integrity_and_connection_close"],
    19: ["test_config_restore_preserves_content_and_metadata"],
    20: ["test_migration_disposable_idempotent_and_live_unchanged"],
    21: ["test_migration_disposable_idempotent_and_live_unchanged"],
    22: ["test_migration_and_application_smoke_fail_before_live_mutation"],
    23: ["test_merge_conflict_stops_and_rolls_back"],
    24: ["test_only_allowlisted_services_may_restart"],
    25: ["test_service_timeout_failed_activation_and_restart_loop"],
    26: ["test_readiness_and_fatal_log_failures_roll_back"],
    27: ["test_finalize_refuses_unexpected_git_change"],
    28: ["test_finalize_requires_acceptance_then_pushes"],
    29: ["test_repeated_finalize_is_safe"],
    30: ["test_end_to_end_stage_record_no_push_rollback"],
    31: ["test_rollback_restores_fixture_database_and_configuration"],
    32: ["test_inactive_service_remains_inactive_after_rollback"],
    33: ["test_sensitive_backups_excluded_from_result_artifacts"],
    34: ["test_redaction_in_argv_and_output"],
    35: ["test_large_output_bounded_and_full_artifact_retained"],
    36: ["test_summary_is_generated_with_exact_failure_reason"],
    37: ["test_coverage_map_has_all_37_scenarios"],
}

R4_REGRESSION_COVERAGE = {
    "push_hook_repository_mutation": [
        "test_post_push_tracked_hook_mutation_requires_recovery",
        "test_post_push_untracked_hook_mutation_requires_recovery",
        "test_post_push_index_hook_mutation_requires_recovery",
        "test_post_push_branch_hook_mutation_requires_recovery",
    ],
    "push_hook_service_mutation": [
        "test_post_push_service_state_hook_mutation_requires_recovery",
        "test_post_push_restart_hook_mutation_requires_recovery",
    ],
    "post_push_executable_checks": [
        "test_post_push_readiness_marker_failure_requires_recovery",
        "test_post_push_fatal_log_marker_failure_requires_recovery",
        "test_post_push_readiness_repository_mutation_requires_recovery",
        "test_post_push_readiness_service_mutation_requires_recovery",
    ],
    "push_outcomes_and_recovery": [
        "test_finalize_requires_acceptance_then_pushes",
        "test_clean_interrupted_push_uses_shared_reconciliation",
        "test_failed_push_with_unchanged_remote_remains_retryable",
        "test_interrupted_finalize_unexpected_remote_is_internal_error",
        "test_recovery_required_refuses_repeat_without_remote_rewrite",
        "test_already_finalized_drift_is_non_mutating_error",
    ],
}

R5_REGRESSION_COVERAGE = {
    "exact_feature_context": [
        "test_real_shape_feature_only_validation_reaches_awaiting_acceptance",
        "test_feature_branch_move_uses_recorded_exact_head",
        "test_dirty_operator_worktree_does_not_affect_exact_validation",
    ],
    "validation_mutation_refusal": [
        "test_validation_tracked_mutation_refused_before_live_mutation",
        "test_validation_staged_mutation_refused_before_live_mutation",
        "test_validation_untracked_and_symlink_mutations_refused_before_live_mutation",
        "test_validation_head_movement_refused_before_live_mutation",
        "test_validation_branch_attachment_refused_before_live_mutation",
    ],
    "checkout_failure_cleanup": [
        "test_validation_worktree_creation_failure_has_no_production_mutation",
        "test_validation_prune_failure_has_no_production_mutation",
        "test_failed_pre_merge_evidence_is_preserved_and_checkout_removed",
    ],
    "execution_boundary": [
        "test_real_shape_feature_only_validation_reaches_awaiting_acceptance",
        "test_post_merge_test_cwd_is_staged_production_main",
    ],
}

OPS_002B_REGRESSION_COVERAGE = {
    "feature_only_migration_and_cwd": ["test_feature_only_migration_runs_from_exact_checkout"],
    "cwd_safety": ["test_migration_checkout_cwd_mapping_and_refusal"],
    "migration_fingerprint": ["test_migration_checkout_mutation_is_refused", "test_smoke_checkout_mutation_is_refused"],
    "migration_cleanup": ["test_migration_failure_and_cleanup_failure_are_pre_mutation"],
}


def cmd(*args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def record(name="synthetic", passed=True, command=None, artifact="test-artifacts/synthetic.log", content=b""):
    return {
        "name": name, "command": command or [sys.executable, "-c", "pass"],
        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_seconds": 1.0, "exit_code": 0 if passed else 1, "passed": passed,
        "bounded_output": content.decode(errors="replace"), "output_artifact": artifact,
        "artifact_size": len(content), "artifact_sha256": hashlib.sha256(content).hexdigest(),
    }


class RepositoryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.repo = self.root / "repo"
        self.feature_worktree = self.root / "feature-worktree"
        cmd("git", "init", "--bare", str(self.remote))
        cmd("git", "init", "-b", "main", str(self.repo))
        cmd("git", "-C", str(self.repo), "config", "user.email", "test@example.invalid")
        cmd("git", "-C", str(self.repo), "config", "user.name", "Test")
        (self.repo / "base.txt").write_text("base\n", encoding="utf-8")
        (self.repo / "test_kven_codex_task.py").write_text(
            "import unittest\n\nclass BaselineTest(unittest.TestCase):\n"
            "    def test_baseline(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        cmd("git", "-C", str(self.repo), "add", "base.txt", "test_kven_codex_task.py")
        cmd("git", "-C", str(self.repo), "commit", "-m", "base")
        cmd("git", "-C", str(self.repo), "remote", "add", "origin", str(self.remote))
        cmd("git", "-C", str(self.repo), "push", "-u", "origin", "main")
        self.base = cmd("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip()
        cmd("git", "-C", str(self.repo), "worktree", "add", "-b", "feature", str(self.feature_worktree), self.base)
        (self.feature_worktree / "feature.txt").write_text("feature\n", encoding="utf-8")
        (self.feature_worktree / "test_kven_integrate_task.py").write_text(
            "import unittest\n\nclass FeatureIntegrationTest(unittest.TestCase):\n"
            "    def test_feature_integration(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        (self.feature_worktree / "test_feature_only.py").write_text(
            "import unittest\n\nclass FeatureOnlyTest(unittest.TestCase):\n"
            "    def test_feature_only(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        check = {"name": "synthetic", "command": [sys.executable, "-c", "pass"], "timeout": 30}
        self.contract = {
            "schema_version": "2.0", "expected_baseline": self.base, "feature_branch": "feature",
            "allowed_paths": ["deployment-contract.json", "feature.txt", "test_feature_only.py",
                              "test_kven_integrate_task.py"], "services": [],
            "backups": {"sqlite": [], "configuration": []}, "migration_checks": [],
            "result_validation_tests": [copy.deepcopy(check)], "pre_merge_tests": [copy.deepcopy(check)],
            "post_merge_tests": [copy.deepcopy(check)], "readiness_checks": [], "fatal_log_checks": [],
            "acceptance_checklist": ["Observe synthetic behavior"],
            "rollback": {"restore_git": True, "restore_backups": True, "verify_readiness": True},
        }
        (self.feature_worktree / "deployment-contract.json").write_text(json.dumps(self.contract), encoding="utf-8")
        cmd("git", "-C", str(self.feature_worktree), "add", "feature.txt", "deployment-contract.json",
            "test_feature_only.py", "test_kven_integrate_task.py")
        cmd("git", "-C", str(self.feature_worktree), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "feature")
        self.feature = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.package = self.root / "package"
        (self.package / "test-artifacts").mkdir(parents=True)
        (self.package / "test-artifacts" / "synthetic.log").write_bytes(b"")
        contract_hash = integration.canonical_contract_sha256(integration.validate_contract(self.contract, {
            "baseline_head": self.base, "feature_branch": "feature",
        }))
        self.manifest = {
            "schema_version": "2.0", "task_id": "TEST", "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00", "codex_runtime_seconds": 1.0,
            "requested_model": None, "actual_runtime_model": "fixture-model",
            "actual_runtime_provider": "fixture-provider", "actual_reasoning_effort": "high",
            "runtime_session_id": "fixture-session", "token_usage": None,
            "network_use": {"allowed": False, "used": False, "observation": "network prohibited"},
            "evidence_provenance": {"method": "bootstrap_postprocessing", "runner_contained_manifest_features": False, "description": "fixture"},
            "exit_code": 0, "final_codex_status": "PASS", "repository_path": str(self.repo),
            "baseline_branch": "main", "baseline_head": self.base, "origin_main_head": self.base,
            "feature_branch": "feature", "feature_head": self.feature,
            "worktree_path": str(self.feature_worktree),
            "commits_created": [{"sha": self.feature, "subject": "feature"}],
            "changed_files": ["deployment-contract.json", "feature.txt", "test_feature_only.py",
                              "test_kven_integrate_task.py"], "tests": [record()],
            "git_diff_check": {"passed": True}, "secret_scan": {"passed": True},
            "evidence_secret_scan": {"passed": True},
            "deployment_contract": self.contract, "deployment_contract_path": "deployment-contract.json",
            "deployment_contract_sha256": contract_hash,
        }
        self.path = self.package / "result-manifest.json"
        self.results = self.root / "results"
        self.backups = self.root / "backups"
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def bind_contract(self):
        """Amend the synthetic feature so executable contract changes are authoritative."""
        (self.feature_worktree / "deployment-contract.json").write_text(json.dumps(self.contract), encoding="utf-8")
        cmd("git", "-C", str(self.feature_worktree), "add", "deployment-contract.json")
        result = cmd(
            "git", "-C", str(self.feature_worktree), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "--amend", "--no-edit",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.feature = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.manifest["feature_head"] = self.feature
        self.manifest["commits_created"] = [{"sha": self.feature, "subject": "feature"}]
        self.manifest["changed_files"] = integration.git_changed_files(self.repo, self.base, self.feature)
        normalized = integration.validate_contract(self.contract, self.manifest)
        self.manifest["deployment_contract_sha256"] = integration.canonical_contract_sha256(normalized)
        self.write()

    def amend_feature_content(self, content: bytes):
        (self.feature_worktree / "feature.txt").write_bytes(content)
        cmd("git", "-C", str(self.feature_worktree), "add", "feature.txt")
        result = cmd(
            "git", "-C", str(self.feature_worktree), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "--amend", "--no-edit",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.feature = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.manifest["feature_head"] = self.feature
        self.manifest["commits_created"] = [{"sha": self.feature, "subject": "feature"}]
        self.manifest["changed_files"] = integration.git_changed_files(self.repo, self.base, self.feature)
        self.write()

    def inspect(self):
        return integration.inspect_manifest(self.manifest, manifest_path=self.path, expected_repository=self.repo)

    def stage_cli(self, *extra):
        self.bind_contract()
        return cmd(
            sys.executable, str(SCRIPT), "stage", str(self.path), "--repository", str(self.repo),
            "--result-root", str(self.results), "--backup-root", str(self.backups), *extra,
        )

    def run_dir(self):
        return next((self.results / "TEST").iterdir())

    def install_pre_push_hook(self, body):
        hook = integration.git_common_dir(self.repo) / "hooks" / "pre-push"
        hook.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
        hook.chmod(0o755)
        return hook

    def finalize_cli(self, run_dir=None):
        return cmd(
            sys.executable, str(SCRIPT), "finalize", str(run_dir or self.run_dir()),
            "--accept", "PASS", "--repository", str(self.repo),
        )

    def test_valid_manifest_parsing_and_inspection(self):
        _, loaded = integration.load_manifest(str(self.package))
        self.assertEqual(loaded["task_id"], "TEST")
        report = self.inspect()
        self.assertTrue(report["eligible"])
        self.assertEqual(report["actual_changed_files"], [
            "deployment-contract.json", "feature.txt", "test_feature_only.py", "test_kven_integrate_task.py",
        ])
        self.assertEqual(report["deployment_contract_sha256"], self.manifest["deployment_contract_sha256"])

    def test_package_only_contract_mutations_and_hash_mismatch_are_rejected(self):
        mutations = []
        injected = copy.deepcopy(self.contract)
        injected["deployment_steps"] = [{"name": "injected", "command": [sys.executable, "-c", "pass"], "timeout": 30}]
        mutations.append(injected)
        changed_test = copy.deepcopy(self.contract)
        changed_test["post_merge_tests"][0]["command"] = [sys.executable, "-c", "print('changed')"]
        mutations.append(changed_test)
        changed_scope = copy.deepcopy(self.contract)
        changed_scope["services"] = ["kven2-main.service"]
        changed_scope["readiness_checks"] = [{"command": [sys.executable, "-c", "pass"]}]
        changed_scope["fatal_log_checks"] = [{"command": [sys.executable, "-c", "pass"]}]
        changed_scope["backups"]["configuration"] = [{"path": "/tmp/fixture", "services": []}]
        changed_scope["migration_checks"] = [{
            "database": "/tmp/fixture.sqlite", "command": [sys.executable, "-c", "pass"],
            "application_smoke_command": [sys.executable, "-c", "pass"],
        }]
        mutations.append(changed_scope)
        for candidate in mutations:
            with self.subTest(keys=sorted(candidate)):
                self.manifest["deployment_contract"] = candidate
                self.manifest["deployment_contract_sha256"] = integration.canonical_contract_sha256(
                    integration.validate_contract(candidate, self.manifest)
                )
                with self.assertRaisesRegex(integration.IntegrationError, "differs from exact committed contract"):
                    self.inspect()
        self.manifest["deployment_contract"] = self.contract
        self.manifest["deployment_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(integration.IntegrationError, "SHA-256 binding mismatch"):
            self.inspect()

    def test_committed_contract_missing_and_malformed_are_rejected(self):
        (self.feature_worktree / "deployment-contract.json").unlink()
        cmd("git", "-C", str(self.feature_worktree), "add", "deployment-contract.json")
        cmd("git", "-C", str(self.feature_worktree), "commit", "--amend", "--no-edit")
        self.feature = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.manifest["feature_head"] = self.feature
        self.manifest["commits_created"] = [{"sha": self.feature, "subject": "feature"}]
        self.manifest["changed_files"] = integration.git_changed_files(self.repo, self.base, self.feature)
        with self.assertRaisesRegex(integration.IntegrationError, "committed deployment contract is missing"):
            self.inspect()

        (self.feature_worktree / "deployment-contract.json").write_text("{", encoding="utf-8")
        cmd("git", "-C", str(self.feature_worktree), "add", "deployment-contract.json")
        cmd("git", "-C", str(self.feature_worktree), "commit", "--amend", "--no-edit")
        self.feature = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.manifest["feature_head"] = self.feature
        self.manifest["commits_created"] = [{"sha": self.feature, "subject": "feature"}]
        self.manifest["changed_files"] = integration.git_changed_files(self.repo, self.base, self.feature)
        with self.assertRaisesRegex(integration.IntegrationError, "committed deployment contract is malformed"):
            self.inspect()

    def test_exact_changed_content_scanner_rejects_all_credential_forms_without_values(self):
        value = "runtime-" + "private-material-123"
        forms = [
            "tool --" + "token " + value,
            "api_" + "key=" + value,
            "Author" + "ization: " + value,
            "Bear" + "er " + value,
            "-----BEGIN " + "PRIVATE KEY-----",
        ]
        for form in forms:
            with self.subTest(form=form.split()[0]):
                self.amend_feature_content((form + "\n").encode())
                with self.assertRaises(integration.IntegrationError) as raised:
                    self.inspect()
                self.assertIn("offline secret scan failed", str(raised.exception))
                self.assertNotIn(value, str(raised.exception))

    def test_changed_binary_is_rejected_as_unscannable(self):
        value = ("runtime-" + "private-material-456").encode()
        self.amend_feature_content(b"\x00" + ("--" + "token ").encode() + value)
        with self.assertRaisesRegex(integration.IntegrationError, "changed_binary_unscannable"):
            self.inspect()

    def test_missing_malformed_and_strict_type_rejection(self):
        with self.assertRaises(integration.IntegrationError):
            integration.load_manifest(str(self.root / "missing"))
        self.path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(integration.IntegrationError, "malformed"):
            integration.load_manifest(str(self.path))
        self.write()
        self.manifest["deployment_contract"]["pre_merge_tests"] = [{"command": "echo unsafe"}]
        with self.assertRaisesRegex(integration.IntegrationError, "argv array"):
            integration.validate_manifest_types(self.manifest)

    def test_default_repository_boundary_rejects_manifest_redirect(self):
        with self.assertRaisesRegex(integration.IntegrationError, "unexpected repository"):
            integration.inspect_manifest(self.manifest, manifest_path=self.path)

    def test_baseline_head_mismatch_rejected(self):
        self.manifest["baseline_head"] = "0" * 40
        self.contract["expected_baseline"] = "0" * 40
        with self.assertRaisesRegex(integration.IntegrationError, "baseline HEAD mismatch"):
            self.inspect()

    def test_dirty_main_rejected(self):
        (self.repo / "dirty.txt").write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(integration.IntegrationError, "dirty"):
            self.inspect()

    def test_origin_main_mismatch_rejected(self):
        cmd("git", "-C", str(self.repo), "update-ref", "refs/remotes/origin/main", self.feature)
        with self.assertRaisesRegex(integration.IntegrationError, "origin-main"):
            self.inspect()

    def test_missing_feature_branch_rejected(self):
        cmd("git", "-C", str(self.repo), "worktree", "remove", str(self.feature_worktree))
        cmd("git", "-C", str(self.repo), "branch", "-D", "feature")
        with self.assertRaisesRegex(integration.IntegrationError, "missing"):
            self.inspect()

    def test_feature_head_mismatch_rejected(self):
        self.manifest["feature_head"] = "0" * 40
        with self.assertRaisesRegex(integration.IntegrationError, "exact feature commit object is missing"):
            self.inspect()

    def test_failed_codex_status_rejected(self):
        self.manifest["final_codex_status"] = "FAIL"
        with self.assertRaisesRegex(integration.IntegrationError, "not successful"):
            self.inspect()

    def test_empty_and_failed_test_evidence_rejected(self):
        self.manifest["tests"] = []
        with self.assertRaisesRegex(integration.IntegrationError, "missing or empty"):
            self.inspect()
        self.manifest["tests"] = [record(passed=False)]
        with self.assertRaisesRegex(integration.IntegrationError, "required tests failed"):
            self.inspect()

    def test_test_evidence_must_match_contract_exactly_and_in_order(self):
        unrelated = record(name="unrelated")
        self.manifest["tests"] = [unrelated]
        with self.assertRaisesRegex(integration.IntegrationError, "does not match"):
            self.inspect()

        self.manifest["tests"] = []
        with self.assertRaisesRegex(integration.IntegrationError, "missing or empty"):
            self.inspect()

        second_command = [sys.executable, "-c", "print('second')"]
        second_content = b"second\n"
        (self.package / "test-artifacts" / "second.log").write_bytes(second_content)
        self.contract["result_validation_tests"].append({"name": "second", "command": second_command, "timeout": 30})
        self.bind_contract()
        second = record("second", command=second_command, artifact="test-artifacts/second.log", content=second_content)
        first = record()
        self.manifest["tests"] = [first]
        with self.assertRaisesRegex(integration.IntegrationError, "count differs"):
            self.inspect()
        self.manifest["tests"] = [first, second, record()]
        with self.assertRaisesRegex(integration.IntegrationError, "count differs"):
            self.inspect()
        self.manifest["tests"] = [second, first]
        with self.assertRaisesRegex(integration.IntegrationError, "does not match"):
            self.inspect()
        mismatched = copy.deepcopy(first)
        mismatched["command"] = [sys.executable, "-c", "print('wrong')"]
        self.manifest["tests"] = [mismatched, second]
        with self.assertRaisesRegex(integration.IntegrationError, "does not match"):
            self.inspect()

    def test_test_artifact_path_integrity_and_file_type_are_enforced(self):
        artifact = self.package / "test-artifacts" / "synthetic.log"
        artifact.unlink()
        with self.assertRaisesRegex(integration.IntegrationError, "outside or missing"):
            self.inspect()

        artifact.write_bytes(b"tampered")
        with self.assertRaisesRegex(integration.IntegrationError, "size/checksum"):
            self.inspect()

        outside = self.root / "outside.log"
        outside.write_bytes(b"")
        artifact.unlink()
        artifact.symlink_to(outside)
        with self.assertRaisesRegex(integration.IntegrationError, "outside or missing|regular non-symlink"):
            self.inspect()

        artifact.unlink()
        artifact.write_bytes(b"")
        self.manifest["tests"][0]["output_artifact"] = "../outside.log"
        with self.assertRaisesRegex(integration.IntegrationError, "outside"):
            self.inspect()
        self.manifest["tests"][0]["output_artifact"] = str(outside)
        with self.assertRaisesRegex(integration.IntegrationError, "artifact metadata"):
            self.inspect()

    def test_hidden_actual_changed_path_and_allowed_scope_rejected(self):
        self.manifest["changed_files"] = []
        with self.assertRaisesRegex(integration.IntegrationError, "changed_files mismatch"):
            self.inspect()
        self.manifest["changed_files"] = [
            "deployment-contract.json", "feature.txt", "test_feature_only.py", "test_kven_integrate_task.py",
        ]
        self.contract["allowed_paths"] = ["different.txt"]
        self.bind_contract()
        with self.assertRaisesRegex(integration.IntegrationError, "outside contract"):
            self.inspect()

    def test_independent_diff_and_secret_checks_use_exact_commits(self):
        detected_value = "tok" + "en=" + "runtime-private-material"
        (self.feature_worktree / "feature.txt").write_text(detected_value + "\n", encoding="utf-8")
        cmd("git", "-C", str(self.feature_worktree), "add", "feature.txt")
        cmd("git", "-C", str(self.feature_worktree), "commit", "-m", "secret")
        moved = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.manifest["feature_head"] = moved
        self.manifest["commits_created"].append({"sha": moved, "subject": "secret"})
        with self.assertRaisesRegex(integration.IntegrationError, "independent offline secret scan failed"):
            self.inspect()

        (self.feature_worktree / "feature.txt").write_text("trailing whitespace   \n", encoding="utf-8")
        cmd("git", "-C", str(self.feature_worktree), "add", "feature.txt")
        cmd("git", "-C", str(self.feature_worktree), "commit", "-m", "whitespace")
        moved = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        self.manifest["feature_head"] = moved
        self.manifest["commits_created"].append({"sha": moved, "subject": "whitespace"})
        with self.assertRaisesRegex(integration.IntegrationError, "independent git diff --check failed"):
            self.inspect()

    def test_missing_contract_rejected(self):
        self.manifest["deployment_contract"] = None
        with self.assertRaisesRegex(integration.IntegrationError, "deployment contract must be an object"):
            self.inspect()

    def test_literal_credentials_are_rejected_without_reflection(self):
        literal = "literal-" + "sensitive-value-123"
        forms = [
            ["tool", "--" + "token", literal],
            ["tool", "api_" + "key=" + literal],
            ["tool", "Author" + "ization", "Bear" + "er " + literal],
            ["tool", "-----BEGIN " + "PRIVATE KEY-----", literal],
        ]
        for command in forms:
            with self.subTest(form=command[1]):
                contract = copy.deepcopy(self.contract)
                contract["pre_merge_tests"] = [{"name": "unsafe", "command": command, "timeout": 30}]
                with self.assertRaises(integration.IntegrationError) as raised:
                    integration.validate_contract(contract, self.manifest)
                self.assertNotIn(literal, str(raised.exception))
        placeholder = copy.deepcopy(self.contract)
        placeholder["pre_merge_tests"] = [{
            "name": "fixture", "command": ["tool", "--" + "token", "synthetic-non-secret-placeholder"],
            "timeout": 30,
        }]
        self.assertTrue(integration.validate_contract(placeholder, self.manifest))

    def test_dry_run_is_read_only(self):
        before = integration.repository_state(self.repo)
        objects_before = cmd("git", "-C", str(self.repo), "count-objects", "-v").stdout
        result = cmd(sys.executable, str(SCRIPT), "dry-run", str(self.path), "--repository", str(self.repo))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"dry_run": true', result.stdout)
        self.assertIn(self.feature, result.stdout)
        self.assertEqual(before, integration.repository_state(self.repo))
        self.assertEqual(objects_before, cmd("git", "-C", str(self.repo), "count-objects", "-v").stdout)

    def test_end_to_end_stage_record_no_push_rollback(self):
        result = self.stage_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run_dir = self.run_dir()
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertTrue((run_dir / "git-before.json").is_file())
        self.assertTrue((run_dir / "post-merge-tests.json").is_file())
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)
        rollback = cmd(sys.executable, str(SCRIPT), "rollback", str(run_dir), "--repository", str(self.repo))
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        self.assertEqual(cmd("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip(), self.base)
        self.assertTrue((run_dir / "rollback-result.json").is_file())

    def test_feature_branch_move_uses_recorded_exact_head(self):
        args = self._stage_args()
        recorded = self.manifest["feature_head"]
        real_inspect = integration.inspect_manifest
        calls = 0
        moved = None

        def moving_inspect(*values, **kwargs):
            nonlocal calls, moved
            calls += 1
            if calls == 2:
                cmd("git", "-C", str(self.feature_worktree), "commit", "--allow-empty", "-m", "race")
                moved = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
            return real_inspect(*values, **kwargs)

        with mock.patch.object(integration, "inspect_manifest", side_effect=moving_inspect):
            run_dir, state = integration.stage(self.path, self.manifest, args)
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertEqual(state["feature_head"], recorded)
        self.assertNotEqual(moved, recorded)
        self.assertEqual(cmd("git", "-C", str(self.repo), "merge-base", "--is-ancestor", recorded,
                             state["staged_head"]).returncode, 0)
        self.assertNotEqual(cmd("git", "-C", str(self.repo), "merge-base", "--is-ancestor", moved,
                                state["staged_head"]).returncode, 0)
        self.assertTrue(state["validation_checkout"]["cleanup"]["passed"])
        self.assertFalse(Path(state["validation_checkout"]["checkout_path"]).exists())

    def _stage_with_pre_merge(self, command, name="r5-validation"):
        self.contract["pre_merge_tests"] = [{"name": name, "command": command, "timeout": 30}]
        args = self._stage_args()
        before = integration.repository_state(self.repo)
        run_dir, state = integration.stage(self.path, self.manifest, args)
        return run_dir, state, before

    def _assert_pre_mutation_validation_refusal(self, state, before):
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertFalse(state["live_mutation_started"])
        self.assertEqual(integration.repository_state(self.repo), before)
        self.assertFalse(self.backups.exists())
        evidence = state["validation_checkout"]
        self.assertFalse(Path(evidence["checkout_path"]).exists())
        checkout = str(Path(evidence["checkout_path"]).resolve())
        self.assertNotIn(f"worktree {checkout}", cmd(
            "git", "-C", str(self.repo), "worktree", "list", "--porcelain",
        ).stdout)
        return evidence

    def test_real_shape_feature_only_validation_reaches_awaiting_acceptance(self):
        exact = [
            sys.executable, "-m", "unittest", "test_kven_codex_task", "test_kven_integrate_task",
        ]
        feature_only = [sys.executable, "-m", "unittest", "test_feature_only"]
        no_bytecode = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        baseline_exact = cmd(*exact, cwd=self.repo, env=no_bytecode)
        baseline_feature_only = cmd(*feature_only, cwd=self.repo, env=no_bytecode)
        self.assertNotEqual(baseline_exact.returncode, 0)
        self.assertIn("test_kven_integrate_task", baseline_exact.stderr)
        self.assertNotEqual(baseline_feature_only.returncode, 0)
        self.assertIn("test_feature_only", baseline_feature_only.stderr)
        self.contract["pre_merge_tests"] = [
            {"name": "real-shape", "command": exact, "timeout": 30},
            {"name": "feature-only", "command": feature_only, "timeout": 30},
        ]
        result = self.stage_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        evidence = state["validation_checkout"]
        self.assertTrue(evidence["cleanup"]["passed"])
        self.assertFalse(Path(evidence["checkout_path"]).exists())
        for item in evidence["commands"]:
            self.assertEqual(item["cwd"], evidence["checkout_path"])
            self.assertEqual(item["exact_feature_head"], state["feature_head"])
            self.assertEqual(item["before"], item["after"])
            self.assertEqual(item["before"]["branch"], "")
            self.assertEqual(item["before"]["head"], state["feature_head"])
            self.assertTrue(item["command"]["passed"])
        parents = cmd("git", "-C", str(self.repo), "rev-list", "--parents", "-n", "1",
                      state["staged_head"]).stdout.split()
        self.assertIn(state["feature_head"], parents[1:])

    def test_dirty_operator_worktree_does_not_affect_exact_validation(self):
        args = self._stage_args()
        recorded = self.manifest["feature_head"]
        (self.feature_worktree / "feature.txt").write_text("operator dirty data\n", encoding="utf-8")
        (self.feature_worktree / "operator-untracked.txt").write_text("preserve\n", encoding="utf-8")
        _, state = integration.stage(self.path, self.manifest, args)
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertEqual(state["feature_head"], recorded)
        self.assertEqual((self.feature_worktree / "feature.txt").read_text(), "operator dirty data\n")
        self.assertTrue((self.feature_worktree / "operator-untracked.txt").is_file())
        self.assertTrue(state["validation_checkout"]["cleanup"]["passed"])

    def configure_feature_only_migration(self, *, command_body=None, smoke_body=None, cwd=None):
        source = self.root / "live.sqlite"
        sqlite3.connect(source).close()
        scripts = self.feature_worktree / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "feature-migrate.py").write_text(command_body or (
            "import os, sqlite3\n"
            "c=sqlite3.connect(os.environ['DB'])\n"
            "c.execute('create table if not exists added(id integer)')\n"
            "c.commit(); c.close()\n"
        ), encoding="utf-8")
        (scripts / "feature-smoke.py").write_text(smoke_body or (
            "import os, sqlite3\n"
            "c=sqlite3.connect(os.environ['DB'])\n"
            "assert c.execute(\"select count(*) from sqlite_master where name='added'\").fetchone()[0] == 1\n"
            "c.close()\n"
        ), encoding="utf-8")
        self.contract["allowed_paths"].append("scripts/")
        self.contract["backups"]["sqlite"] = [{"path": str(source), "services": []}]
        self.contract["migration_checks"] = [{
            "database": str(source), "database_env": "DB", "cwd": str(cwd or self.repo),
            "command": [sys.executable, "scripts/feature-migrate.py"],
            "application_smoke_command": [sys.executable, "scripts/feature-smoke.py"],
            "idempotent": True, "verification_sql": ["select * from added"], "timeout": 30,
        }]
        cmd("git", "-C", str(self.feature_worktree), "add", "scripts")
        return source

    def test_feature_only_migration_runs_from_exact_checkout(self):
        source = self.configure_feature_only_migration()
        self.assertFalse((self.repo / "scripts" / "feature-migrate.py").exists())
        result = self.stage_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        evidence = state["migration_checkout"]
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertEqual(evidence["initial_fingerprint"]["head"], state["feature_head"])
        self.assertEqual(evidence["initial_fingerprint"], evidence["final_fingerprint"])
        self.assertTrue(evidence["cleanup"]["passed"])
        self.assertFalse(Path(evidence["checkout_path"]).exists())
        check = state["migration_validation"][0]
        self.assertEqual(Path(check["cwd"]), Path(evidence["checkout_path"]))
        self.assertTrue(all(item["unchanged"] for item in check["attempt_fingerprints"]))
        self.assertTrue(check["application_smoke_fingerprint"]["unchanged"])
        with sqlite3.connect(source) as live:
            self.assertEqual(live.execute("select count(*) from sqlite_master where name='added'").fetchone()[0], 0)

    def test_migration_checkout_cwd_mapping_and_refusal(self):
        checkout = self.root / "checkout"
        (checkout / "scripts").mkdir(parents=True)
        self.assertEqual(integration.migration_checkout_cwd(self.repo, self.repo, checkout), checkout)
        self.assertEqual(
            integration.migration_checkout_cwd(self.repo / "scripts", self.repo, checkout),
            checkout / "scripts",
        )
        external = self.root / "external"
        external.mkdir()
        self.assertEqual(integration.migration_checkout_cwd(external, self.repo, checkout), external)
        with self.assertRaisesRegex(integration.IntegrationError, "escapes"):
            integration.migration_checkout_cwd("../outside", self.repo, checkout)
        (checkout / "linked").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(integration.IntegrationError, "escapes"):
            integration.migration_checkout_cwd("linked", self.repo, checkout)

    def assert_migration_checkout_mutation_refused(self):
        before = integration.repository_state(self.repo)
        _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertFalse(state["live_mutation_started"])
        self.assertEqual(integration.repository_state(self.repo), before)
        self.assertTrue(state["migration_checkout"]["cleanup"]["passed"])
        self.assertFalse(Path(state["migration_checkout"]["checkout_path"]).exists())

    def test_migration_checkout_mutation_is_refused(self):
        self.configure_feature_only_migration(
            command_body="from pathlib import Path\nPath('feature.txt').write_text('changed')\n",
        )
        self.assert_migration_checkout_mutation_refused()

    def test_smoke_checkout_mutation_is_refused(self):
        self.configure_feature_only_migration(
            smoke_body="from pathlib import Path\nPath('feature.txt').write_text('changed')\n",
        )
        self.assert_migration_checkout_mutation_refused()

    def test_migration_failure_and_cleanup_failure_are_pre_mutation(self):
        self.configure_feature_only_migration(command_body="raise SystemExit(9)\n")
        before = integration.repository_state(self.repo)
        _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertFalse(state["live_mutation_started"])
        self.assertEqual(integration.repository_state(self.repo), before)
        self.assertTrue(state["migration_checkout"]["cleanup"]["passed"])

    def test_migration_cleanup_failure_is_pre_mutation(self):
        self.configure_feature_only_migration()
        before = integration.repository_state(self.repo)
        real_cleanup = integration.cleanup_validation_checkout
        calls = 0

        def fail_second_cleanup(*args, **kwargs):
            nonlocal calls
            calls += 1
            record_value = real_cleanup(*args, **kwargs)
            if calls == 2:
                record_value["passed"] = False
                record_value["bounded_output"] = "synthetic cleanup verification failure"
            return record_value

        with mock.patch.object(integration, "cleanup_validation_checkout", side_effect=fail_second_cleanup):
            _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertFalse(state["live_mutation_started"])
        self.assertEqual(integration.repository_state(self.repo), before)
        self.assertFalse(state["migration_checkout"]["cleanup"]["passed"])

    def test_migration_checkout_creation_failure_is_pre_mutation(self):
        self.configure_feature_only_migration()
        before = integration.repository_state(self.repo)
        real_create = integration.create_validation_checkout
        calls = 0

        def fail_second_create(*args, **kwargs):
            nonlocal calls
            calls += 1
            parent, checkout, record_value = real_create(*args, **kwargs)
            if calls == 2:
                real_cleanup = integration.cleanup_validation_checkout(args[0], parent, checkout)
                self.assertTrue(real_cleanup["passed"])
                parent.mkdir()
                record_value["passed"] = False
                record_value["exit_code"] = 1
            return parent, checkout, record_value

        with mock.patch.object(integration, "create_validation_checkout", side_effect=fail_second_create):
            _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertFalse(state["live_mutation_started"])
        self.assertEqual(integration.repository_state(self.repo), before)
        self.assertTrue(state["migration_checkout"]["cleanup"]["passed"])


    def test_validation_tracked_mutation_refused_before_live_mutation(self):
        command = [
            sys.executable, "-c",
            "from pathlib import Path; Path('feature.txt').write_text('validation mutation\\n')",
        ]
        _, state, before = self._stage_with_pre_merge(command)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        self.assertFalse(evidence["commands"][0]["fingerprint_unchanged"])

    def test_validation_staged_mutation_refused_before_live_mutation(self):
        command = ["git", "update-index", "--chmod=+x", "feature.txt"]
        _, state, before = self._stage_with_pre_merge(command)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        item = evidence["commands"][0]
        self.assertNotEqual(item["before"]["index_tree"], item["after"]["index_tree"])

    def test_validation_untracked_and_symlink_mutations_refused_before_live_mutation(self):
        command = [
            sys.executable, "-c",
            "from pathlib import Path; import os; Path('created.txt').write_text('created'); "
            "os.symlink('feature.txt', 'created-link')",
        ]
        _, state, before = self._stage_with_pre_merge(command)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        item = evidence["commands"][0]
        self.assertIn("created.txt", item["after"]["status"])
        self.assertTrue(any(link["path"] == "created-link" for link in item["after"]["symlinks"]))

    def test_validation_head_movement_refused_before_live_mutation(self):
        command = ["git", "commit", "--allow-empty", "-m", "validation head movement"]
        _, state, before = self._stage_with_pre_merge(command)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        item = evidence["commands"][0]
        self.assertNotEqual(item["before"]["head"], item["after"]["head"])

    def test_validation_branch_attachment_refused_before_live_mutation(self):
        command = ["git", "switch", "-c", "validation-attached"]
        _, state, before = self._stage_with_pre_merge(command)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        self.assertEqual(evidence["commands"][0]["after"]["branch"], "validation-attached")

    def test_validation_worktree_creation_failure_has_no_production_mutation(self):
        args = self._stage_args()
        before = integration.repository_state(self.repo)
        real_run = integration.run

        def fail_add(command, **kwargs):
            values = [str(value) for value in command]
            if "worktree" in values and "add" in values:
                return subprocess.CompletedProcess(values, 1, "", "synthetic creation failure")
            return real_run(command, **kwargs)

        with mock.patch.object(integration, "run", side_effect=fail_add):
            _, state = integration.stage(self.path, self.manifest, args)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        self.assertFalse(evidence["creation"]["passed"])
        self.assertTrue(evidence["cleanup"]["passed"])

    def test_validation_prune_failure_has_no_production_mutation(self):
        args = self._stage_args()
        before = integration.repository_state(self.repo)
        real_run = integration.run

        def fail_prune(command, **kwargs):
            values = [str(value) for value in command]
            if len(values) >= 2 and values[-2:] == ["worktree", "prune"]:
                return subprocess.CompletedProcess(values, 1, "", "synthetic prune failure")
            return real_run(command, **kwargs)

        with mock.patch.object(integration, "run", side_effect=fail_prune):
            _, state = integration.stage(self.path, self.manifest, args)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        self.assertFalse(evidence["cleanup"]["passed"])
        self.assertEqual(evidence["cleanup"]["prune_exit_code"], 1)
        self.assertIn("cleanup or prune failed", state["failure"])

    def test_failed_pre_merge_evidence_is_preserved_and_checkout_removed(self):
        command = [sys.executable, "-c", "print('expected failure'); raise SystemExit(7)"]
        run_dir, state, before = self._stage_with_pre_merge(command)
        evidence = self._assert_pre_mutation_validation_refusal(state, before)
        item = evidence["commands"][0]
        self.assertFalse(item["command"]["passed"])
        self.assertEqual(item["command"]["exit_code"], 7)
        self.assertEqual(item["command"]["cwd"], evidence["checkout_path"])
        self.assertTrue(Path(item["command"]["output_artifact"]).is_file())
        self.assertTrue((run_dir / "validation-checkout.json").is_file())

    def test_post_merge_test_cwd_is_staged_production_main(self):
        result = self.stage_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertEqual(state["post_merge_tests"][0]["cwd"], str(self.repo.resolve()))
        self.assertEqual(integration.repository_state(self.repo)["head"], state["staged_head"])

    def _stage_args(self, service_manager=None, timeout=1):
        self.bind_contract()
        return type("Args", (), {
            "repository": str(self.repo), "result_root": str(self.results), "backup_root": str(self.backups),
            "service_manager": service_manager or ["true"], "service_timeout": timeout,
        })()

    def test_merge_conflict_stops_and_rolls_back(self):
        real_git = integration.git

        def conflict(repo, *args):
            if args and args[0] == "merge" and "--abort" not in args:
                return subprocess.CompletedProcess(args, 1, "", "synthetic conflict")
            return real_git(repo, *args)

        with mock.patch.object(integration, "git", side_effect=conflict):
            _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "ROLLED_BACK")
        self.assertIn("local merge failed", state["failure"])

    def test_pre_merge_test_failure_performs_no_git_or_service_rollback(self):
        self.contract["pre_merge_tests"] = [{
            "name": "pre-fail", "command": [sys.executable, "-c", "raise SystemExit(7)"], "timeout": 30,
        }]
        before = integration.repository_state(self.repo)
        result = self.stage_cli()
        self.assertEqual(result.returncode, 3)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertEqual(integration.repository_state(self.repo), before)
        self.assertFalse((integration.git_common_dir(self.repo) / "kven-integration-active.json").exists())

    def test_migration_validation_failure_leaves_live_database_and_git_untouched(self):
        database = self.root / "migration-live.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("create table values_table(value text)")
            connection.execute("insert into values_table values ('original')")
        self.contract["backups"]["sqlite"] = [{"path": str(database), "services": []}]
        self.contract["migration_checks"] = [{
            "database": str(database),
            "command": [sys.executable, "-c", "raise SystemExit(8)"],
            "application_smoke_command": [sys.executable, "-c", "pass"],
            "idempotent": True,
        }]
        before = integration.repository_state(self.repo)
        self.write()
        _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertEqual(integration.repository_state(self.repo), before)
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("select value from values_table").fetchone()[0], "original")

    def test_finalize_requires_acceptance_then_pushes(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        refused = cmd(sys.executable, str(SCRIPT), "finalize", str(run_dir), "--repository", str(self.repo))
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("requires --accept PASS", refused.stderr)
        finalized = cmd(sys.executable, str(SCRIPT), "finalize", str(run_dir), "--accept", "PASS", "--repository", str(self.repo))
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZED")
        self.assertTrue((run_dir / "acceptance-result.json").is_file())
        self.assertTrue((run_dir / "push-evidence.json").is_file())
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), state["staged_head"])
        push = state["push_evidence"]
        self.assertEqual(push["command"], [
            "git", "-C", str(self.repo), "push", "origin", f"{state['staged_head']}:refs/heads/main",
        ])
        self.assertEqual(push["origin_main_before"], self.base)
        self.assertEqual(push["origin_main_after"], state["staged_head"])
        self.assertTrue(push["remote_known_updated"])
        self.assertEqual(push["post_push_reconciliation"]["status"], "PASSED")
        self.assertTrue(Path(push["output_artifact"]).is_file())
        for field in ("started_at", "finished_at", "duration_seconds", "exit_code", "bounded_output"):
            self.assertIn(field, push)

    def test_post_push_tracked_hook_mutation_requires_recovery(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        target = self.repo / "base.txt"
        self.install_pre_push_hook(
            "from pathlib import Path\n"
            f"Path({str(target)!r}).write_text('hook-mutated\\n', encoding='utf-8')"
        )
        finalized = cmd(
            sys.executable, str(SCRIPT), "finalize", str(run_dir), "--accept", "PASS",
            "--repository", str(self.repo),
        )
        self.assertNotEqual(finalized.returncode, 0)
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(target.read_text(), "hook-mutated\n")
        self.assertEqual(
            cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(),
            state["staged_head"],
        )
        self.assertFalse((integration.git_common_dir(self.repo) / "kven-integration-active.json").exists())

    def test_post_push_untracked_hook_mutation_requires_recovery(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        target = self.repo / "hook-untracked.txt"
        self.install_pre_push_hook(
            "from pathlib import Path\n"
            f"Path({str(target)!r}).write_text('preserve\\n', encoding='utf-8')"
        )
        self.assertNotEqual(self.finalize_cli().returncode, 0)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(target.read_text(), "preserve\n")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), state["staged_head"])

    def test_post_push_index_hook_mutation_requires_recovery(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        target = self.repo / "feature.txt"
        self.install_pre_push_hook(
            "import subprocess\nfrom pathlib import Path\n"
            f"Path({str(target)!r}).write_text('index mutation\\n', encoding='utf-8')\n"
            f"subprocess.run(['git', '-C', {str(self.repo)!r}, 'add', 'feature.txt'], check=True)"
        )
        self.assertNotEqual(self.finalize_cli().returncode, 0)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertNotEqual(integration.repository_state(self.repo)["index_tree"], state["expected_repository_state"]["index_tree"])

    def test_post_push_branch_hook_mutation_requires_recovery(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        self.install_pre_push_hook(
            "import subprocess\n"
            f"repo={str(self.repo)!r}\n"
            "subprocess.run(['git', '-C', repo, 'branch', 'hook-branch', 'HEAD'], check=True)\n"
            "subprocess.run(['git', '-C', repo, 'symbolic-ref', 'HEAD', 'refs/heads/hook-branch'], check=True)"
        )
        self.assertNotEqual(self.finalize_cli().returncode, 0)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(integration.repository_state(self.repo)["branch"], "hook-branch")

    def _post_push_marker_failure(self, category):
        marker = self.root / f"post-push-{category}-marker"
        self.contract[category] = [{
            "name": category,
            "command": [
                sys.executable, "-c",
                f"from pathlib import Path; raise SystemExit(7 if Path({str(marker)!r}).exists() else 0)",
            ],
            "timeout": 30,
        }]
        self.assertEqual(self.stage_cli().returncode, 0)
        self.install_pre_push_hook(f"from pathlib import Path\nPath({str(marker)!r}).write_text('pushed')")
        self.assertNotEqual(self.finalize_cli().returncode, 0)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(state["push_evidence"]["post_push_reconciliation"]["status"], "FAILED")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), state["staged_head"])

    def test_post_push_readiness_marker_failure_requires_recovery(self):
        self._post_push_marker_failure("readiness_checks")

    def test_post_push_fatal_log_marker_failure_requires_recovery(self):
        self._post_push_marker_failure("fatal_log_checks")

    def test_post_push_readiness_repository_mutation_requires_recovery(self):
        counter = self.root / "readiness-counter"
        target = self.repo / "base.txt"
        code = (
            "from pathlib import Path; "
            f"c=Path({str(counter)!r}); t=Path({str(target)!r}); "
            "n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n)); "
            "t.write_text('post-push readiness mutation\\n') if n >= 3 else None"
        )
        self.contract["readiness_checks"] = [{"name": "mutating-readiness", "command": [sys.executable, "-c", code], "timeout": 30}]
        self.assertEqual(self.stage_cli().returncode, 0)
        self.install_pre_push_hook("pass")
        self.assertNotEqual(self.finalize_cli().returncode, 0)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(target.read_text(), "post-push readiness mutation\n")

    def test_failed_push_with_unchanged_remote_remains_retryable(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        self.install_pre_push_hook("raise SystemExit(9)")
        result = self.finalize_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("finalize is retryable", result.stderr)
        state = json.loads((self.run_dir() / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "FINALIZING")
        self.assertFalse(state["push_evidence"]["remote_known_updated"])
        self.assertEqual(state["push_evidence"]["origin_main_after"], self.base)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)
        self.assertTrue((integration.git_common_dir(self.repo) / "kven-integration-active.json").exists())

    def test_recovery_required_refuses_repeat_without_remote_rewrite(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        target = self.repo / "base.txt"
        self.install_pre_push_hook(
            "from pathlib import Path\n"
            f"Path({str(target)!r}).write_text('preserve after push\\n', encoding='utf-8')"
        )
        self.assertNotEqual(self.finalize_cli().returncode, 0)
        remote_after_push = cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip()
        repeat = self.finalize_cli()
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn("automatic finalize and ordinary rollback are refused", repeat.stderr)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), remote_after_push)
        self.assertEqual(target.read_text(), "preserve after push\n")

    def test_finalize_before_awaiting_acceptance_is_refused(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        state["status"] = "INTERRUPTED"
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        marker_path = integration.git_common_dir(self.repo) / "kven-integration-active.json"
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "not awaiting acceptance"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertTrue(marker_path.exists())

    def test_repeated_finalize_is_safe(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        command = [sys.executable, str(SCRIPT), "finalize", str(run_dir), "--accept", "PASS", "--repository", str(self.repo)]
        self.assertEqual(cmd(*command).returncode, 0)
        first = cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip()
        self.assertEqual(cmd(*command).returncode, 0)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), first)

    def interrupted_finalize_fixture(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        integration.persist(run_dir, state, "FINALIZING", "FINALIZING")
        pushed = cmd("git", "-C", str(self.repo), "push", "origin", f"{state['staged_head']}:refs/heads/main")
        self.assertEqual(pushed.returncode, 0, pushed.stderr)
        return run_dir, state

    def test_interrupted_finalize_dirty_tracked_requires_recovery(self):
        run_dir, state = self.interrupted_finalize_fixture()
        (self.repo / "base.txt").write_text("operator drift\n", encoding="utf-8")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "recovery is required"):
                integration.finalize_run(run_dir, state, args, marker)
        persisted = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(persisted["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual((self.repo / "base.txt").read_text(), "operator drift\n")

    def test_interrupted_finalize_untracked_requires_recovery(self):
        run_dir, state = self.interrupted_finalize_fixture()
        untracked = self.repo / "operator-untracked.txt"
        untracked.write_text("preserve\n", encoding="utf-8")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "recovery is required"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertTrue(untracked.is_file())
        self.assertEqual(json.loads((run_dir / "integration-manifest.json").read_text())["status"], "FINALIZE_RECOVERY_REQUIRED")

    def test_interrupted_finalize_exact_clean_reconciliation_finalizes(self):
        run_dir, state = self.interrupted_finalize_fixture()
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            result = integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(result["status"], "FINALIZED")

    def test_clean_interrupted_push_uses_shared_reconciliation(self):
        run_dir, state = self.interrupted_finalize_fixture()
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            result = integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(result["status"], "FINALIZED")
        self.assertEqual(result["push_evidence"]["post_push_reconciliation"]["status"], "PASSED")

    def test_interrupted_finalize_unexpected_remote_is_internal_error(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        integration.persist(run_dir, state, "FINALIZING", "FINALIZING")
        cmd("git", "-C", str(self.repo), "update-ref", "refs/remotes/origin/main", self.feature)
        result = self.finalize_cli(run_dir)
        self.assertNotEqual(result.returncode, 0)
        persisted = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(persisted["status"], "INTERNAL_ERROR")
        self.assertFalse((integration.git_common_dir(self.repo) / "kven-integration-active.json").exists())
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def test_already_finalized_drift_is_non_mutating_error(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        command = [sys.executable, str(SCRIPT), "finalize", str(run_dir), "--accept", "PASS", "--repository", str(self.repo)]
        self.assertEqual(cmd(*command).returncode, 0)
        pushed = cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip()
        untracked = self.repo / "post-finalize-drift.txt"
        untracked.write_text("preserve\n", encoding="utf-8")
        result = cmd(*command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repository changed", result.stderr)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), pushed)
        self.assertTrue(untracked.is_file())

    def test_finalize_refuses_unexpected_git_change(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        (self.repo / "feature.txt").write_text("operator data\n", encoding="utf-8")
        result = cmd(sys.executable, str(SCRIPT), "finalize", str(run_dir), "--accept", "PASS", "--repository", str(self.repo))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repository changed", result.stderr)
        self.assertEqual((self.repo / "feature.txt").read_text(), "operator data\n")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def test_stage_readiness_tracked_mutation_never_awaits_acceptance(self):
        target = self.repo / "base.txt"
        self.contract["readiness_checks"] = [{
            "name": "dirty-tracked", "command": [sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).write_text('readiness mutation')"],
            "timeout": 30,
        }]
        self.write()
        _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "INTERNAL_ERROR")
        self.assertNotEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertEqual(target.read_text(), "readiness mutation")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def test_stage_readiness_untracked_mutation_is_preserved_and_refused(self):
        target = self.repo / "readiness-untracked.txt"
        self.contract["readiness_checks"] = [{
            "name": "dirty-untracked", "command": [sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).write_text('preserve')"],
            "timeout": 30,
        }]
        self.write()
        _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "INTERNAL_ERROR")
        self.assertEqual(target.read_text(), "preserve")

    def test_stage_fatal_log_mutation_causes_guarded_rollback_refusal(self):
        target = self.repo / "feature.txt"
        self.contract["fatal_log_checks"] = [{
            "name": "dirty-fatal", "command": [sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).write_text('fatal mutation')"],
            "timeout": 30,
        }]
        self.write()
        _, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "INTERNAL_ERROR")
        self.assertIn("refusing rollback", state["rollback_failure"])
        self.assertEqual(target.read_text(), "fatal mutation")

    def _assert_finalize_check_mutation_refused(self, category):
        marker = self.root / f"{category}-second-run"
        target = self.repo / "base.txt"
        code = (
            "from pathlib import Path; "
            f"m=Path({str(marker)!r}); t=Path({str(target)!r}); "
            "t.write_text('finalize mutation') if m.exists() else m.write_text('stage complete')"
        )
        self.contract[category] = [{"name": category, "command": [sys.executable, "-c", code], "timeout": 30}]
        self.write()
        run_dir, state = integration.stage(self.path, self.manifest, self._stage_args())
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as active:
            with self.assertRaisesRegex(integration.IntegrationError, "repository changed"):
                integration.finalize_run(run_dir, state, args, active)
        self.assertEqual(target.read_text(), "finalize mutation")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def test_finalize_readiness_repository_mutation_prevents_push(self):
        self._assert_finalize_check_mutation_refused("readiness_checks")

    def test_finalize_fatal_log_repository_mutation_prevents_push(self):
        self._assert_finalize_check_mutation_refused("fatal_log_checks")

    def test_dirty_tracked_and_untracked_rollback_refused_without_loss(self):
        for untracked in (False, True):
            with self.subTest(untracked=untracked):
                if self.results.exists():
                    shutil_rmtree(self.results)
                marker = integration.git_common_dir(self.repo) / "kven-integration-active.json"
                marker.unlink(missing_ok=True)
                if integration.repository_state(self.repo)["head"] != self.base:
                    cmd("git", "-C", str(self.repo), "reset", "--hard", self.base)
                self.assertEqual(self.stage_cli().returncode, 0)
                run_dir = self.run_dir()
                target = self.repo / ("operator-untracked.txt" if untracked else "feature.txt")
                target.write_text("must survive\n", encoding="utf-8")
                result = cmd(sys.executable, str(SCRIPT), "rollback", str(run_dir), "--repository", str(self.repo))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(target.read_text(), "must survive\n")
                state = json.loads((run_dir / "integration-manifest.json").read_text())
                self.assertEqual(state["status"], "INTERNAL_ERROR")
                target.unlink() if untracked else cmd("git", "-C", str(self.repo), "restore", "feature.txt")
                cmd("git", "-C", str(self.repo), "reset", "--hard", self.base)
                marker.unlink(missing_ok=True)

    def test_summary_is_generated_with_exact_failure_reason(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        state["failure"] = "synthetic exact refusal"
        integration.summary(run_dir, state)
        summary = (run_dir / "integration-summary.md").read_text()
        self.assertIn("synthetic exact refusal", summary)
        self.assertIn("Finalize after PASS", summary)

    def test_persisted_run_state_types_are_strict(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        state["service_manager"] = "unsafe shell string"
        (run_dir / "integration-manifest.json").write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(integration.IntegrationError, "service_manager must be an argv array"):
            integration.load_run(str(run_dir))

    def test_persisted_contract_mutation_is_refused_before_finalize_and_rollback(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        original = json.loads((run_dir / "integration-manifest.json").read_text())
        mutated = copy.deepcopy(original)
        mutated["contract"]["acceptance_checklist"] = ["Package-only mutation"]
        mutated["deployment_contract_sha256"] = integration.canonical_contract_sha256(
            integration.validate_contract(mutated["contract"], mutated)
        )
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "differs from exact committed contract"):
                integration.finalize_run(run_dir, copy.deepcopy(mutated), args, marker)
            with self.assertRaisesRegex(integration.IntegrationError, "differs from exact committed contract"):
                integration.rollback_state(run_dir, copy.deepcopy(mutated), "fixture", marker=marker)

    def test_concurrent_active_stage_is_refused(self):
        marker = integration.git_common_dir(self.repo) / "kven-integration-active.json"
        integration.atomic_json(marker, {"run_id": "different-run", "run_dir": "/tmp/different", "repository": str(self.repo)}, mode=0o600)
        try:
            with self.assertRaisesRegex(integration.IntegrationError, "another integration run is active"):
                integration.stage(self.path, self.manifest, self._stage_args())
        finally:
            marker.unlink(missing_ok=True)


def shutil_rmtree(path):
    import shutil
    shutil.rmtree(path)


class BackupMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_wal_online_backup_integrity_and_connection_close(self):
        source = self.root / "live.sqlite"
        backup = self.root / "protected" / "copy.sqlite"
        connection = sqlite3.connect(source)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE private(value TEXT)")
        connection.execute("INSERT INTO private VALUES ('synthetic')")
        connection.commit()
        result = integration.sqlite_backup(source, backup)
        self.assertEqual(result["integrity_check"], "ok")
        with sqlite3.connect(backup) as verify:
            self.assertEqual(verify.execute("SELECT value FROM private").fetchone()[0], "synthetic")
        connection.close()
        os.replace(source, self.root / "renamed.sqlite")
        os.replace(backup, self.root / "renamed-backup.sqlite")

    def test_stale_wal_cannot_override_restore(self):
        source = self.root / "live.sqlite"
        with sqlite3.connect(source) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE values_table(value TEXT)")
            connection.execute("INSERT INTO values_table VALUES ('baseline')")
        backup = self.root / "protected" / "copy.sqlite"
        result = integration.sqlite_backup(source, backup)
        stale = sqlite3.connect(source)
        stale.execute("PRAGMA journal_mode=WAL")
        stale.execute("UPDATE values_table SET value='newer-wal'")
        stale.commit()
        self.assertTrue(Path(str(source) + "-wal").exists())
        restored = integration.restore_backups([result], service_manager=["true"])
        stale.close()
        with sqlite3.connect(source) as verify:
            self.assertEqual(verify.execute("SELECT value FROM values_table").fetchone()[0], "baseline")
        self.assertTrue(restored[0]["moved_sidecars"])

    def test_tampered_protected_backup_is_rejected(self):
        source = self.root / "live.sqlite"
        sqlite3.connect(source).close()
        backup = self.root / "protected" / "copy.sqlite"
        result = integration.sqlite_backup(source, backup)
        with backup.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(integration.IntegrationError, "checksum/size mismatch"):
            integration.restore_backups([result], service_manager=["true"])

    def test_config_restore_preserves_content_and_metadata(self):
        source = self.root / "config"
        source.write_text("original\n", encoding="utf-8")
        source.chmod(0o640)
        backup = self.root / "protected" / "config"
        result = integration.config_backup(source, backup)
        source.write_text("changed\n", encoding="utf-8")
        source.chmod(0o600)
        integration.restore_backups([result], service_manager=["true"])
        self.assertEqual(source.read_text(), "original\n")
        self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o640)
        self.assertEqual(source.stat().st_uid, result["uid"])
        self.assertEqual(source.stat().st_gid, result["gid"])

    def test_configuration_tamper_and_symlink_rejected(self):
        source = self.root / "config"
        source.write_text("original\n", encoding="utf-8")
        result = integration.config_backup(source, self.root / "protected" / "config")
        Path(result["backup"]).write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(integration.IntegrationError, "checksum/size mismatch"):
            integration.restore_backups([result], service_manager=["true"])

    def test_declared_direct_and_parent_symlinks_are_rejected_before_backup(self):
        real_config = self.root / "real.conf"
        real_config.write_text("fixture\n", encoding="utf-8")
        direct_config = self.root / "declared.conf"
        direct_config.symlink_to(real_config)
        contract = {"backups": {"sqlite": [], "configuration": [{"path": str(direct_config), "services": []}]}}
        with self.assertRaisesRegex(integration.IntegrationError, "symlink"):
            integration.create_backups(contract, self.root / "protected-config")

        real_database = self.root / "real.sqlite"
        sqlite3.connect(real_database).close()
        direct_database = self.root / "declared.sqlite"
        direct_database.symlink_to(real_database)
        contract = {"backups": {"sqlite": [{"path": str(direct_database), "services": []}], "configuration": []}}
        with self.assertRaisesRegex(integration.IntegrationError, "symlink"):
            integration.create_backups(contract, self.root / "protected-database")

        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        nested = real_parent / "nested.conf"
        nested.write_text("fixture\n", encoding="utf-8")
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        contract = {"backups": {"sqlite": [], "configuration": [{"path": str(linked_parent / "nested.conf"), "services": []}]}}
        with self.assertRaisesRegex(integration.IntegrationError, "symlink"):
            integration.create_backups(contract, self.root / "protected-parent")

    def test_restore_rejects_inode_identity_substitution(self):
        source = self.root / "identity.conf"
        source.write_text("original\n", encoding="utf-8")
        record_value = integration.config_backup(source, self.root / "protected" / "identity.conf")
        replacement = self.root / "replacement.conf"
        replacement.write_text("replacement\n", encoding="utf-8")
        os.replace(replacement, source)
        with self.assertRaisesRegex(integration.IntegrationError, "identity changed"):
            integration.restore_backups([record_value], service_manager=["true"])
        result = integration.config_backup(source, self.root / "protected" / "config2")
        source.unlink(); source.symlink_to(self.root / "other")
        with self.assertRaisesRegex(integration.IntegrationError, "symlink"):
            integration.restore_backups([result], service_manager=["true"])

    def migration_fixture(self):
        source = self.root / "source.sqlite"
        sqlite3.connect(source).close()
        backup = self.root / "protected" / "copy.sqlite"
        result = integration.sqlite_backup(source, backup)
        migration = self.root / "migration.py"
        migration.write_text("import os,sqlite3; c=sqlite3.connect(os.environ['DB']); c.execute('create table if not exists added(id integer)'); c.commit(); c.close()", encoding="utf-8")
        smoke = self.root / "smoke.py"
        smoke.write_text("import os,sqlite3; c=sqlite3.connect(os.environ['DB']); c.execute('insert into added values (1)'); c.commit(); assert c.execute('select count(*) from added').fetchone()[0] >= 1; c.close()", encoding="utf-8")
        contract = {"migration_checks": [{
            "database": str(source), "database_env": "DB", "command": [sys.executable, str(migration)],
            "application_smoke_command": [sys.executable, str(smoke)], "idempotent": True,
            "verification_sql": ["select * from added"], "timeout": 30,
        }]}
        return source, result, contract

    def test_migration_disposable_idempotent_and_live_unchanged(self):
        source, backup, contract = self.migration_fixture()
        result = integration.validate_migrations(contract, [backup], self.root / "run")
        self.assertEqual(len(result[0]["attempts"]), 2)
        self.assertTrue(result[0]["application_smoke"]["passed"])
        with sqlite3.connect(source) as live:
            self.assertEqual(live.execute("select count(*) from sqlite_master where name='added'").fetchone()[0], 0)

    def test_migration_and_application_smoke_fail_before_live_mutation(self):
        source, backup, contract = self.migration_fixture()
        contract["migration_checks"][0]["command"] = [sys.executable, "-c", "raise SystemExit(3)"]
        with self.assertRaisesRegex(integration.IntegrationError, "migration validation failed"):
            integration.validate_migrations(contract, [backup], self.root / "failed-migration")
        _, backup, contract = self.migration_fixture()
        contract["migration_checks"][0]["application_smoke_command"] = [sys.executable, "-c", "raise SystemExit(4)"]
        with self.assertRaisesRegex(integration.IntegrationError, "application migration smoke failed"):
            integration.validate_migrations(contract, [backup], self.root / "failed-smoke")
        with sqlite3.connect(source) as live:
            self.assertEqual(live.execute("select count(*) from sqlite_master where name='added'").fetchone()[0], 0)

    def test_redaction_in_argv_and_output(self):
        credential_value = "synthetic-private-value"
        credential_assignment = "tok" + "en=" + credential_value
        artifact = self.root / "full.log"
        result = integration.command_record(
            [sys.executable, "-c", f"print({credential_assignment!r})", "--password", credential_value], artifact,
        )
        serialized = json.dumps(result) + artifact.read_text()
        self.assertNotIn(credential_value, serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_large_output_bounded_and_full_artifact_retained(self):
        artifact = self.root / "full.log"
        result = integration.command_record([sys.executable, "-c", "print('x'*10000)"], artifact)
        self.assertGreater(artifact.stat().st_size, len(result["bounded_output"]))
        self.assertIn("truncated", result["bounded_output"])

    def test_sensitive_backups_excluded_from_result_artifacts(self):
        private = "synthetic database private content"
        source = self.root / "private.sqlite"
        with sqlite3.connect(source) as connection:
            connection.execute("create table private(value text)")
            connection.execute("insert into private values (?)", (private,))
        backup = integration.sqlite_backup(source, self.root / "protected" / "copy.sqlite")
        run_dir = self.root / "user-readable"
        integration.atomic_json(run_dir / "backup-manifest.json", [backup])
        contents = "".join(path.read_text(errors="ignore") for path in run_dir.rglob("*") if path.is_file())
        self.assertNotIn(private, contents)


class FakeServiceTests(RepositoryFixture):
    def setUp(self):
        super().setUp()
        self.service_state_file = self.root / "service-state.json"
        self.manager_script = self.root / "fake-service-manager.py"
        self.manager_script.write_text("""import json, pathlib, sys
path=pathlib.Path(sys.argv[1]); action=sys.argv[2]; service=sys.argv[3]
data=json.loads(path.read_text()); item=data[service]
if action=='is-active': print(item['state']); raise SystemExit(0 if item['state']=='active' else 3)
if action=='restart':
 if item.get('restart_fail'): raise SystemExit(5)
 item['restarts']=item.get('restarts',0)+(10 if item.get('loop') else 1)
 if not item.get('timeout'): item['state']='active'
if action=='start':
 if item.get('start_fail'): item['state']='failed'; path.write_text(json.dumps(data)); raise SystemExit(6)
 item['state']='active'
if action=='stop': item['state']='inactive'
if action=='show':
 if item.get('show_fail'): raise SystemExit(9)
 print(item.get('restart_output',item.get('restarts',0)))
if action=='status': print(item['state'])
path.write_text(json.dumps(data))
""", encoding="utf-8")
        self.manager = [sys.executable, str(self.manager_script), str(self.service_state_file)]
        self.set_service("active")

    def set_service(self, state, **behavior):
        value = {"state": state, "restarts": 0}
        value.update(behavior)
        self.service_state_file.write_text(json.dumps({"kven2-main.service": value}), encoding="utf-8")

    def service_contract(self, readiness_exit=0, fatal_exit=0):
        def one_shot(name, exit_code):
            marker = self.root / f"{name}-failure-marker"
            marker.unlink(missing_ok=True)
            if not exit_code:
                return [sys.executable, "-c", "pass"]
            code = f"import pathlib; p=pathlib.Path({str(marker)!r}); exists=p.exists(); p.write_text('seen'); raise SystemExit(0 if exists else {exit_code})"
            return [sys.executable, "-c", code]

        self.contract["services"] = ["kven2-main.service"]
        self.contract["readiness_checks"] = [{"name": "readiness", "command": one_shot("readiness", readiness_exit), "timeout": 30}]
        self.contract["fatal_log_checks"] = [{"name": "fatal-log", "command": one_shot("fatal", fatal_exit), "timeout": 30}]

    def stage_service(self, timeout=1):
        self.write()
        return integration.stage(self.path, self.manifest, self._stage_args(self.manager, timeout))

    def test_only_allowlisted_services_may_restart(self):
        self.contract["services"] = ["unknown.service"]
        with self.assertRaisesRegex(integration.IntegrationError, "not allowlisted"):
            integration.validate_contract(self.contract, self.manifest)
        with self.assertRaisesRegex(integration.IntegrationError, "not allowlisted"):
            integration.service_action(self.manager, "restart", "unknown.service")

    def test_service_timeout_failed_activation_and_restart_loop(self):
        self.set_service("inactive", timeout=True)
        with self.assertRaisesRegex(integration.IntegrationError, "timeout"):
            integration.restart_declared_service(self.manager, "kven2-main.service", timeout=0.2)
        self.set_service("active", restart_fail=True)
        with self.assertRaisesRegex(integration.IntegrationError, "restart failed"):
            integration.restart_declared_service(self.manager, "kven2-main.service", timeout=1)
        self.set_service("active", loop=True)
        with self.assertRaisesRegex(integration.IntegrationError, "rapid restart loop"):
            integration.restart_declared_service(self.manager, "kven2-main.service", timeout=1)

    def test_restart_count_unavailable_malformed_and_negative_are_refused(self):
        cases = ({"show_fail": True}, {"restart_output": "malformed"}, {"restart_output": -1})
        for behavior in cases:
            with self.subTest(behavior=behavior):
                self.set_service("active", **behavior)
                with self.assertRaisesRegex(integration.IntegrationError, "restart count"):
                    integration.restart_count(self.manager, "kven2-main.service")

    def test_pre_stage_restart_count_unavailable_has_zero_live_mutation(self):
        self.set_service("active", show_fail=True)
        self.service_contract()
        before = integration.repository_state(self.repo)
        _, state = self.stage_service()
        self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
        self.assertEqual(integration.repository_state(self.repo), before)
        data = json.loads(self.service_state_file.read_text())["kven2-main.service"]
        self.assertEqual(data["state"], "active")
        self.assertEqual(data["restarts"], 0)

    def test_post_restart_count_unavailable_triggers_rollback(self):
        self.set_service("active")
        self.service_contract()
        real_restart_count = integration.restart_count
        calls = 0

        def unavailable_after_snapshot(manager, service, timeout=10):
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise integration.IntegrationError("restart count unavailable for fixture")
            return real_restart_count(manager, service, timeout=timeout)

        self.bind_contract()
        with mock.patch.object(integration, "restart_count", side_effect=unavailable_after_snapshot):
            _, state = integration.stage(self.path, self.manifest, self._stage_args(self.manager, 1))
        self.assertEqual(state["status"], "ROLLED_BACK")
        self.assertEqual(integration.repository_state(self.repo)["head"], self.base)

    def test_finalize_restart_count_unavailable_prevents_finalized(self):
        self.service_contract()
        run_dir, state = self.stage_service()
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with mock.patch.object(integration, "restart_count", side_effect=integration.IntegrationError("restart count unavailable")):
            with integration.repository_lock(self.repo) as marker:
                with self.assertRaisesRegex(integration.IntegrationError, "restart count unavailable"):
                    integration.finalize_run(run_dir, state, args, marker)
        self.assertNotEqual(state["status"], "FINALIZED")

    def test_systemd_rc3_preserves_exact_failed_and_transitional_states(self):
        for state in ("failed", "activating", "deactivating", "reloading", "unknown"):
            with self.subTest(state=state):
                self.set_service(state)
                self.assertEqual(integration.service_state(self.manager, "kven2-main.service"), state)

    def test_unsupported_pre_stage_service_state_has_zero_live_mutation(self):
        for unsupported in ("failed", "activating", "deactivating"):
            with self.subTest(state=unsupported):
                marker = integration.git_common_dir(self.repo) / "kven-integration-active.json"
                marker.unlink(missing_ok=True)
                if self.results.exists():
                    shutil_rmtree(self.results)
                self.set_service(unsupported)
                self.service_contract()
                before = integration.repository_state(self.repo)
                _, state = self.stage_service()
                data = json.loads(self.service_state_file.read_text())["kven2-main.service"]
                self.assertEqual(state["status"], "AUTOMATED_CHECK_FAILED")
                self.assertEqual(data["state"], unsupported)
                self.assertEqual(data["restarts"], 0)
                self.assertEqual(integration.repository_state(self.repo), before)
                self.assertFalse(marker.exists())

    def test_delayed_restart_count_drift_after_readiness_is_detected(self):
        self.service_contract()
        code = (
            "import json,pathlib; "
            f"p=pathlib.Path({str(self.service_state_file)!r}); d=json.loads(p.read_text()); "
            "d['kven2-main.service']['restarts']+=1; p.write_text(json.dumps(d))"
        )
        self.contract["readiness_checks"] = [{"name": "delayed-loop", "command": [sys.executable, "-c", code], "timeout": 30}]
        _, state = self.stage_service()
        self.assertEqual(state["status"], "ROLLED_BACK")
        self.assertIn("restart count changed", state["failure"])

    def test_finalize_detects_restart_count_drift(self):
        self.service_contract()
        run_dir, state = self.stage_service()
        data = json.loads(self.service_state_file.read_text())
        data["kven2-main.service"]["restarts"] += 1
        self.service_state_file.write_text(json.dumps(data), encoding="utf-8")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "restart count changed"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def test_finalize_check_service_state_mutation_prevents_push(self):
        self.service_contract()
        marker_path = self.root / "finalize-service-check"
        code = (
            "import json,pathlib; "
            f"m=pathlib.Path({str(marker_path)!r}); p=pathlib.Path({str(self.service_state_file)!r}); "
            "d=json.loads(p.read_text()); "
            "d['kven2-main.service']['state']='inactive' if m.exists() else d['kven2-main.service']['state']; "
            "p.write_text(json.dumps(d)); m.write_text('seen')"
        )
        self.contract["readiness_checks"] = [{"name": "state-drift", "command": [sys.executable, "-c", code], "timeout": 30}]
        run_dir, state = self.stage_service()
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "service state or restart count changed"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def _post_push_service_hook_mutation(self, drift):
        self.service_contract()
        run_dir, state = self.stage_service()
        change = (
            "d['kven2-main.service']['state']='inactive'"
            if drift == "state"
            else "d['kven2-main.service']['restarts']+=1"
        )
        self.install_pre_push_hook(
            "import json\nfrom pathlib import Path\n"
            f"p=Path({str(self.service_state_file)!r})\n"
            "d=json.loads(p.read_text())\n"
            f"{change}\n"
            "p.write_text(json.dumps(d))"
        )
        result = self.finalize_cli(run_dir)
        self.assertNotEqual(result.returncode, 0)
        persisted = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(persisted["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(persisted["push_evidence"]["post_push_reconciliation"]["status"], "FAILED")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), state["staged_head"])

    def test_post_push_service_state_hook_mutation_requires_recovery(self):
        self._post_push_service_hook_mutation("state")

    def test_post_push_restart_hook_mutation_requires_recovery(self):
        self._post_push_service_hook_mutation("restart")

    def test_post_push_readiness_service_mutation_requires_recovery(self):
        self.service_contract()
        counter = self.root / "service-readiness-counter"
        code = (
            "import json; from pathlib import Path; "
            f"c=Path({str(counter)!r}); p=Path({str(self.service_state_file)!r}); "
            "n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n)); "
            "d=json.loads(p.read_text()); "
            "d['kven2-main.service']['state']='inactive' if n >= 3 else d['kven2-main.service']['state']; "
            "p.write_text(json.dumps(d))"
        )
        self.contract["readiness_checks"] = [{"name": "mutating-readiness", "command": [sys.executable, "-c", code], "timeout": 30}]
        run_dir, state = self.stage_service()
        self.install_pre_push_hook("pass")
        result = self.finalize_cli(run_dir)
        self.assertNotEqual(result.returncode, 0)
        persisted = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(persisted["status"], "FINALIZE_RECOVERY_REQUIRED")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), state["staged_head"])

    def test_readiness_and_fatal_log_failures_roll_back(self):
        for readiness, fatal in ((7, 0), (0, 8)):
            with self.subTest(readiness=readiness, fatal=fatal):
                marker = integration.git_common_dir(self.repo) / "kven-integration-active.json"
                marker.unlink(missing_ok=True)
                if integration.repository_state(self.repo)["head"] != self.base:
                    cmd("git", "-C", str(self.repo), "reset", "--hard", self.base)
                if self.results.exists():
                    shutil_rmtree(self.results)
                self.set_service("active")
                self.service_contract(readiness, fatal)
                _, state = self.stage_service()
                self.assertEqual(state["status"], "ROLLED_BACK")
                self.assertEqual(integration.repository_state(self.repo)["head"], self.base)
                self.assertEqual(integration.service_state(self.manager, "kven2-main.service"), "active")

    def test_inactive_service_remains_inactive_after_rollback(self):
        self.set_service("inactive")
        self.service_contract(readiness_exit=9)
        _, state = self.stage_service()
        self.assertEqual(state["status"], "ROLLED_BACK")
        self.assertEqual(integration.service_state(self.manager, "kven2-main.service"), "inactive")

    def test_finalize_refuses_changed_service_and_readiness(self):
        self.service_contract()
        run_dir, state = self.stage_service()
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        staged_service_data = self.service_state_file.read_text()
        self.set_service("inactive")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "service state or restart count changed"):
                integration.finalize_run(run_dir, state, args, marker)
        self.service_state_file.write_text(staged_service_data, encoding="utf-8")
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        state["contract"]["readiness_checks"][0]["command"] = [sys.executable, "-c", "raise SystemExit(2)"]
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "differs from exact committed contract"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

    def _interrupted_finalize_service_drift(self, drift):
        self.service_contract()
        run_dir, state = self.stage_service()
        integration.persist(run_dir, state, "FINALIZING", "FINALIZING")
        cmd("git", "-C", str(self.repo), "push", "origin", f"{state['staged_head']}:refs/heads/main")
        data = json.loads(self.service_state_file.read_text())
        if drift == "state":
            data["kven2-main.service"]["state"] = "inactive"
        else:
            data["kven2-main.service"]["restarts"] += 1
        self.service_state_file.write_text(json.dumps(data), encoding="utf-8")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "recovery is required"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(json.loads((run_dir / "integration-manifest.json").read_text())["status"], "FINALIZE_RECOVERY_REQUIRED")

    def test_interrupted_finalize_service_drift_requires_recovery(self):
        self._interrupted_finalize_service_drift("state")

    def test_interrupted_finalize_restart_drift_requires_recovery(self):
        self._interrupted_finalize_service_drift("restart")

    def test_rollback_restores_fixture_database_and_configuration(self):
        database = self.root / "live.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("create table values_table(value text)")
            connection.execute("insert into values_table values ('original')")
        config = self.root / "live.conf"
        config.write_text("original\n", encoding="utf-8")
        config.chmod(0o640)
        self.service_contract(readiness_exit=5)
        self.contract["backups"] = {
            "sqlite": [{"path": str(database), "services": ["kven2-main.service"]}],
            "configuration": [{"path": str(config), "services": ["kven2-main.service"]}],
        }
        deployment = self.root / "mutate.py"
        deployment.write_text(f"import sqlite3,pathlib; c=sqlite3.connect({str(database)!r}); c.execute(\"update values_table set value='changed'\"); c.commit(); c.close(); pathlib.Path({str(config)!r}).write_text('changed\\n')", encoding="utf-8")
        self.contract["deployment_steps"] = [{"name": "mutate", "command": [sys.executable, str(deployment)], "timeout": 30}]
        _, state = self.stage_service()
        self.assertEqual(state["status"], "ROLLED_BACK")
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("select value from values_table").fetchone()[0], "original")
        self.assertEqual(config.read_text(), "original\n")
        self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o640)


class CoverageTests(unittest.TestCase):
    def test_coverage_map_has_all_37_scenarios(self):
        self.assertEqual(set(SCENARIO_COVERAGE), set(range(1, 38)))
        names = {name for tests in SCENARIO_COVERAGE.values() for name in tests}
        discovered = {name for cls in (RepositoryFixture, BackupMigrationTests, FakeServiceTests, CoverageTests)
                      for name in dir(cls) if name.startswith("test_")}
        self.assertTrue(names <= discovered)

    def test_r4_regression_map_references_discovered_tests(self):
        names = {name for tests in R4_REGRESSION_COVERAGE.values() for name in tests}
        discovered = {name for cls in (RepositoryFixture, BackupMigrationTests, FakeServiceTests, CoverageTests)
                      for name in dir(cls) if name.startswith("test_")}
        self.assertTrue(names <= discovered)

    def test_r5_regression_map_references_discovered_tests(self):
        names = {name for tests in R5_REGRESSION_COVERAGE.values() for name in tests}
        discovered = {name for cls in (RepositoryFixture, BackupMigrationTests, FakeServiceTests, CoverageTests)
                      for name in dir(cls) if name.startswith("test_")}
        self.assertTrue(names <= discovered)


if __name__ == "__main__":
    unittest.main()
