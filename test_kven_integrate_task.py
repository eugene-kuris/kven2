import copy
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


def cmd(*args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def record(name="synthetic-test", passed=True):
    return {
        "name": name, "command": [sys.executable, "-c", "pass"],
        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_seconds": 1.0, "exit_code": 0 if passed else 1, "passed": passed,
        "bounded_output": "", "output_artifact": "/tmp/synthetic-test.log",
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
        cmd("git", "-C", str(self.repo), "add", "base.txt")
        cmd("git", "-C", str(self.repo), "commit", "-m", "base")
        cmd("git", "-C", str(self.repo), "remote", "add", "origin", str(self.remote))
        cmd("git", "-C", str(self.repo), "push", "-u", "origin", "main")
        self.base = cmd("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip()
        cmd("git", "-C", str(self.repo), "worktree", "add", "-b", "feature", str(self.feature_worktree), self.base)
        (self.feature_worktree / "feature.txt").write_text("feature\n", encoding="utf-8")
        cmd("git", "-C", str(self.feature_worktree), "add", "feature.txt")
        cmd("git", "-C", str(self.feature_worktree), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "feature")
        self.feature = cmd("git", "-C", str(self.feature_worktree), "rev-parse", "HEAD").stdout.strip()
        check = {"name": "synthetic", "command": [sys.executable, "-c", "pass"], "timeout": 30}
        self.contract = {
            "schema_version": "2.0", "expected_baseline": self.base, "feature_branch": "feature",
            "allowed_paths": ["feature.txt"], "services": [],
            "backups": {"sqlite": [], "configuration": []}, "migration_checks": [],
            "result_validation_tests": [copy.deepcopy(check)], "pre_merge_tests": [copy.deepcopy(check)],
            "post_merge_tests": [copy.deepcopy(check)], "readiness_checks": [], "fatal_log_checks": [],
            "acceptance_checklist": ["Observe synthetic behavior"],
            "rollback": {"restore_git": True, "restore_backups": True, "verify_readiness": True},
        }
        self.manifest = {
            "schema_version": "2.0", "task_id": "TEST", "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00", "codex_runtime_seconds": 1.0,
            "exit_code": 0, "final_codex_status": "PASS", "repository_path": str(self.repo),
            "baseline_branch": "main", "baseline_head": self.base, "origin_main_head": self.base,
            "feature_branch": "feature", "feature_head": self.feature,
            "worktree_path": str(self.feature_worktree),
            "commits_created": [{"sha": self.feature, "subject": "feature"}],
            "changed_files": ["feature.txt"], "tests": [record()],
            "git_diff_check": {"passed": True}, "secret_scan": {"passed": True},
            "deployment_contract": self.contract,
        }
        self.package = self.root / "package"
        self.package.mkdir()
        self.path = self.package / "result-manifest.json"
        self.results = self.root / "results"
        self.backups = self.root / "backups"
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def inspect(self):
        return integration.inspect_manifest(self.manifest, expected_repository=self.repo)

    def stage_cli(self, *extra):
        self.write()
        return cmd(
            sys.executable, str(SCRIPT), "stage", str(self.path), "--repository", str(self.repo),
            "--result-root", str(self.results), "--backup-root", str(self.backups), *extra,
        )

    def run_dir(self):
        return next((self.results / "TEST").iterdir())

    def test_valid_manifest_parsing_and_inspection(self):
        _, loaded = integration.load_manifest(str(self.package))
        self.assertEqual(loaded["task_id"], "TEST")
        report = self.inspect()
        self.assertTrue(report["eligible"])
        self.assertEqual(report["actual_changed_files"], ["feature.txt"])

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
            integration.inspect_manifest(self.manifest)

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
        cmd("git", "-C", str(self.feature_worktree), "commit", "--allow-empty", "-m", "moved")
        with self.assertRaisesRegex(integration.IntegrationError, "feature HEAD mismatch"):
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

    def test_hidden_actual_changed_path_and_allowed_scope_rejected(self):
        self.manifest["changed_files"] = []
        with self.assertRaisesRegex(integration.IntegrationError, "changed_files mismatch"):
            self.inspect()
        self.manifest["changed_files"] = ["feature.txt"]
        self.contract["allowed_paths"] = ["different.txt"]
        with self.assertRaisesRegex(integration.IntegrationError, "outside contract"):
            self.inspect()

    def test_independent_diff_and_secret_checks_use_exact_commits(self):
        (self.feature_worktree / "feature.txt").write_text("token=synthetic-secret-material\n", encoding="utf-8")
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

    def test_dry_run_is_read_only(self):
        before = integration.repository_state(self.repo)
        result = cmd(sys.executable, str(SCRIPT), "dry-run", str(self.path), "--repository", str(self.repo))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"dry_run": true', result.stdout)
        self.assertIn(self.feature, result.stdout)
        self.assertEqual(before, integration.repository_state(self.repo))

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

    def test_branch_move_between_preflights_is_rejected(self):
        args = self._stage_args()
        real_inspect = integration.inspect_manifest
        calls = 0

        def moving_inspect(*values, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                cmd("git", "-C", str(self.feature_worktree), "commit", "--allow-empty", "-m", "race")
            return real_inspect(*values, **kwargs)

        with mock.patch.object(integration, "inspect_manifest", side_effect=moving_inspect):
            run_dir, state = integration.stage(self.path, self.manifest, args)
        self.assertIn(state["status"], {"ROLLED_BACK", "INTERNAL_ERROR"})
        self.assertIn("feature HEAD mismatch", state["failure"])
        self.assertEqual(integration.repository_state(self.repo)["head"], self.base)

    def _stage_args(self, service_manager=None, timeout=1):
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

    def test_finalize_refuses_unexpected_git_change(self):
        self.assertEqual(self.stage_cli().returncode, 0)
        run_dir = self.run_dir()
        (self.repo / "feature.txt").write_text("operator data\n", encoding="utf-8")
        result = cmd(sys.executable, str(SCRIPT), "finalize", str(run_dir), "--accept", "PASS", "--repository", str(self.repo))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repository changed", result.stderr)
        self.assertEqual((self.repo / "feature.txt").read_text(), "operator data\n")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

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
        result = integration.config_backup(source, self.root / "protected" / "config2")
        source.unlink(); source.symlink_to(self.root / "other")
        with self.assertRaisesRegex(integration.IntegrationError, "not a regular file"):
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
        secret = "synthetic-private-value"
        artifact = self.root / "full.log"
        result = integration.command_record(
            [sys.executable, "-c", f"print('token={secret}')", "--password", secret], artifact,
        )
        serialized = json.dumps(result) + artifact.read_text()
        self.assertNotIn(secret, serialized)
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
if action=='show': print(item.get('restarts',0))
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
        self.set_service("inactive")
        args = type("Args", (), {"repository": str(self.repo), "accept": "PASS", "notes": ""})()
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "service state changed"):
                integration.finalize_run(run_dir, state, args, marker)
        self.set_service("active")
        state = json.loads((run_dir / "integration-manifest.json").read_text())
        state["contract"]["readiness_checks"][0]["command"] = [sys.executable, "-c", "raise SystemExit(2)"]
        with integration.repository_lock(self.repo) as marker:
            with self.assertRaisesRegex(integration.IntegrationError, "finalize-readiness command failed"):
                integration.finalize_run(run_dir, state, args, marker)
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)

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


if __name__ == "__main__":
    unittest.main()
