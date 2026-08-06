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


SCRIPT = Path(__file__).parent / "scripts" / "kven-integrate-task"
loader = importlib.machinery.SourceFileLoader("kven_integrate_task", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
integration = importlib.util.module_from_spec(spec)
loader.exec_module(integration)


def cmd(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.remote = root / "remote.git"; self.repo = root / "repo"
        cmd("git", "init", "--bare", str(self.remote))
        cmd("git", "init", "-b", "main", str(self.repo))
        cmd("git", "-C", str(self.repo), "config", "user.email", "test@example.invalid")
        cmd("git", "-C", str(self.repo), "config", "user.name", "Test")
        (self.repo / "base.txt").write_text("base\n")
        cmd("git", "-C", str(self.repo), "add", "."); cmd("git", "-C", str(self.repo), "commit", "-m", "base")
        cmd("git", "-C", str(self.repo), "remote", "add", "origin", str(self.remote)); cmd("git", "-C", str(self.repo), "push", "-u", "origin", "main")
        self.base = cmd("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip()
        cmd("git", "-C", str(self.repo), "checkout", "-b", "feature")
        (self.repo / "feature.txt").write_text("feature\n")
        cmd("git", "-C", str(self.repo), "add", "."); cmd("git", "-C", str(self.repo), "commit", "-m", "feature")
        self.feature = cmd("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip()
        cmd("git", "-C", str(self.repo), "checkout", "main")
        self.contract = {"schema_version": "1.0", "expected_baseline": self.base,
                         "feature_branch": "feature", "allowed_paths": ["feature.txt"],
                         "services": [], "backups": {"sqlite": [], "configuration": []},
                         "migration_checks": [], "pre_merge_tests": [], "post_merge_tests": [],
                         "readiness_checks": [], "acceptance_checklist": ["Observe behavior"],
                         "rollback": {"restore_git": True}}
        self.manifest = {"schema_version": "1.0", "task_id": "TEST", "exit_code": 0,
                         "final_codex_status": "PASS", "repository_path": str(self.repo),
                         "baseline_branch": "main", "baseline_head": self.base,
                         "origin_main_head": self.base, "feature_branch": "feature",
                         "feature_head": self.feature, "changed_files": ["feature.txt"],
                         "commits_created": [{"sha": self.feature, "subject": "feature"}], "tests": [],
                         "git_diff_check": {"passed": True}, "secret_scan": {"passed": True},
                         "deployment_contract": self.contract}
        self.package = root / "package"; self.package.mkdir()
        self.path = self.package / "result-manifest.json"; self.write()

    def tearDown(self): self.temp.cleanup()
    def write(self): self.path.write_text(json.dumps(self.manifest))
    def inspect(self): return integration.inspect_manifest(self.manifest, expected_repository=self.repo)

    def test_valid_manifest_and_inspect(self):
        _, loaded = integration.load_manifest(str(self.package)); self.assertEqual(loaded["task_id"], "TEST")
        self.assertTrue(self.inspect()["eligible"])

    def test_missing_and_malformed_manifest_rejected(self):
        with self.assertRaises(integration.IntegrationError): integration.load_manifest(str(Path(self.temp.name) / "missing"))
        self.path.write_text("{")
        with self.assertRaises(integration.IntegrationError): integration.load_manifest(str(self.path))

    def test_unexpected_repository_rejected(self):
        with self.assertRaisesRegex(integration.IntegrationError, "unexpected repository"):
            integration.inspect_manifest(self.manifest, expected_repository=Path(self.temp.name) / "other")

    def test_baseline_dirty_and_origin_guards(self):
        self.manifest["baseline_head"] = "0" * 40
        self.contract["expected_baseline"] = "0" * 40
        with self.assertRaisesRegex(integration.IntegrationError, "baseline HEAD mismatch"): self.inspect()
        self.manifest["baseline_head"] = self.base; self.contract["expected_baseline"] = self.base
        (self.repo / "dirty").write_text("x")
        with self.assertRaisesRegex(integration.IntegrationError, "dirty"): self.inspect()
        (self.repo / "dirty").unlink(); cmd("git", "-C", str(self.repo), "update-ref", "refs/remotes/origin/main", self.feature)
        with self.assertRaisesRegex(integration.IntegrationError, "origin-main"): self.inspect()

    def test_missing_and_wrong_feature_rejected(self):
        cmd("git", "-C", str(self.repo), "branch", "-D", "feature")
        with self.assertRaisesRegex(integration.IntegrationError, "missing"): self.inspect()

    def test_failed_status_test_diff_and_secret_rejected(self):
        for mutation, message in [
            (("final_codex_status", "FAIL"), "not successful"),
            (("git_diff_check", {"passed": False}), "diff check"),
            (("secret_scan", {"passed": False}), "secret scan"),
            (("tests", [{"command": "bad", "passed": False}]), "tests failed")]:
            original = self.manifest[mutation[0]]; self.manifest[mutation[0]] = mutation[1]
            with self.assertRaisesRegex(integration.IntegrationError, message): self.inspect()
            self.manifest[mutation[0]] = original

    def test_contract_and_path_guards(self):
        saved = self.manifest["deployment_contract"]; self.manifest["deployment_contract"] = None
        with self.assertRaisesRegex(integration.IntegrationError, "contract is missing"): self.inspect()
        self.manifest["deployment_contract"] = saved; self.manifest["changed_files"] = ["outside.txt"]
        with self.assertRaisesRegex(integration.IntegrationError, "outside contract"): self.inspect()

    def test_dry_run_makes_no_changes(self):
        before = cmd("git", "-C", str(self.repo), "status", "--porcelain=v1", "--branch").stdout
        result = cmd(sys.executable, str(SCRIPT), "dry-run", str(self.path), "--repository", str(self.repo))
        self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('"dry_run": true', result.stdout)
        self.assertEqual(before, cmd("git", "-C", str(self.repo), "status", "--porcelain=v1", "--branch").stdout)

    def test_stage_record_no_push_and_rollback(self):
        results = Path(self.temp.name) / "results"; backups = Path(self.temp.name) / "backups"
        result = cmd(sys.executable, str(SCRIPT), "stage", str(self.path), "--repository", str(self.repo),
                     "--result-root", str(results), "--backup-root", str(backups))
        self.assertEqual(result.returncode, 0, result.stderr)
        run_dir = next((results / "TEST").iterdir()); state = json.loads((run_dir / "integration-manifest.json").read_text())
        self.assertEqual(state["status"], "AWAITING_ACCEPTANCE")
        self.assertEqual(cmd("git", "--git-dir", str(self.remote), "rev-parse", "main").stdout.strip(), self.base)
        rollback = cmd(sys.executable, str(SCRIPT), "rollback", str(run_dir))
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        self.assertEqual(cmd("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip(), self.base)


class BackupTests(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_wal_online_backup_integrity_and_close(self):
        source = self.root / "live.sqlite"; backup = self.root / "protected" / "copy.sqlite"
        connection = sqlite3.connect(source); connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE private(value TEXT)"); connection.execute("INSERT INTO private VALUES ('synthetic')"); connection.commit()
        record = integration.sqlite_backup(source, backup)
        self.assertEqual(record["integrity_check"], "ok")
        self.assertEqual(sqlite3.connect(backup).execute("SELECT value FROM private").fetchone()[0], "synthetic")
        connection.close(); os.replace(source, self.root / "renamed.sqlite")

    def test_config_backup_preserves_mode_and_contents(self):
        source = self.root / "config"; source.write_text("synthetic\n"); source.chmod(0o640)
        destination = self.root / "backup" / "config"
        record = integration.config_backup(source, destination)
        self.assertEqual(destination.read_text(), "synthetic\n"); self.assertEqual(record["mode"], 0o640)

    def test_migration_disposable_twice_and_failure(self):
        source = self.root / "source.sqlite"; sqlite3.connect(source).close()
        backup = self.root / "backup.sqlite"; integration.sqlite_backup(source, backup)
        script = self.root / "migration.py"
        script.write_text("import os,sqlite3; c=sqlite3.connect(os.environ['DB']); c.execute('create table if not exists added(id integer)'); c.commit(); c.close()")
        contract = {"migration_checks": [{"database": str(source), "database_env": "DB",
                    "command": [sys.executable, str(script)], "idempotent": True,
                    "verification_sql": ["select * from added"]}]}
        result = integration.validate_migrations(contract, [{"kind": "sqlite", "source": str(source.resolve()), "backup": str(backup)}], self.root / "run")
        self.assertEqual(len(result[0]["attempts"]), 2)
        self.assertEqual(sqlite3.connect(source).execute("select count(*) from sqlite_master where name='added'").fetchone()[0], 0)
        contract["migration_checks"][0]["command"] = [sys.executable, "-c", "raise SystemExit(3)"]
        with self.assertRaisesRegex(integration.IntegrationError, "migration validation failed"):
            integration.validate_migrations(contract, [{"kind": "sqlite", "source": str(source.resolve()), "backup": str(backup)}], self.root / "failed")

    def test_redaction_and_bounding(self):
        text = "pass" + "word=" + "abcdefghijklmnop\n" + "x" * 5000
        self.assertNotIn("abcdefghijklmnop", integration.bounded(text, 100))
        artifact = self.root / "full.log"; record = integration.command_record([sys.executable, "-c", "print('x'*5000)"], artifact)
        self.assertGreater(len(artifact.read_text()), len(record["bounded_output"]))

    def test_allowlist(self):
        with self.assertRaisesRegex(integration.IntegrationError, "not allowlisted"):
            integration.service_action(["true"], "restart", "unknown.service")


if __name__ == "__main__": unittest.main()
