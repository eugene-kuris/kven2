import importlib.machinery
import importlib.util
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import json


SCRIPT = Path(__file__).parent / "scripts" / "kven-codex-task"
loader = importlib.machinery.SourceFileLoader("kven_codex_task", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
runner = importlib.util.module_from_spec(spec)
loader.exec_module(runner)


def command(*args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class UnitTests(unittest.TestCase):
    def test_exact_task_id_line(self):
        self.assertEqual(runner.task_id_from("x\nTASK ID: ABC-12\ny", "a.txt"), "ABC-12")
        self.assertEqual(runner.task_id_from(" Task ID: wrong", "named.task"), "named")

    def test_safe_sanitization(self):
        self.assertEqual(runner.sanitize_task_id(" ../Hello__WORLD! "), "hello-world")
        self.assertEqual(runner.sanitize_task_id("///"), "task")

    def test_file_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.txt"
            path.write_text("do it", encoding="utf-8")
            self.assertEqual(runner.read_task(str(path)), "do it")

    def test_stdin_input(self):
        self.assertEqual(runner.read_task("-", io.StringIO("stdin task")), "stdin task")

    def test_empty_task_rejected(self):
        with self.assertRaises(runner.RunnerError):
            runner.read_task("-", io.StringIO(" \n"))

    def test_exact_permission_arguments_and_model(self):
        cmd = runner.make_codex_command("codex", Path("/w"), Path("/o"), "chosen")
        self.assertEqual(cmd[:5], ["codex", "--sandbox", "danger-full-access", "--ask-for-approval", "never"])
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[-3:], ["--output-last-message", "/o", "-"])

    def test_model_is_not_implicitly_forwarded(self):
        self.assertNotIn("--model", runner.make_codex_command("codex", Path("/w"), Path("/o"), None))

    def test_default_network_prohibition(self):
        self.assertIn("Network access is prohibited", runner.execution_contract("task", False))

    def test_allow_network_is_task_limited(self):
        text = runner.execution_contract("task", True)
        self.assertIn("only where the supplied task explicitly requires it", text)

    def test_secret_pattern_detection(self):
        candidate = "api_" + "key=" + ("a" * 16)
        self.assertIsNotNone(runner.SECRET_PATTERN.search(candidate))
        self.assertIsNone(runner.SECRET_PATTERN.search("api_key=placeholder"))

    def test_empty_tests_make_summary_ineligible(self):
        manifest = {
            "task_id": "TEST", "final_codex_status": "PASS", "deployment_contract": {},
            "tests": [], "commits_created": [], "changed_files": [], "result_package_path": "/tmp/result",
            "deployment_contract_error": None,
        }
        self.assertIn("Eligible for integration: **no**", runner.make_result_summary(manifest))
        self.assertIn("/opt/kven2/scripts/kven-integrate-task inspect", runner.make_result_summary(manifest))

    def test_command_argv_secrets_are_redacted(self):
        command = ["tool", "--" + "token", "synthetic-secret-value", "api_" + "key=" + "another-secret-value"]
        redacted = runner.redact_argv(command)
        self.assertNotIn("synthetic-secret-value", json.dumps(redacted))
        self.assertNotIn("another-secret-value", json.dumps(redacted))

    def test_validation_evidence_directory_is_retry_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {"result_validation_tests": [{
                "name": "retry-safe", "command": [sys.executable, "-c", "pass"], "timeout": 30,
            }]}
            first = runner.run_validation_tests(contract, root, root / "result")
            second = runner.run_validation_tests(contract, root, root / "result")
            self.assertTrue(first[0]["passed"])
            self.assertTrue(second[0]["passed"])

    def test_runtime_model_reasoning_and_token_header_parsing(self):
        stderr = """OpenAI Codex v0.test
--------
workdir: /fixture
model: gpt-fixture
provider: openai
reasoning effort: high
session id: fixture-session
--------
progress
tokens used
12,345
"""
        parsed = runner.parse_runtime_metadata("model: stdout-spoof\n", stderr)
        self.assertEqual(parsed, {
            "actual_model": "gpt-fixture", "actual_provider": "openai",
            "actual_reasoning_effort": "high", "session_id": "fixture-session", "token_usage": 12345,
        })

    def test_runtime_metadata_ignores_later_spoof_and_requires_unique_header(self):
        header = """OpenAI Codex v0.test
--------
model: trusted-model
provider: openai
reasoning effort: high
session id: fixture-session
--------
model: later-spoof
tokens used
9
"""
        self.assertEqual(runner.parse_runtime_metadata("model: stdout-spoof", header)["actual_model"], "trusted-model")
        conflict = header.replace("model: trusted-model", "model: trusted-model\nmodel: conflict")
        with self.assertRaisesRegex(runner.RunnerError, "duplicated or conflicting"):
            runner.parse_runtime_metadata("", conflict)
        self.assertIsNone(runner.parse_runtime_metadata("", header + "later progress\n")["token_usage"])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.repo = root / "main"
        self.worktrees = root / "worktrees"
        self.results = root / "results"
        self.bin = root / "bin"
        self.bin.mkdir()
        command("git", "init", "-b", "main", str(self.repo))
        command("git", "-C", str(self.repo), "config", "user.email", "test@example.invalid")
        command("git", "-C", str(self.repo), "config", "user.name", "Test")
        (self.repo / "base.txt").write_text("base\n", encoding="utf-8")
        command("git", "-C", str(self.repo), "add", "base.txt")
        command("git", "-C", str(self.repo), "commit", "-m", "baseline")
        head = command("git", "-C", str(self.repo), "rev-parse", "HEAD").stdout.strip()
        command("git", "-C", str(self.repo), "update-ref", "refs/remotes/origin/main", head)
        fake = self.bin / "codex"
        fake.write_text("""#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
if sys.argv[1:3] == ['login', 'status']:
 print('Logged in using ChatGPT'); raise SystemExit(0)
if sys.argv[1:] == ['--version']:
 print('codex-cli test'); raise SystemExit(0)
args=sys.argv[1:]; work=pathlib.Path(args[args.index('-C')+1]); out=pathlib.Path(args[args.index('--output-last-message')+1])
prompt=sys.stdin.read(); (work/'agent.txt').write_text(prompt)
mutation='''import os,pathlib,subprocess
mode=os.environ.get('VALIDATION_MUTATION') or ('untracked' if os.environ.get('VALIDATION_DIRTY') else '')
if mode:
 p=pathlib.Path('validation-'+mode); p.write_text('preserve')
 if mode in {'stage','commit'}: subprocess.run(['git','add',str(p)],check=True)
 if mode=='commit': subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','validation mutation'],check=True,stdout=subprocess.DEVNULL)
raise SystemExit(int(os.environ.get('VALIDATION_EXIT','0')))
'''
base=subprocess.run(['git','-C',str(work),'rev-parse','HEAD'],text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
branch=subprocess.run(['git','-C',str(work),'branch','--show-current'],text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
check={'name':'synthetic-validation','command':[sys.executable,'-c',mutation],'timeout':30}
contract={'schema_version':'2.0','expected_baseline':base,'feature_branch':branch,
 'allowed_paths':['agent.txt','deployment-contract.json'],'services':[],
 'backups':{'sqlite':[],'configuration':[]},'migration_checks':[],
 'result_validation_tests':[check],'pre_merge_tests':[dict(check)],'post_merge_tests':[dict(check)],
 'readiness_checks':[],'fatal_log_checks':[],'acceptance_checklist':['Synthetic acceptance'],
 'rollback':{'restore_git':True,'restore_backups':True,'verify_readiness':True}}
if os.environ.get('CONTRACT_LITERAL'): contract['result_validation_tests'][0]['command'] += ['--'+'token', os.environ['CONTRACT_LITERAL']]
(work/'deployment-contract.json').write_text(json.dumps(contract))
subprocess.run(['git','add','agent.txt','deployment-contract.json'],cwd=work,check=True)
subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','agent work'],cwd=work,check=True,stdout=subprocess.DEVNULL)
out.write_text('final report --'+'token '+os.environ['CONTRACT_LITERAL'] if os.environ.get('CONTRACT_LITERAL') else 'final report')
model=os.environ.get('ACTUAL_MODEL') or (args[args.index('--model')+1] if '--model' in args else 'default-fixture-model')
print('fake stdout'+((' --'+'token '+os.environ['CONTRACT_LITERAL']) if os.environ.get('CONTRACT_LITERAL') else ''))
print('OpenAI Codex v0.test',file=sys.stderr); print('--------',file=sys.stderr)
print('workdir: '+str(work),file=sys.stderr); print('model: '+model,file=sys.stderr); print('provider: openai',file=sys.stderr)
print('reasoning effort: high',file=sys.stderr); print('session id: fixture-session',file=sys.stderr); print('--------',file=sys.stderr)
print('fake progress',file=sys.stderr); print('tokens used',file=sys.stderr); print('1234',file=sys.stderr)
raise SystemExit(int(os.environ.get('FAKE_CODEX_EXIT','0')))
""", encoding="utf-8")
        fake.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update({
            "PATH": str(self.bin) + os.pathsep + self.env["PATH"],
            runner.ENV_REPOSITORY: str(self.repo),
            runner.ENV_RESULT_ROOT: str(self.results),
            runner.ENV_WORKTREE_ROOT: str(self.worktrees),
            "PYTHONDONTWRITEBYTECODE": "1",
        })

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *extra, task="TASK ID: DEMO\nDo work\n", env=None):
        task_file = Path(self.temp.name) / "task.txt"
        task_file.write_text(task, encoding="utf-8")
        return command(sys.executable, "-B", str(SCRIPT), *extra, str(task_file), env=env or self.env)

    def test_success_package_permissions_names_and_preservation(self):
        result = self.invoke("--model", "unit-model")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        dirs = list(self.results.glob("demo-*"))
        self.assertEqual(len(dirs), 1)
        package = dirs[0]
        self.assertTrue((package / "final-response.txt").is_file())
        self.assertTrue((package / "service-state.json").is_file())
        self.assertTrue((package / "result-manifest.json").is_file())
        self.assertTrue((package / "result-summary.md").is_file())
        manifest = json.loads((package / "result-manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], "2.0")
        self.assertEqual(manifest["final_codex_status"], "PASS")
        self.assertEqual(len(manifest["tests"]), 1)
        record = manifest["tests"][0]
        self.assertTrue(record["passed"])
        self.assertIsInstance(record["command"], list)
        self.assertIn("started_at", record)
        self.assertIn("finished_at", record)
        self.assertIn("duration_seconds", record)
        self.assertEqual(record["exit_code"], 0)
        self.assertFalse(Path(record["output_artifact"]).is_absolute())
        self.assertTrue((package / record["output_artifact"]).is_file())
        self.assertEqual(record["artifact_size"], (package / record["output_artifact"]).stat().st_size)
        self.assertEqual(manifest["requested_model"], "unit-model")
        self.assertEqual(manifest["actual_runtime_model"], "unit-model")
        self.assertEqual(manifest["actual_reasoning_effort"], "high")
        self.assertEqual(manifest["token_usage"], 1234)
        self.assertFalse(manifest["network_use"]["used"])
        self.assertEqual(manifest["evidence_provenance"]["method"], "automatic_runner")
        self.assertEqual(stat.S_IMODE(package.stat().st_mode), 0o755)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o644 for p in package.iterdir() if p.is_file()))
        branches = command("git", "-C", str(self.repo), "branch", "--format=%(refname:short)").stdout
        self.assertIn("codex/demo-", branches)
        self.assertEqual(len(list(self.worktrees.glob("demo-*"))), 1)
        prompt = next(self.worktrees.glob("demo-*")) / "agent.txt"
        self.assertIn("Network access is prohibited", prompt.read_text(encoding="utf-8"))

    def test_dirty_main_rejected(self):
        (self.repo / "dirty.txt").write_text("dirty", encoding="utf-8")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("main worktree is dirty", result.stderr)

    def test_origin_main_mismatch_rejected(self):
        command("git", "-C", str(self.repo), "commit", "--allow-empty", "-m", "main moved")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not equal", result.stderr)

    def test_failure_exit_propagates_and_preserves_evidence(self):
        env = self.env.copy(); env["FAKE_CODEX_EXIT"] = "7"
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 7)
        package = next(self.results.glob("demo-*"))
        self.assertEqual((package / "exit-code.txt").read_text().strip(), "7")
        self.assertEqual(len(list(self.worktrees.glob("demo-*"))), 1)

    def test_failed_runner_validation_sets_fail_status(self):
        env = self.env.copy(); env["VALIDATION_EXIT"] = "9"
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 1)
        package = next(self.results.glob("demo-*"))
        manifest = json.loads((package / "result-manifest.json").read_text())
        self.assertEqual(manifest["final_codex_status"], "FAIL")
        self.assertFalse(manifest["tests"][0]["passed"])
        self.assertEqual(manifest["tests"][0]["exit_code"], 9)

    def test_runner_rechecks_worktree_after_validation_command(self):
        env = self.env.copy(); env["VALIDATION_DIRTY"] = "1"
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 1)
        package = next(self.results.glob("demo-*"))
        manifest = json.loads((package / "result-manifest.json").read_text())
        self.assertEqual(manifest["final_codex_status"], "FAIL")
        self.assertIn("feature-worktree-not-clean", (package / "summary.txt").read_text())

    def test_validation_command_commit_does_not_replace_tested_feature_head(self):
        env = self.env.copy(); env["VALIDATION_MUTATION"] = "commit"
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 1)
        package = next(self.results.glob("demo-*"))
        manifest = json.loads((package / "result-manifest.json").read_text())
        worktree = next(self.worktrees.glob("demo-*"))
        actual = command("git", "-C", str(worktree), "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(manifest["final_codex_status"], "FAIL")
        self.assertNotEqual(manifest["feature_head"], actual)
        self.assertIn("feature-state-changed-during-validation", (package / "summary.txt").read_text())

    def test_validation_command_stages_file_and_runner_fails(self):
        env = self.env.copy(); env["VALIDATION_MUTATION"] = "stage"
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 1)
        package = next(self.results.glob("demo-*"))
        manifest = json.loads((package / "result-manifest.json").read_text())
        self.assertEqual(manifest["final_codex_status"], "FAIL")
        worktree = next(self.worktrees.glob("demo-*"))
        self.assertIn("validation-stage", command("git", "-C", str(worktree), "diff", "--cached", "--name-only").stdout)

    def test_validation_command_untracked_data_is_preserved_on_failure(self):
        env = self.env.copy(); env["VALIDATION_MUTATION"] = "untracked"
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 1)
        worktree = next(self.worktrees.glob("demo-*"))
        self.assertEqual((worktree / "validation-untracked").read_text(), "preserve")

    def test_requested_actual_model_mismatch_fails(self):
        env = self.env.copy(); env["ACTUAL_MODEL"] = "different-runtime-model"
        result = self.invoke("--model", "requested-model", env=env)
        self.assertEqual(result.returncode, 1)
        manifest = json.loads((next(self.results.glob("demo-*")) / "result-manifest.json").read_text())
        self.assertEqual(manifest["requested_model"], "requested-model")
        self.assertEqual(manifest["actual_runtime_model"], "different-runtime-model")
        self.assertEqual(manifest["final_codex_status"], "FAIL")

    def test_no_requested_model_still_records_actual_runtime_model(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0)
        manifest = json.loads((next(self.results.glob("demo-*")) / "result-manifest.json").read_text())
        self.assertIsNone(manifest["requested_model"])
        self.assertEqual(manifest["actual_runtime_model"], "default-fixture-model")

    def test_literal_contract_credential_never_enters_result_evidence(self):
        literal = "literal-" + "sensitive-value-123"
        env = self.env.copy(); env["CONTRACT_LITERAL"] = literal
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 1)
        package = next(self.results.glob("demo-*"))
        manifest = json.loads((package / "result-manifest.json").read_text())
        self.assertIsNone(manifest["deployment_contract"])
        for path in package.rglob("*"):
            if path.is_file():
                self.assertNotIn(literal, path.read_text(encoding="utf-8", errors="replace"), str(path))


if __name__ == "__main__":
    unittest.main()
