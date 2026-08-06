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
import os, pathlib, subprocess, sys
if sys.argv[1:3] == ['login', 'status']:
 print('Logged in using ChatGPT'); raise SystemExit(0)
if sys.argv[1:] == ['--version']:
 print('codex-cli test'); raise SystemExit(0)
args=sys.argv[1:]; work=pathlib.Path(args[args.index('-C')+1]); out=pathlib.Path(args[args.index('--output-last-message')+1])
prompt=sys.stdin.read(); (work/'agent.txt').write_text(prompt)
subprocess.run(['git','add','agent.txt'],cwd=work,check=True)
subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','agent work'],cwd=work,check=True,stdout=subprocess.DEVNULL)
out.write_text('final report')
print('fake stdout'); print('fake progress',file=sys.stderr)
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


if __name__ == "__main__":
    unittest.main()
