import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MailCourierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receiver = load_module("captured_receiver", "infra/mail/receiver.py")
        cls.sender = (ROOT / "infra/mail/send_file.py").read_text()

    def test_safe_filename_rules(self):
        valid = ["task.sh", "notes.md", "bundle.tar.gz", "data.JSON"]
        invalid = ["../task.sh", "/tmp/task.sh", ".hidden.sh", "bad.exe",
                   "dir\\task.sh", "x" * 181 + ".txt", ""]
        self.assertEqual(valid, [self.receiver.safe_filename(x) for x in valid])
        self.assertTrue(all(self.receiver.safe_filename(x) is None for x in invalid))

    def test_correlation_is_narrow(self):
        self.assertIn("CORR_RE.fullmatch(corr)", self.sender)
        self.assertIn("Purpose: file transport only", self.sender)
        self.assertIn('"automatic_execution": False', (ROOT / "infra/mail/receiver.py").read_text())

    def test_sanitized_mail_config_schema(self):
        data = json.loads((ROOT / "infra/mail/kven-mail-courier.example.json").read_text())
        self.assertEqual({
            "EMAIL_USER", "EMAIL_PASS", "IMAP_HOST", "IMAP_PORT", "SMTP_HOST",
            "SMTP_PORT", "accepted_sender", "outbound_recipient", "inbox_dir",
            "max_attachment_bytes", "poll_seconds",
        }, set(data))
        self.assertIn("REPLACE_", data["EMAIL_PASS"])


class QweniiAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("captured_runner", "infra/qwenii/runner.py")

    def fixture(self, root, corr="TASK-0001", digest_override=None, extra=None):
        inbox = root / "inbox"
        state = root / "state"
        inbox.mkdir()
        payload = inbox / f"KVEN-QWENII-PAYLOAD-{corr}.sh"
        payload.write_bytes(b"#!/bin/bash\nprintf ok\n")
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        auth = inbox / f"KVEN-QWENII-AUTH-{corr}.json"
        env = {
            "protocol": "KVEN-QWENII-TASK/1",
            "authorization": "EXECUTE_AS_QWENII",
            "task_id": corr,
            "correlation_id": corr,
            "payload": {"filename": payload.name, "sha256": digest_override or digest},
            "timeout_seconds": 30,
        }
        if extra:
            env.update(extra)
        auth.write_text(json.dumps(env))
        cfg = {
            "accepted_sender": "sender@example.invalid", "inbox_dir": str(inbox),
            "state_dir": str(state), "max_timeout_seconds": 60,
        }
        for target in (payload, auth):
            rec = {
                "correlation_id": corr, "sender": cfg["accepted_sender"],
                "subject": f"[KVEN-BRIDGE] FILE {corr}", "message_id": "<sanitized@example.invalid>",
                "filename": target.name, "path": str(target), "size": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "saved_mode": "0644", "automatic_execution": False,
            }
            Path(str(target) + ".receipt.json").write_text(json.dumps(rec))
        return cfg, auth, payload

    def test_authorization_binds_correlation_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, auth, payload = self.fixture(Path(tmp))
            rec = {"correlation_id": "TASK-0001"}
            self.assertEqual(payload, self.runner.validate_envelope(cfg, auth, "TASK-0001", rec))
            self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), rec["payload_sha256"])
            self.assertEqual("VALIDATED", rec["state"])

    def test_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, auth, _ = self.fixture(Path(tmp), digest_override="0" * 64)
            with self.assertRaisesRegex(ValueError, "envelope_payload_hash"):
                self.runner.validate_envelope(cfg, auth, "TASK-0001", {"correlation_id": "TASK-0001"})

    def test_malformed_or_generic_command_envelope_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, auth, _ = self.fixture(Path(tmp), extra={"COMMAND": "anything"})
            with self.assertRaisesRegex(ValueError, "envelope_schema"):
                self.runner.validate_envelope(cfg, auth, "TASK-0001", {"correlation_id": "TASK-0001"})

    def test_returned_state_is_at_most_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, auth, _ = self.fixture(root)
            state = Path(cfg["state_dir"]) / "tasks"
            state.mkdir(parents=True)
            (state / "TASK-0001.json").write_text(json.dumps(
                {"correlation_id": "TASK-0001", "state": "RETURNED", "execution_count": 1}
            ))
            with mock.patch.object(self.runner, "run_task") as run:
                self.runner.process_auth(cfg, auth)
            run.assert_not_called()

    def test_sanitized_runner_config_schema(self):
        data = json.loads((ROOT / "infra/qwenii/kven-qwenii-runner.example.json").read_text())
        source = (ROOT / "infra/qwenii/runner.py").read_text()
        for key in data:
            self.assertIn(key, source)
        self.assertNotIn("COMMAND", source)


class NotifierAndUnitTests(unittest.TestCase):
    def test_notifier_vocabulary_and_request_validation(self):
        allowed = {"NEED_USER", "WORK_FAILED", "WORK_COMPLETE"}
        request = (ROOT / "infra/notify/kven-request-human").read_text()
        dispatch = (ROOT / "infra/notify/kven-human-notify-dispatch").read_text()
        notifier = (ROOT / "infra/notify/kven-human-notifier").read_text()
        for event in allowed:
            self.assertIn(event, request)
            self.assertIn(event, dispatch)
            self.assertIn(event, notifier)
        self.assertIn('^[A-Za-z0-9._:-]{1,80}$', request)
        self.assertIn('"$size" -gt 256', dispatch)
        self.assertIn('"$owner" != "qwenii"', dispatch)
        self.assertIn("duplicate_suppressed", dispatch)

    def test_units_preserve_boundaries(self):
        courier = (ROOT / "infra/mail/systemd/kven-mail-courier.service").read_text()
        runner = (ROOT / "infra/qwenii/systemd/kven-qwenii-runner.service").read_text()
        path = (ROOT / "infra/notify/systemd/kven-human-notify.path").read_text()
        self.assertIn("transport only, no execution", courier)
        self.assertIn("ReadWritePaths=/var/lib/kven-mail-courier /home/qwenii/inbox", courier)
        self.assertIn("ReadOnlyPaths=", runner)
        self.assertIn("/opt/kven2 /agent/data", runner)
        self.assertIn("PathExistsGlob=/home/qwenii/notify/*.req", path)

    def test_notifier_example_is_sanitized(self):
        text = (ROOT / "infra/notify/kven-human-notifier.example.conf").read_text()
        self.assertIn("REPLACE_FROM_PROTECTED_CONFIG", text)
        self.assertNotIn("192.168.1.110", text)


if __name__ == "__main__":
    unittest.main()
