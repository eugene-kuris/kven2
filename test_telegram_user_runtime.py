import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from telegram_user_runtime import (
    PrivateMessageEvidence,
    TelegramUserConfig,
    TelegramUserRuntime,
    load_config,
    normalize_private_message,
    prepare_session_directory,
    send_explicit_private_text,
)
from telegram_user_auth import authorize
from telegram_user_control import request_send


class FakeClient:
    def __init__(self, *, authorized=True, send_error=None):
        self.authorized = authorized
        self.send_error = send_error
        self.connected = False
        self.connect_count = 0
        self.disconnected = False
        self.start_calls = []
        self.handlers = []

    async def connect(self):
        self.connected = True
        self.connect_count += 1
    async def disconnect(self): self.disconnected = True
    async def is_user_authorized(self): return self.authorized
    async def get_me(self): return types.SimpleNamespace(id=777)
    def add_event_handler(self, handler, builder): self.handlers.append((handler, builder))
    async def start(self, **kwargs):
        self.start_calls.append(kwargs)
        kwargs["code_callback"]()
        kwargs["password"]()
        self.authorized = True
    async def send_message(self, peer_id, text):
        if self.send_error: raise self.send_error
        return types.SimpleNamespace(id=91, date=datetime(2026, 8, 8, tzinfo=timezone.utc))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = {
            "TELEGRAM_USER_API_ID": "12345",
            "TELEGRAM_USER_API_HASH": "test-api-hash-not-secret",
            "TELEGRAM_USER_SESSION_PATH": str(self.root / "kven.session"),
            "TELEGRAM_USER_CONTROL_SOCKET": str(self.root / "control.sock"),
        }
    def tearDown(self): self.temp.cleanup()

    def test_required_config_and_secret_repr_redaction(self):
        config = load_config(self.env, session_root=self.root)
        self.assertEqual(config.api_id, 12345)
        self.assertNotIn(self.env["TELEGRAM_USER_API_HASH"], repr(config))
        self.assertNotIn("12345", repr(config))

    def test_missing_and_invalid_api_values_are_rejected_without_secret(self):
        for env in ({}, {**self.env, "TELEGRAM_USER_API_ID": "x"}, {**self.env, "TELEGRAM_USER_API_ID": "0"}, {**self.env, "TELEGRAM_USER_API_HASH": " "}):
            with self.subTest(env=sorted(env)):
                with self.assertRaises(ValueError) as raised:
                    load_config(env, session_root=self.root)
                self.assertNotIn(self.env["TELEGRAM_USER_API_HASH"], str(raised.exception))

    def test_session_must_be_dedicated_absolute_session_path(self):
        for value in ("relative.session", str(self.root.parent / "escape.session"), str(self.root / "bad.db")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    load_config({**self.env, "TELEGRAM_USER_SESSION_PATH": value}, session_root=self.root)

    def test_session_directory_is_created_mode_0700(self):
        nested = self.root / "state" / "kven.session"
        prepare_session_directory(nested, session_root=self.root)
        self.assertEqual(nested.parent.stat().st_mode & 0o777, 0o700)


class EventAndSendTests(unittest.IsolatedAsyncioTestCase):
    def test_private_message_normalization_and_unsupported_events(self):
        message = types.SimpleNamespace(id=44, date=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc), message="marker")
        event = types.SimpleNamespace(is_private=True, out=False, sender_id=123, chat_id=123, message=message)
        evidence = normalize_private_message(event, 777)
        self.assertIsInstance(evidence, PrivateMessageEvidence)
        self.assertEqual((evidence.own_user_id, evidence.sender_user_id, evidence.peer_id, evidence.message_id), (777, 123, 123, 44))
        self.assertEqual(evidence.text_length, 6)
        self.assertEqual(len(evidence.text_sha256), 64)
        for unsupported in (types.SimpleNamespace(is_private=False), types.SimpleNamespace(is_private=True, out=True)):
            self.assertIsNone(normalize_private_message(unsupported, 777))

    async def test_explicit_send_success_and_failure(self):
        result = await send_explicit_private_text(FakeClient(), 123, "one message")
        self.assertTrue(result["ok"])
        self.assertEqual(result["message_id"], 91)
        with self.assertRaises(RuntimeError):
            await send_explicit_private_text(FakeClient(send_error=RuntimeError("network detail")), 123, "one")

    async def test_control_command_requests_exactly_one_send(self):
        requests = []
        async def handler(reader, writer):
            requests.append((await reader.readline()).decode())
            writer.write(b'{"ok": true, "message_id": 8}\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        with tempfile.TemporaryDirectory() as directory:
            socket = Path(directory) / "control.sock"
            server = await asyncio.start_unix_server(handler, path=str(socket))
            try:
                response = await request_send(123, "exact", socket_path=socket)
            finally:
                server.close()
                await server.wait_closed()
        self.assertEqual(response["message_id"], 8)
        self.assertEqual(len(requests), 1)
        self.assertIn('"peer_id": 123', requests[0])

    async def test_receive_observer_never_sends_or_calls_inference(self):
        client = FakeClient()
        config = TelegramUserConfig(1, "hash", Path("/unused/kven.session"), Path("/tmp/unused.sock"))
        runtime = TelegramUserRuntime(config, client_factory=lambda _: client, session_preparer=lambda _: None, event_builder_factory=lambda: "private-events")
        runtime.own_user_id = 777
        event = types.SimpleNamespace(is_private=True, out=False, sender_id=123, chat_id=123, message=types.SimpleNamespace(id=1, date=datetime.now(timezone.utc), message="private"))
        await runtime._observe(event)
        self.assertFalse(hasattr(client, "kven_client"))


class LifecycleAndAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_connects_serves_and_disconnects_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeClient()
            config = TelegramUserConfig(1, "hash", root / "kven.session", root / "control.sock")
            runtime = TelegramUserRuntime(config, client_factory=lambda _: client, session_preparer=lambda _: None, event_builder_factory=lambda: "private-events")
            stop = asyncio.Event()
            task = asyncio.create_task(runtime.run(stop))
            for _ in range(100):
                if config.control_socket.exists(): break
                await asyncio.sleep(0.001)
            self.assertTrue(client.connected)
            stop.set()
            await task
            self.assertTrue(client.disconnected)
            self.assertFalse(config.control_socket.exists())

    async def test_unauthorized_runtime_fails_closed_and_disconnects(self):
        client = FakeClient(authorized=False)
        config = TelegramUserConfig(1, "hash", Path("/unused/kven.session"), Path("/tmp/unused.sock"))
        runtime = TelegramUserRuntime(config, client_factory=lambda _: client, session_preparer=lambda _: None, event_builder_factory=lambda: "private-events")
        with self.assertRaisesRegex(RuntimeError, "HUMAN_REQUIRED"):
            await runtime.run(asyncio.Event())
        self.assertTrue(client.disconnected)

    async def test_restart_reconnects_same_client_session_without_auth_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeClient()
            config = TelegramUserConfig(1, "hash", root / "kven.session", root / "control.sock")
            for _ in range(2):
                runtime = TelegramUserRuntime(config, client_factory=lambda _: client, session_preparer=lambda _: None, event_builder_factory=lambda: "private-events")
                stop = asyncio.Event()
                stop.set()
                await runtime.run(stop)
            self.assertEqual(client.connect_count, 2)
            self.assertEqual(client.start_calls, [])

    async def test_authorized_bootstrap_never_prompts(self):
        client = FakeClient()
        config = TelegramUserConfig(1, "hash", Path("/unused/kven.session"))
        with mock.patch("telegram_user_auth.prepare_session_directory"):
            own_id = await authorize(config=config, client_factory=lambda _: client, input_fn=lambda _: self.fail("prompted"))
        self.assertEqual(own_id, 777)
        self.assertEqual(client.start_calls, [])

    async def test_unauthorized_bootstrap_uses_ephemeral_callbacks(self):
        client = FakeClient(authorized=False)
        prompts = iter(["+10000000000", "12345"])
        passwords = []
        config = TelegramUserConfig(1, "hash", Path("/unused/kven.session"))
        with mock.patch("telegram_user_auth.prepare_session_directory"):
            own_id = await authorize(config=config, client_factory=lambda _: client, input_fn=lambda _: next(prompts), password_fn=lambda _: passwords.append("asked") or "password")
        self.assertEqual(own_id, 777)
        self.assertEqual(passwords, ["asked"])
        self.assertTrue(client.disconnected)


if __name__ == "__main__":
    unittest.main()
