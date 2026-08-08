"""Telegram MTProto user-account transport without Kven inference integration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import stat
from typing import Any


DEFAULT_SESSION_ROOT = Path("/agent/data/kven2/telegram_user")
DEFAULT_SESSION_PATH = DEFAULT_SESSION_ROOT / "kven.session"
DEFAULT_CONTROL_ROOT = Path("/run/kven2-telegram-user")
DEFAULT_CONTROL_SOCKET = DEFAULT_CONTROL_ROOT / "control.sock"
DEFAULT_LOG_LEVEL = "INFO"
MAX_TELEGRAM_PEER_ID = 2**63 - 1
MAX_SEND_TEXT_LENGTH = 4096
MAX_CONTROL_REQUEST_BYTES = 16_384
LOGGER = logging.getLogger("kven.telegram_user")


@dataclass(frozen=True, repr=False)
class TelegramUserConfig:
    api_id: int
    api_hash: str
    session_path: Path = DEFAULT_SESSION_PATH
    control_socket: Path = DEFAULT_CONTROL_SOCKET
    log_level: str = DEFAULT_LOG_LEVEL

    def __repr__(self) -> str:
        return (
            "TelegramUserConfig(api_id=<redacted>, api_hash=<redacted>, "
            f"session_path={str(self.session_path)!r}, "
            f"control_socket={str(self.control_socket)!r}, log_level={self.log_level!r})"
        )


@dataclass(frozen=True)
class PrivateMessageEvidence:
    transport: str
    direction: str
    own_user_id: int
    sender_user_id: int
    peer_id: int
    message_id: int
    timestamp: str
    text_length: int
    text_sha256: str


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"Required environment variable is missing or blank: {name}")
    return str(value).strip()


def validate_session_path(path: Path, *, session_root: Path = DEFAULT_SESSION_ROOT) -> Path:
    candidate = path if path.is_absolute() else Path()
    try:
        resolved_parent = candidate.parent.resolve(strict=False)
        resolved_root = session_root.resolve(strict=False)
        resolved_parent.relative_to(resolved_root)
    except (OSError, ValueError):
        raise ValueError("TELEGRAM_USER_SESSION_PATH must be an absolute .session path under the dedicated session root") from None
    if candidate.suffix != ".session" or candidate.name in {".session", ""}:
        raise ValueError("TELEGRAM_USER_SESSION_PATH must end in .session")
    return candidate


def validate_control_socket(path: Path, *, control_root: Path = DEFAULT_CONTROL_ROOT) -> Path:
    candidate = path if path.is_absolute() else Path()
    try:
        candidate.parent.resolve(strict=False).relative_to(control_root.resolve(strict=False))
    except (OSError, ValueError):
        raise ValueError(
            "TELEGRAM_USER_CONTROL_SOCKET must be an absolute socket path under the dedicated runtime directory"
        ) from None
    if candidate.name in {"", ".", ".."}:
        raise ValueError("TELEGRAM_USER_CONTROL_SOCKET must name a socket")
    return candidate


def load_config(
    environ: Mapping[str, str] | None = None,
    *,
    session_root: Path = DEFAULT_SESSION_ROOT,
    control_root: Path = DEFAULT_CONTROL_ROOT,
) -> TelegramUserConfig:
    environment = os.environ if environ is None else environ
    api_id_raw = _required(environment, "TELEGRAM_USER_API_ID")
    api_hash = _required(environment, "TELEGRAM_USER_API_HASH")
    try:
        api_id = int(api_id_raw)
    except ValueError:
        raise ValueError("TELEGRAM_USER_API_ID must be a positive integer") from None
    if api_id <= 0:
        raise ValueError("TELEGRAM_USER_API_ID must be a positive integer")
    session_path = validate_session_path(
        Path(environment.get("TELEGRAM_USER_SESSION_PATH", str(DEFAULT_SESSION_PATH))),
        session_root=session_root,
    )
    control_socket = validate_control_socket(
        Path(environment.get("TELEGRAM_USER_CONTROL_SOCKET", str(DEFAULT_CONTROL_SOCKET))),
        control_root=control_root,
    )
    log_level = environment.get("TELEGRAM_USER_LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("TELEGRAM_USER_LOG_LEVEL is invalid")
    return TelegramUserConfig(api_id, api_hash, session_path, control_socket, log_level)


def prepare_session_directory(path: Path, *, session_root: Path = DEFAULT_SESSION_ROOT) -> None:
    validate_session_path(path, session_root=session_root)
    if path.is_symlink():
        raise ValueError("Telegram user session file must not be a symlink")
    parent = path.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError("Telegram user session parent must be a real directory")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    mode = stat.S_IMODE(parent.stat().st_mode)
    if mode != 0o700:
        raise PermissionError("Telegram user session directory must have mode 0700")


def default_client_factory(config: TelegramUserConfig) -> Any:
    try:
        from telethon import TelegramClient
    except ImportError as exc:
        raise RuntimeError("Telethon 1.44.0 is required for the Telegram user-account runtime") from exc
    return TelegramClient(str(config.session_path), config.api_id, config.api_hash)


def default_event_builder() -> Any:
    from telethon import events

    return events.NewMessage(incoming=True)


def normalize_private_message(event: Any, own_user_id: int) -> PrivateMessageEvidence | None:
    if not bool(getattr(event, "is_private", False)) or bool(getattr(event, "out", False)):
        return None
    sender_id = getattr(event, "sender_id", None)
    peer_id = getattr(event, "chat_id", None)
    message = getattr(event, "message", event)
    message_id = getattr(message, "id", None)
    timestamp = getattr(message, "date", None)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (own_user_id, sender_id, peer_id, message_id)):
        return None
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    text = getattr(message, "message", "") or ""
    if not isinstance(text, str):
        text = ""
    return PrivateMessageEvidence(
        transport="telegram_mtproto_user",
        direction="incoming_private",
        own_user_id=own_user_id,
        sender_user_id=sender_id,
        peer_id=peer_id,
        message_id=message_id,
        timestamp=timestamp.astimezone(timezone.utc).isoformat(),
        text_length=len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def validate_peer_id(value: Any, field: str = "peer_id") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer in the signed 64-bit positive range")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer in the signed 64-bit positive range") from None
    if result < 1 or result > MAX_TELEGRAM_PEER_ID or str(result) != str(value).strip():
        raise ValueError(f"{field} must be an integer in the signed 64-bit positive range")
    return result


async def send_explicit_private_text(client: Any, peer_id: Any, text: Any) -> dict[str, Any]:
    numeric_peer = validate_peer_id(peer_id)
    if not isinstance(text, str) or not text or len(text) > MAX_SEND_TEXT_LENGTH:
        raise ValueError(f"text must contain 1..{MAX_SEND_TEXT_LENGTH} characters")
    message = await client.send_message(numeric_peer, text)
    timestamp = getattr(message, "date", None)
    return {
        "ok": True,
        "transport": "telegram_mtproto_user",
        "direction": "outgoing_private",
        "peer_id": numeric_peer,
        "message_id": int(message.id),
        "timestamp": timestamp.astimezone(timezone.utc).isoformat() if isinstance(timestamp, datetime) else None,
        "text_length": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class TelegramUserRuntime:
    def __init__(
        self,
        config: TelegramUserConfig,
        *,
        client_factory: Callable[[TelegramUserConfig], Any] = default_client_factory,
        session_preparer: Callable[[Path], None] = prepare_session_directory,
        event_builder_factory: Callable[[], Any] = default_event_builder,
    ):
        self.config = config
        self.client = client_factory(config)
        self.session_preparer = session_preparer
        self.event_builder_factory = event_builder_factory
        self.own_user_id: int | None = None
        self.server: asyncio.AbstractServer | None = None

    async def _observe(self, event: Any) -> None:
        evidence = normalize_private_message(event, self.own_user_id or 0)
        if evidence is None:
            LOGGER.debug("telegram_user_event_ignored type=unsupported_or_non_private")
            return
        LOGGER.info("telegram_user_private_message %s", json.dumps(asdict(evidence), sort_keys=True))

    async def _control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        response: dict[str, Any]
        try:
            raw = await reader.readline()
            if not raw or len(raw) > MAX_CONTROL_REQUEST_BYTES:
                raise ValueError("invalid control request size")
            request = json.loads(raw)
            if not isinstance(request, dict) or request.get("action") != "send_private_text":
                raise ValueError("unsupported control action")
            response = await send_explicit_private_text(self.client, request.get("peer_id"), request.get("text"))
            LOGGER.info("telegram_user_explicit_send %s", json.dumps(response, sort_keys=True))
        except Exception as exc:
            response = {"ok": False, "error_type": type(exc).__name__, "error": "explicit send failed"}
            LOGGER.warning("telegram_user_explicit_send_failed error_type=%s", type(exc).__name__)
        writer.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self, stop_event: asyncio.Event) -> None:
        self.session_preparer(self.config.session_path)
        await self.client.connect()
        try:
            if not await self.client.is_user_authorized():
                raise RuntimeError("HUMAN_REQUIRED: run the one-time Telegram user authorization command")
            me = await self.client.get_me()
            self.own_user_id = validate_peer_id(getattr(me, "id", None), "own_user_id")
            self.client.add_event_handler(self._observe, self.event_builder_factory())
            socket = self.config.control_socket
            socket.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            if socket.exists():
                if not socket.is_socket():
                    raise RuntimeError("Control socket path exists and is not a socket")
                socket.unlink()
            self.server = await asyncio.start_unix_server(self._control, path=str(socket))
            os.chmod(socket, 0o600)
            LOGGER.info("telegram_user_ready own_user_id=%d session_path=%s", self.own_user_id, self.config.session_path)
            await stop_event.wait()
        finally:
            if self.server is not None:
                self.server.close()
                await self.server.wait_closed()
            if self.config.control_socket.exists() and self.config.control_socket.is_socket():
                self.config.control_socket.unlink()
            await self.client.disconnect()
            LOGGER.info("telegram_user_stopped")


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)


async def run_runtime(config: TelegramUserConfig | None = None) -> None:
    resolved = config or load_config()
    logging.basicConfig(level=getattr(logging, resolved.log_level), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)
    await TelegramUserRuntime(resolved).run(stop_event)


def main() -> None:
    asyncio.run(run_runtime())


if __name__ == "__main__":
    main()
