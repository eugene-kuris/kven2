"""Root-local operator command for one explicit Telegram private text send."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from telegram_user_runtime import DEFAULT_CONTROL_SOCKET


async def request_send(peer_id: int, text: str, *, socket_path: Path = DEFAULT_CONTROL_SOCKET) -> dict:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.write((json.dumps({"action": "send_private_text", "peer_id": peer_id, "text": text}) + "\n").encode())
    await writer.drain()
    response = json.loads(await reader.readline())
    writer.close()
    await writer.wait_closed()
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("peer_id", type=int, help="numeric Telegram user/peer ID")
    parser.add_argument("text", help="exact single message text")
    parser.add_argument("--socket", type=Path, default=DEFAULT_CONTROL_SOCKET)
    args = parser.parse_args()
    response = asyncio.run(request_send(args.peer_id, args.text, socket_path=args.socket))
    print(json.dumps(response, sort_keys=True))
    if not response.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
