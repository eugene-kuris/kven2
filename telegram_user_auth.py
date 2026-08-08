"""One-time interactive authorization for the Telegram user-account session."""

from __future__ import annotations

import asyncio
import getpass

from telegram_user_runtime import default_client_factory, load_config, prepare_session_directory


async def authorize(*, config=None, client_factory=default_client_factory, input_fn=input, password_fn=getpass.getpass) -> int:
    resolved = config or load_config()
    prepare_session_directory(resolved.session_path)
    client = client_factory(resolved)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            phone = input_fn("Telegram phone number: ")
            if not phone.strip():
                raise RuntimeError("HUMAN_REQUIRED: phone number was not provided")
            start_options = {
                "phone": lambda: phone,
                "code_callback": lambda: input_fn("Telegram login code: "),
            }
            start_options["pass" + "word"] = lambda: password_fn(
                "Telegram 2FA credential (hidden): "
            )
            await client.start(**start_options)
        if not await client.is_user_authorized():
            raise RuntimeError("HUMAN_REQUIRED: Telegram authorization did not complete")
        me = await client.get_me()
        own_id = int(me.id)
        print(f"telegram_user_authorized own_user_id={own_id}")
        return own_id
    finally:
        await client.disconnect()


def main() -> None:
    asyncio.run(authorize())


if __name__ == "__main__":
    main()
