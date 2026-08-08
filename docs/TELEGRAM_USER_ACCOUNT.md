# Telegram user-account transport

This component is a separate Telethon 1.44.0 MTProto transport for the existing
Kven Telegram user identity. It does not import or call the Bot API gateway,
Kven inference, relationship storage, or automatic reply code. The existing
`kven2-telegram-gateway.service` remains independent and operational.

## Production contract

Install `Telethon==1.44.0` into `/opt/kven2/venv`. Install
`infra/telegram-user/systemd/kven2-telegram-user.service` as
`/etc/systemd/system/kven2-telegram-user.service`. Create
`/etc/kven2/telegram-user.env` as root-owned mode `0600` using the tracked
example. `TELEGRAM_USER_API_ID` and `TELEGRAM_USER_API_HASH` are the only
long-lived API credentials; never copy the Bot API token into these fields.

Create `/agent/data/kven2/telegram_user` as root-owned mode `0700`. The default
session is `/agent/data/kven2/telegram_user/kven.session`, a Telethon SQLite
session outside Git. The unit has `RequiresMountsFor=/agent/data`, so an absent
durable mount cannot silently create session state on the root filesystem.
Service stop disconnects cleanly and never logs out. The unit independently
creates `/run/kven2-telegram-user` as root-owned mode `0700`; it does not depend
on the Bot API service for runtime state. The local control socket is
`/run/kven2-telegram-user/control.sock`, mode `0600`.

## Controlled deployment and live acceptance (not for Codex execution)

Use a VM console and safe placeholders. Do not paste credentials into shell
history; write the root-only env file through the operator's approved secret
procedure.

1. Verify `stat -c '%U:%G %a' /etc/kven2/telegram-user.env` reports
   `root:root 600`, and verify the session directory reports `root:root 700`.
2. Run `/opt/kven2/venv/bin/python /opt/kven2/telegram_user_auth.py`. If the
   session is already authorized it exits without asking for phone, code, or
   password. Otherwise enter ephemeral prompts at the console. Record only the
   reported numeric own user ID.
3. Start only the new unit: `systemctl start kven2-telegram-user.service`.
   Confirm its sanitized ready record names `telegram_mtproto_user`, the own
   numeric ID, and the configured session path.
4. Send a distinctive private marker to the Kven user account. Use
   `journalctl -u kven2-telegram-user.service` to capture sender ID, peer ID,
   message ID, UTC timestamp, length, and SHA-256. Locally compute the expected
   UTF-8 SHA-256 for the marker and compare it. No message body is logged.
5. Send exactly one message with
   `/opt/kven2/venv/bin/python /opt/kven2/telegram_user_control.py OWNER_NUMERIC_ID 'DISTINCTIVE TEXT'`.
   Preserve the sanitized returned message ID/timestamp/digest and visually
   confirm the real Kven user identity sent it.
6. Restart only `kven2-telegram-user.service`. Confirm no interactive prompt,
   the same session path and own ID, then repeat one receive and one explicit
   send with new markers.
7. Confirm `systemctl is-active kven2-telegram-gateway.service`, then exercise
   its established text and vision acceptance paths. Evidence from this
   component always uses transport `telegram_mtproto_user`, distinguishing it
   from Bot API traffic.

## Rollback

Stop and disable only `kven2-telegram-user.service`, remove its installed unit
and `/etc/kven2/telegram-user.env` under the approved secret procedure, and
remove Telethon only if no other reviewed component depends on it. Preserve the
session directory and `.session` file so authorization remains reusable; do not
call `log_out()`. Revert the feature commit for tracked code. Do not change the
Bot API service, env file, database, or media directory.
