# MAIL, QWENII, and SMS infrastructure source capture

This directory is the sanitized repository source of truth for the validated
WORK-KVEN-MAIL-001A through 001D and WORK-KVEN-NOTIFY-001A infrastructure.
The executable and systemd files are byte-for-byte captures of the approved
non-secret live files observed on 2026-08-08. Configuration examples preserve
the keys consumed by those programs but contain documentation-only values.

Nothing here is an installer that mutates the host. Reconstruction is an
operator procedure documented in [the runbook](../docs/MAIL_QWENII_SMS_INFRASTRUCTURE.md).
Real configuration, mailbox data, receipts, execution history, notifier state,
and generated task/result artifacts are intentionally excluded.

Core invariant: `TRANSPORT ACCEPTANCE != EXECUTION AUTHORIZATION`.
The mail courier transports files and records `automatic_execution=false`; it
never executes them. Only a separately delivered, courier-receipted,
correlation- and SHA256-bound `KVEN-QWENII-TASK/1` authorization can reach the
narrow runner profile. There is no generic command or natural-language-to-shell
translation interface.
