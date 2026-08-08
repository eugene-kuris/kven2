# MAIL / QWENII / SMS infrastructure

Status: validated infrastructure capture for WORK-KVEN-OPS-004A. This closes
the infrastructure phase; the system is sufficient for its current purpose and
the project returns to product work.

## Architecture and trust boundaries

The external mail route is Internet/Gmail → mail.ssi.ua → VPN →
mail.kuris.kiev.ua → qwen@kuris.kiev.ua → the VM192 courier. Observed real Gmail
traffic passed SPF, DKIM, and DMARC at the mail boundary. Mail infrastructure
authenticates Internet mail; KVEN-BRIDGE validates protocol identity, state, and
artifacts; Linux DAC grants execution capability. VM192 deliberately does not
duplicate SPF/DKIM/DMARC checks. Plaintext SMTP/IMAP inside this isolated lab is
an accepted design and is not changed by this capture.

The courier at /opt/kven-mail-courier is transport only. It validates the sender,
subject and correlation, exactly one allowed attachment, size, and safe basename;
writes atomically; records size, SHA256, Message-ID and a durable receipt; and
sets automatic_execution=false. Delivery never authorizes execution:

    TRANSPORT ACCEPTANCE != EXECUTION AUTHORIZATION

The historical email-agent.service architecture (/opt/llm-agent/email_agent.py)
is rejected for KVEN-BRIDGE and is expected stopped and disabled. Its preserved
host source and configuration are intentionally not captured because they may
contain historical credential handling.

## QWENII authorization and execution

The dedicated qwenii account (observed UID 1002, GID 1003) has a locked password,
/bin/bash, no SSH keys, sudo, privileged supplementary groups, or production
repository access. The trusted inbox is root:qwenii 0750: qwenii can read a
root-delivered mode-0644 artifact but cannot replace it. Work and results are
qwenii-owned. Linux DAC is the capability boundary; there is no custom permission
engine. /opt is root:root 0755 and /opt/kven2 is root:root 0750. Never recursively
normalize these paths or grant qwenii production-repository access. Future source
development would use a separate qwenii-owned clone/worktree.

A courier payload is inert. Execution needs a second independently transported
and receipted JSON envelope with protocol KVEN-QWENII-TASK/1 and authorization
EXECUTE_AS_QWENII. The envelope binds task ID and correlation, the exact expected
payload filename, SHA256, and a bounded integer timeout. Its closed schema admits
no COMMAND field and no arbitrary natural-language command semantics.

VM192, not Gmail state, owns the durable lifecycle:

    RECEIVED → VALIDATED → RUNNING → SUCCEEDED / FAILED / TIMED_OUT → RETURNED

REJECTED is final for invalid input. Gmail unread/seen state, threads, and arrival
order are not workflow authority. Execution is at-most-once: RETURNED, REJECTED,
terminal, and RUNNING states cannot launch again. A restart converts an interrupted
RUNNING record to FAILED and renders a result instead of replaying possibly
side-effectful work. Terminal results are returned by the courier send helper;
failed mail return can retry without repeating execution.

## SMS escalation

qwenii can only queue NEED_USER, WORK_FAILED, or WORK_COMPLETE with a bounded work
ID under /home/qwenii/notify (qwenii:qwenii 0700). The privileged path-triggered
dispatcher validates owner, size, closed fields, vocabulary, and work-ID syntax.
It invokes the root notifier; qwenii has no unrestricted notifier access.
Duplicates are suppressed by SHA256(event + NUL + work_id).

The proven physical path is VM192 → authenticated HTTP → GoIP 192.168.1.110 → GSM
Line 3 → user phone. ens192 currently has secondary 192.168.1.192/24 on the same
L2 network. GoIP authentication and phone values remain only in protected config.
Physical receipt passed. Persistence of the secondary address must be verified at
the next ordinary reboot; do not reboot merely for this check.

## ChatGPT capability boundary

| Capability | Interactive | Scheduled |
| --- | --- | --- |
| Narrow Gmail read/search | supported | supported |
| Attachment retrieval | supported | supported |
| Exact reasoning/validation | supported | supported |
| Gmail send/write | supported | blocked by current platform |
| Write from immutable pre-created draft | not applicable | blocked |
| Write with standing authorization | not applicable | blocked |
| Autonomous wake | not applicable | supported |

The scheduled Gmail-write limitation is accepted. Do not work around it in this
infrastructure phase.

## Inventory and configuration

| Component | Source / unit | Expected state | Config | Durable state |
| --- | --- | --- | --- | --- |
| Mail courier | /opt/kven-mail-courier/receiver.py; kven-mail-courier.service | active, enabled | /etc/kven-mail-courier.json (0600) | /var/lib/kven-mail-courier |
| Mail result sender | /opt/kven-mail-courier/send_file.py; /usr/local/bin/kven-mail-send-file symlink | invoked | same | none |
| Legacy agent | email-agent.service | inactive, disabled | excluded | preserved live only |
| QWENII runner | /opt/kven-qwenii-runner/runner.py; kven-qwenii-runner.service | active, enabled | /etc/kven-qwenii-runner.json (0600) | /var/lib/kven-qwenii-runner |
| SMS request | /usr/local/bin/kven-request-human | invoked as qwenii | none | /home/qwenii/notify |
| SMS dispatch | kven-human-notify.path/service and two /usr/local/sbin executables | path active, enabled | /etc/kven-human-notifier/config (0600) | /var/lib/kven-human-notifier |

Repository examples are schemas, not usable credentials. Runtime state, receipts,
message IDs, mailbox content, execution records, SMS markers, and generated task
or result payloads must never be copied into Git.

## Reconstruction

This capture does not deploy. A future authorized operator should:

1. Create the locked qwenii identity and directories with the ownership/modes
   above; verify no sudo, keys, privileged groups, or /opt/kven2 access.
2. Install captured Python and shell sources at the inventory paths with the
   recorded modes (Python/request helper 0755; privileged notifier scripts 0750).
   Create the live-equivalent `/usr/local/bin/kven-mail-send-file` symlink
   pointing to `/opt/kven-mail-courier/send_file.py`; the runner's
   `send_helper` configuration depends on that exact path.
3. Build root-owned 0600 configs from the examples, supplying secrets through the
   protected operator channel; never commit or print them.
4. Create the root-owned state directories. The notifier hierarchy must include
   `/var/lib/kven-human-notifier/sent` and
   `/var/lib/kven-human-notifier/failed`, both root:root 0755, because the
   dispatcher writes duplicate markers and failed requests there. Install
   captured units, daemon-reload, enable/start only under separate deployment
   authority, and preserve the documented service ordering and sandbox paths.
5. Configure the secondary address through the host's existing network mechanism
   and verify same-L2 reachability without exposing GoIP credentials.

## Verification

Run the offline repository checks:

    /opt/kven2/venv/bin/python -m unittest -v test_mail_qwenii_sms_infra.py
    /opt/kven2/venv/bin/python scripts/verify_mail_qwenii_sms_capture.py --repository .

The verifier checks syntax, sanitized schemas, invariants, units, exclusions, and
optional byte fidelity against approved live non-secret files. It never reads
secret configs, runtime state, mailboxes, logs, databases, or /agent/data. Service
states may be checked separately with systemctl is-active/is-enabled; no live mail
or SMS is sent because prior empirical tests already proved both paths.

## Rollback/removal

For this repository-only capture, rollback is Git restoration to the recorded
baseline; no runtime rollback is needed. If a separately authorized installation
must later be removed, stop/disable only the four captured units, remove only the
explicit installed source/unit/config paths after protected backup, and preserve
runtime evidence unless the owner separately authorizes deletion. Never remove
legacy evidence, /agent/data, mailbox contents, or unrelated infrastructure.
