#!/usr/bin/env python3
"""Offline validation and optional approved-live fidelity for OPS-004A."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys


CAPTURED = {
    "infra/mail/receiver.py": "/opt/kven-mail-courier/receiver.py",
    "infra/mail/send_file.py": "/opt/kven-mail-courier/send_file.py",
    "infra/qwenii/runner.py": "/opt/kven-qwenii-runner/runner.py",
    "infra/notify/kven-human-notifier": "/usr/local/sbin/kven-human-notifier",
    "infra/notify/kven-request-human": "/usr/local/bin/kven-request-human",
    "infra/notify/kven-human-notify-dispatch": "/usr/local/sbin/kven-human-notify-dispatch",
    "infra/mail/systemd/kven-mail-courier.service": "/etc/systemd/system/kven-mail-courier.service",
    "infra/qwenii/systemd/kven-qwenii-runner.service": "/etc/systemd/system/kven-qwenii-runner.service",
    "infra/notify/systemd/kven-human-notify.path": "/etc/systemd/system/kven-human-notify.path",
    "infra/notify/systemd/kven-human-notify.service": "/etc/systemd/system/kven-human-notify.service",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--live-fidelity", action="store_true",
                        help="compare only the allowlisted non-secret live files")
    args = parser.parse_args(argv)
    root = Path(args.repository).resolve()
    failures = []

    for rel in ("infra/mail/receiver.py", "infra/mail/send_file.py", "infra/qwenii/runner.py"):
        try:
            ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
        except Exception as exc:
            failures.append(f"{rel}: syntax: {type(exc).__name__}")
    for rel in ("infra/mail/kven-mail-courier.example.json",
                "infra/qwenii/kven-qwenii-runner.example.json"):
        try:
            json.loads((root / rel).read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{rel}: json: {type(exc).__name__}")

    combined = "\n".join((root / rel).read_text(encoding="utf-8") for rel in CAPTURED)
    required = ["automatic_execution", "KVEN-QWENII-TASK/1", "EXECUTE_AS_QWENII",
                "NEED_USER", "WORK_FAILED", "WORK_COMPLETE"]
    for marker in required:
        if marker not in combined:
            failures.append(f"missing invariant marker: {marker}")
    if re.search(r"(?i)(password|token|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}", combined):
        failures.append("credential-like assignment in captured executable/unit source")

    forbidden_names = {"seen.json", "tasks", "sent", "failed", ".env"}
    for path in (root / "infra").rglob("*"):
        if path.name in forbidden_names:
            failures.append(f"private/runtime-state path captured: {path.relative_to(root)}")

    if args.live_fidelity:
        for rel, live_name in CAPTURED.items():
            live = Path(live_name)
            if not live.is_file():
                failures.append(f"approved live file missing: {live_name}")
            elif sha(root / rel) != sha(live):
                failures.append(f"live fidelity mismatch: {rel} != {live_name}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"capture_validation=PASS files={len(CAPTURED)} live_fidelity={args.live_fidelity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
