#!/usr/bin/env python3
import email
import hashlib
import imaplib
import json
import logging
import os
import re
import tempfile
import time
from email.header import decode_header, make_header
from email.utils import parseaddr
from pathlib import Path

CONFIG_PATH = Path("/etc/kven-mail-courier.json")
STATE_PATH = Path("/var/lib/kven-mail-courier/seen.json")
SUBJECT_RE = re.compile(r"^\[KVEN-BRIDGE\] FILE ([A-Za-z0-9._:-]{8,128})$")
ALLOWED_SUFFIXES = (
    ".sh", ".txt", ".md", ".json", ".yaml", ".yml",
    ".tar.gz", ".tgz", ".zip"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("kven-mail-courier")

def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def load_state():
    if not STATE_PATH.exists():
        return {"seen_message_ids": []}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("seen_message_ids"), list):
            raise ValueError("invalid state")
        return data
    except Exception as exc:
        log.error("state_read_failed=%s", type(exc).__name__)
        return {"seen_message_ids": []}

def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".seen.", dir=str(STATE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, STATE_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def decode_header_value(value):
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value

def safe_filename(name):
    name = decode_header_value(name or "")
    if not name or name != os.path.basename(name):
        return None
    if name.startswith(".") or "/" in name or "\\" in name or "\x00" in name:
        return None
    if len(name) > 180:
        return None
    lower = name.lower()
    if not any(lower.endswith(sfx) for sfx in ALLOWED_SUFFIXES):
        return None
    return name

def plain_body(msg):
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_type() != "text/plain":
            continue
        if "attachment" in str(part.get("Content-Disposition", "")).lower():
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    return ""

def attachment_parts(msg):
    result = []
    for part in msg.walk():
        cd = str(part.get("Content-Disposition", "")).lower()
        if "attachment" not in cd:
            continue
        result.append(part)
    return result

def atomic_write(path, payload, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def process_once(cfg, state):
    mail = imaplib.IMAP4(cfg["IMAP_HOST"], int(cfg["IMAP_PORT"]))
    accepted = cfg["accepted_sender"].lower()
    inbox_dir = Path(cfg["inbox_dir"])
    max_bytes = int(cfg["max_attachment_bytes"])
    seen = set(state["seen_message_ids"])
    changed = False

    try:
        mail.login(cfg["EMAIL_USER"], cfg["EMAIL_PASS"])
        status, _ = mail.select("inbox", readonly=True)
        if status != "OK":
            raise RuntimeError("imap_select_failed")

        status, data = mail.search(
            None,
            "FROM", f'"{cfg["accepted_sender"]}"',
            "SUBJECT", '"[KVEN-BRIDGE] FILE "'
        )
        if status != "OK":
            raise RuntimeError("imap_search_failed")

        ids = data[0].split()
        for imap_id in ids:
            status, msg_data = mail.fetch(imap_id, "(BODY.PEEK[])")
            if status != "OK":
                continue
            raw = next((x[1] for x in msg_data if isinstance(x, tuple) and len(x) > 1), None)
            if not raw:
                continue

            msg = email.message_from_bytes(raw)
            message_id = decode_header_value(msg.get("Message-ID")) or f"imap:{imap_id.decode()}"
            if message_id in seen:
                continue

            sender = parseaddr(decode_header_value(msg.get("From")))[1].lower()
            subject = decode_header_value(msg.get("Subject"))
            match = SUBJECT_RE.fullmatch(subject)
            if sender != accepted or not match:
                continue

            corr = match.group(1)
            body = plain_body(msg)
            if f"Correlation-ID: {corr}" not in body:
                log.warning("reject correlation_body_mismatch corr=%s", corr)
                seen.add(message_id); changed = True
                continue
            if "Purpose: file transport only" not in body:
                log.warning("reject purpose_missing corr=%s", corr)
                seen.add(message_id); changed = True
                continue

            parts = attachment_parts(msg)
            if len(parts) != 1:
                log.warning("reject attachment_count=%s corr=%s", len(parts), corr)
                seen.add(message_id); changed = True
                continue

            part = parts[0]
            filename = safe_filename(part.get_filename())
            if not filename:
                log.warning("reject unsafe_filename corr=%s", corr)
                seen.add(message_id); changed = True
                continue

            payload = part.get_payload(decode=True)
            if payload is None or len(payload) == 0 or len(payload) > max_bytes:
                log.warning("reject attachment_size=%s corr=%s", 0 if payload is None else len(payload), corr)
                seen.add(message_id); changed = True
                continue

            sha = hashlib.sha256(payload).hexdigest()
            dest = inbox_dir / filename
            if dest.exists():
                existing_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
                if existing_sha != sha:
                    log.warning("reject filename_conflict file=%s corr=%s", filename, corr)
                    seen.add(message_id); changed = True
                    continue
            else:
                atomic_write(dest, payload, 0o644)

            receipt = {
                "correlation_id": corr,
                "sender": sender,
                "subject": subject,
                "message_id": message_id,
                "filename": filename,
                "path": str(dest),
                "size": len(payload),
                "sha256": sha,
                "saved_mode": "0644",
                "automatic_execution": False,
            }
            receipt_path = inbox_dir / f"{filename}.receipt.json"
            atomic_write(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode(), 0o644)
            seen.add(message_id); changed = True
            log.info("received corr=%s file=%s size=%s sha256=%s", corr, filename, len(payload), sha)

        if changed:
            state["seen_message_ids"] = list(seen)[-5000:]
            save_state(state)
    finally:
        try:
            mail.logout()
        except Exception:
            pass

def main():
    cfg = load_config()
    state = load_state()
    log.info(
        "started accepted_sender=%s inbox=%s poll_seconds=%s auto_execute=no",
        cfg["accepted_sender"], cfg["inbox_dir"], cfg["poll_seconds"]
    )
    while True:
        try:
            process_once(cfg, state)
        except Exception as exc:
            log.error("poll_failed=%s", type(exc).__name__)
        time.sleep(int(cfg["poll_seconds"]))

if __name__ == "__main__":
    main()
