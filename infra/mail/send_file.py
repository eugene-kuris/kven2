#!/usr/bin/env python3
import email.utils
import hashlib
import json
import mimetypes
import os
import re
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

CONFIG_PATH = Path("/etc/kven-mail-courier.json")
CORR_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")

def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: kven-mail-send-file <path> <correlation-id>")
    path = Path(sys.argv[1]).resolve()
    corr = sys.argv[2]
    if not CORR_RE.fullmatch(corr):
        raise SystemExit("invalid correlation id")
    if not path.is_file():
        raise SystemExit("file not found")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    if len(payload) == 0 or len(payload) > int(cfg["max_attachment_bytes"]):
        raise SystemExit("file size outside allowed range")

    sha = hashlib.sha256(payload).hexdigest()
    msg = EmailMessage()
    msg["From"] = cfg["EMAIL_USER"]
    msg["To"] = cfg["outbound_recipient"]
    msg["Date"] = email.utils.formatdate(localtime=False)
    msg["Message-ID"] = email.utils.make_msgid(domain="kuris.kiev.ua")
    msg["Subject"] = f"[KVEN-BRIDGE] FILE-RESULT {corr}"
    msg["X-Kven-Bridge-Correlation"] = corr
    msg.set_content(
        "KVEN-BRIDGE file transport artifact.\n"
        f"Correlation-ID: {corr}\n"
        "Purpose: file transport only\n"
        f"Filename: {path.name}\n"
        f"Size: {len(payload)}\n"
        f"SHA256: {sha}\n"
        "Semantics: transport only; no PASS implied.\n"
    )
    ctype, _ = mimetypes.guess_type(path.name)
    maintype, subtype = ("application", "octet-stream")
    if ctype and "/" in ctype:
        maintype, subtype = ctype.split("/", 1)
    msg.add_attachment(payload, maintype=maintype, subtype=subtype, filename=path.name)

    with smtplib.SMTP(cfg["SMTP_HOST"], int(cfg["SMTP_PORT"]), timeout=30) as server:
        server.ehlo()
        server.login(cfg["EMAIL_USER"], cfg["EMAIL_PASS"])
        refused = server.send_message(msg)
        if refused:
            raise SystemExit("smtp refused recipient")

    print(f"mail_file_send=PASS")
    print(f"correlation_id={corr}")
    print(f"filename={path.name}")
    print(f"size={len(payload)}")
    print(f"sha256={sha}")
    print("smtp_transport=plain")

if __name__ == "__main__":
    main()
