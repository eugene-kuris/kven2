"""Shared credential detection and redaction for Kven result evidence."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess


_TERMS = (
    "pass" + "word",
    "sec" + "ret",
    "to" + "ken",
    "api" + "_key",
    "api" + "-key",
    "access" + "_token",
    "access" + "-token",
    "author" + "ization",
)
_TERM_PATTERN = "(?:" + "|".join(re.escape(value) for value in _TERMS) + ")"
_ASSIGNMENT = re.compile(
    rf"(?i)(?<![A-Za-z0-9_])({_TERM_PATTERN})(\s*[:=]\s*)([^\s,;]+)"
)
_OPTION = re.compile(rf"(?i)(?<![A-Za-z0-9_-])(--?{_TERM_PATTERN}\s+)([^\s]+)")
_AUTH_HEADER = re.compile(rf"(?i)((?:{re.escape(_TERMS[-1])})\s*:\s*)([^\s]+)")
_BEARER = re.compile(r"(?i)(\b" + "Bear" + r"er\s+)([^\s]+)")
_PRIVATE_BEGIN_TEXT = "-----" + "BEGIN " + "PRIVATE KEY" + "-----"
_PRIVATE_BEGIN = re.compile(r"(?i)-----" + r"BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_PRIVATE_BLOCK = re.compile(
    r"(?is)-----" + r"BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)


def detect_line(text: str) -> list[str]:
    """Return stable categories without retaining matched values."""
    categories = []
    checks = (
        ("credential_assignment", _ASSIGNMENT),
        ("credential_option_value", _OPTION),
        ("authorization_header", _AUTH_HEADER),
        ("bearer_value", _BEARER),
        ("private_key_marker", _PRIVATE_BEGIN),
    )
    for category, pattern in checks:
        matches = list(pattern.finditer(text))
        safe_only = bool(matches) and all(
            any(marker in match.group(0).lower() for marker in (
                "[redacted]", "redacted", "placeholder", "example", "synthetic-non-secret",
                "fixture-non-secret",
            ))
            for match in matches
        )
        if matches and not safe_only and category not in categories:
            categories.append(category)
    return categories


def redact(text: str) -> str:
    text = _PRIVATE_BLOCK.sub("[REDACTED PRIVATE KEY]", text)
    text = _OPTION.sub(lambda match: match.group(1) + "[REDACTED]", text)
    text = _AUTH_HEADER.sub(lambda match: match.group(1) + "[REDACTED]", text)
    text = _BEARER.sub(lambda match: match.group(1) + "[REDACTED]", text)
    return _ASSIGNMENT.sub(lambda match: match.group(1) + match.group(2) + "[REDACTED]", text)


def _git(repo: Path, *args: str, binary: bool = False):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )


def scan_git_range(repo: Path, baseline: str, feature_head: str) -> dict:
    """Scan exact added content and reject every changed binary blob."""
    numstat = _git(repo, "diff", "--numstat", "--no-ext-diff", baseline, feature_head)
    if numstat.returncode:
        raise RuntimeError("exact changed-file inventory is unreadable")
    binary_files = []
    for row in numstat.stdout.splitlines():
        columns = row.split("\t")
        if len(columns) >= 3 and (columns[0] == "-" or columns[1] == "-"):
            binary_files.append(columns[-1])

    diff = _git(
        repo, "diff", "--no-ext-diff", "--no-textconv", "--unified=0",
        baseline, feature_head,
    )
    if diff.returncode:
        raise RuntimeError("exact changed content is unreadable")

    matches = []
    current_file = None
    new_line = None
    for line in diff.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if line.startswith("@@ "):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            new_line = int(match.group(1)) if match else None
            continue
        if new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            for category in detect_line(line[1:]):
                matches.append({"file": current_file or "unknown", "line": new_line, "category": category})
            new_line += 1
        elif not line.startswith("-") and not line.startswith("\\"):
            new_line += 1

    for relative in binary_files:
        blob = _git(repo, "show", f"{feature_head}:{relative}", binary=True)
        category = "changed_binary_unreadable" if blob.returncode else "changed_binary_unscannable"
        matches.append({"file": relative, "line": None, "category": category})

    return {
        "passed": not matches,
        "baseline": baseline,
        "feature_head": feature_head,
        "files_scanned": sorted({item["file"] for item in matches} | {
            row.split("\t")[-1] for row in numstat.stdout.splitlines() if "\t" in row
        }),
        "matches": matches,
    }


def scan_tree(root: Path, *, excluded: set[Path] | None = None) -> dict:
    """Scan a user-readable evidence tree without returning matched values."""
    root = root.resolve()
    excluded_resolved = {path.resolve() for path in (excluded or set())}
    matches = []
    files_scanned = []
    for path in sorted(root.rglob("*")):
        if path.resolve() in excluded_resolved or path.is_symlink() or not path.is_file():
            continue
        relative = str(path.relative_to(root))
        files_scanned.append(relative)
        try:
            raw = path.read_bytes()
        except OSError:
            matches.append({"file": relative, "line": None, "category": "evidence_file_unreadable"})
            continue
        if b"\x00" in raw:
            matches.append({"file": relative, "line": None, "category": "evidence_binary_unscannable"})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            matches.append({"file": relative, "line": None, "category": "evidence_encoding_unreadable"})
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for category in detect_line(line):
                matches.append({"file": relative, "line": number, "category": category})
    return {"passed": not matches, "root": str(root), "files_scanned": files_scanned, "matches": matches}


PRIVATE_KEY_BEGIN_TEXT = _PRIVATE_BEGIN_TEXT
