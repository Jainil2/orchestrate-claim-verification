#!/usr/bin/env python3
"""UserPromptSubmit hook: append the verbatim user prompt to the AGENTS.md log.

Reads the hook's JSON payload on stdin, writes a §5.2-style entry to
$HOME/hackerrank_orchestrate/log.txt (Windows: %USERPROFILE%). Captures the
verbatim prompt automatically so log entries are never paraphrased; the agent
appends its Response Summary / Actions under the entry during the turn.

Never blocks the turn: any error exits 0 silently.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{8,})"
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return
    prompt = SECRET_RE.sub("[REDACTED]", prompt)

    cwd = payload.get("cwd") or os.getcwd()
    try:
        branch = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        branch = "unknown"

    ts = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    log_path = Path.home() / "hackerrank_orchestrate" / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    title = prompt.splitlines()[0][:72]
    entry = (
        f"\n## [{ts}] {title}\n\n"
        f"User Prompt (verbatim, secrets redacted):\n{prompt}\n\n"
        f"Context:\n"
        f"tool=Claude Code\n"
        f"branch={branch}\n"
        f"repo_root={cwd}\n"
        f"worktree=main\n"
        f"parent_agent=none\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # never block the user's turn
